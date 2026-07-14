from __future__ import annotations

import math
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airtime_bias.io.paths import INTERIM_DIR


SHORT_RANGE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
DEFAULT_MODEL_PATH = INTERIM_DIR / "models" / "blaze_face_short_range.tflite"


@dataclass(frozen=True)
class FaceDetectorConfig:
    """Configuration for the reusable MediaPipe face detector."""

    model_selection: int = 0
    min_detection_confidence: float = 0.50
    min_suppression_threshold: float = 0.30
    model_path: str | None = None
    auto_download_model: bool = True

    def __post_init__(self) -> None:
        if self.model_selection not in {0, 1}:
            raise ValueError("model_selection must be 0 (short range) or 1 (full range).")
        if not 0.0 < self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in the interval (0, 1].")
        if not 0.0 <= self.min_suppression_threshold <= 1.0:
            raise ValueError("min_suppression_threshold must be in the interval [0, 1].")


def face_area_ratio(face: dict) -> float:
    """Return normalized bounding-box area for a detected face."""
    return max(0.0, float(face.get("width", 0.0))) * max(
        0.0, float(face.get("height", 0.0))
    )


def face_center_distance(face: dict) -> float:
    """Return Euclidean distance from face center to normalized frame center."""
    center_x = float(face.get("xmin", 0.0)) + float(face.get("width", 0.0)) / 2.0
    center_y = float(face.get("ymin", 0.0)) + float(face.get("height", 0.0)) / 2.0
    return math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)


def select_main_face(faces: list[dict]) -> dict | None:
    """Return the largest detected face, or None when no face is present."""
    if not faces:
        return None
    return max(faces, key=face_area_ratio)


def _load_legacy_face_detection_module(mp):
    """Return the legacy Face Detection module when the wheel still exposes it."""
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_detection"):
        return solutions.face_detection

    try:
        from mediapipe.python.solutions import face_detection
    except (ImportError, ModuleNotFoundError):
        return None
    return face_detection


def _download_model(destination: Path) -> Path:
    """Download the official MediaPipe short-range BlazeFace model atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")

    request = urllib.request.Request(
        SHORT_RANGE_MODEL_URL,
        headers={"User-Agent": "airtime-bias/0.1"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        temporary_path.replace(destination)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "MediaPipe Tasks is available, but the face detector model could not be "
            f"downloaded to '{destination}'. Download the official BlazeFace short-range "
            "model manually and set FaceDetectorConfig(model_path=...)."
        ) from exc

    return destination


class MediaPipeFaceDetector:
    """Reusable face detector supporting legacy and current MediaPipe Python APIs.

    Newer MediaPipe wheels may no longer expose ``mp.solutions``. In that case,
    Airtime Bias automatically uses the current MediaPipe Tasks Face Detector API
    and caches the official BlazeFace short-range model under ``data/interim/models``.
    """

    def __init__(self, config: FaceDetectorConfig | None = None) -> None:
        self.config = config or FaceDetectorConfig()
        self._detector = None
        self._mp = None
        self._backend: str | None = None
        self._model_variant: str | None = None

    @property
    def backend_name(self) -> str | None:
        return self._backend

    @property
    def model_variant(self) -> str | None:
        return self._model_variant

    def _resolve_tasks_model_path(self) -> Path:
        if self.config.model_path:
            model_path = Path(self.config.model_path).expanduser().resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"MediaPipe face detector model not found: {model_path}")
            return model_path

        model_path = DEFAULT_MODEL_PATH
        if model_path.exists():
            return model_path
        if not self.config.auto_download_model:
            raise FileNotFoundError(
                f"MediaPipe face detector model not found: {model_path}. "
                "Enable auto_download_model or provide model_path."
            )
        return _download_model(model_path)

    def _open_tasks_detector(self, mp) -> None:
        try:
            from mediapipe.tasks import python as tasks_python
            from mediapipe.tasks.python import vision
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "This MediaPipe installation exposes neither the legacy Face Detection "
                "solution nor the current MediaPipe Tasks Face Detector API."
            ) from exc

        model_path = self._resolve_tasks_model_path()
        options = vision.FaceDetectorOptions(
            base_options=tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.config.min_detection_confidence,
            min_suppression_threshold=self.config.min_suppression_threshold,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._backend = "mediapipe_tasks"
        self._model_variant = "blaze_face_short_range"

    def open(self) -> "MediaPipeFaceDetector":
        """Initialize the best MediaPipe face detector API available."""
        if self._detector is not None:
            return self

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("Install mediapipe with: pip install mediapipe") from exc

        self._mp = mp
        legacy_face_detection = _load_legacy_face_detection_module(mp)
        if legacy_face_detection is not None:
            self._detector = legacy_face_detection.FaceDetection(
                model_selection=self.config.model_selection,
                min_detection_confidence=self.config.min_detection_confidence,
            )
            self._backend = "mediapipe_legacy_solutions"
            self._model_variant = (
                "full_range" if self.config.model_selection == 1 else "short_range"
            )
            return self

        self._open_tasks_detector(mp)
        return self

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
        self._detector = None
        self._mp = None
        self._backend = None
        self._model_variant = None

    def __enter__(self) -> "MediaPipeFaceDetector":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _normalize_face(
        xmin: float,
        ymin: float,
        width: float,
        height: float,
        confidence: float,
    ) -> dict | None:
        x1 = max(0.0, float(xmin))
        y1 = max(0.0, float(ymin))
        x2 = min(1.0, float(xmin + width))
        y2 = min(1.0, float(ymin + height))
        if x2 <= x1 or y2 <= y1:
            return None

        face = {
            "xmin": x1,
            "ymin": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": float(confidence),
        }
        face["area_ratio"] = face_area_ratio(face)
        face["center_distance"] = face_center_distance(face)
        return face

    def _detect_legacy(self, rgb: np.ndarray) -> list[dict]:
        result = self._detector.process(rgb)
        faces: list[dict] = []
        for detection in result.detections or []:
            bbox = detection.location_data.relative_bounding_box
            face = self._normalize_face(
                xmin=bbox.xmin,
                ymin=bbox.ymin,
                width=bbox.width,
                height=bbox.height,
                confidence=detection.score[0],
            )
            if face is not None:
                faces.append(face)
        return faces

    def _detect_tasks(self, rgb: np.ndarray) -> list[dict]:
        height, width = rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        faces: list[dict] = []
        for detection in result.detections or []:
            bbox = detection.bounding_box
            categories = detection.categories or []
            confidence = float(categories[0].score) if categories else 0.0
            face = self._normalize_face(
                xmin=float(bbox.origin_x) / width,
                ymin=float(bbox.origin_y) / height,
                width=float(bbox.width) / width,
                height=float(bbox.height) / height,
                confidence=confidence,
            )
            if face is not None:
                faces.append(face)
        return faces

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect faces and return normalized, clipped bounding boxes."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("frame_bgr must be a valid BGR image array.")
        if self._detector is None:
            self.open()

        rgb = np.ascontiguousarray(frame_bgr[:, :, :3][:, :, ::-1])
        if self._backend == "mediapipe_legacy_solutions":
            faces = self._detect_legacy(rgb)
        elif self._backend == "mediapipe_tasks":
            faces = self._detect_tasks(rgb)
        else:
            raise RuntimeError("Face detector backend was not initialized correctly.")

        return sorted(faces, key=face_area_ratio, reverse=True)


def detect_faces_mediapipe(
    frame: np.ndarray,
    model_selection: int = 0,
    min_detection_confidence: float = 0.50,
) -> list[dict]:
    """Backward-compatible convenience wrapper for one-off face detection."""
    config = FaceDetectorConfig(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )
    with MediaPipeFaceDetector(config) as detector:
        return detector.detect(frame)
