from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetectorConfig:
    """Configuration for the reusable MediaPipe face detector."""

    model_selection: int = 1
    min_detection_confidence: float = 0.50

    def __post_init__(self) -> None:
        if self.model_selection not in {0, 1}:
            raise ValueError("model_selection must be 0 (short range) or 1 (full range).")
        if not 0.0 < self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in the interval (0, 1].")


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


class MediaPipeFaceDetector:
    """Reusable MediaPipe face detector for batch frame analysis.

    A single detector instance is kept open across a full batch. This avoids the
    overhead of constructing a new MediaPipe graph for every sampled frame.
    """

    def __init__(self, config: FaceDetectorConfig | None = None) -> None:
        self.config = config or FaceDetectorConfig()
        self._detector = None

    def open(self) -> "MediaPipeFaceDetector":
        """Initialize the underlying MediaPipe detector."""
        if self._detector is not None:
            return self

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("Install mediapipe with: pip install mediapipe") from exc

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            raise RuntimeError(
                "This MediaPipe installation does not expose mp.solutions.face_detection. "
                "Install a compatible mediapipe>=0.10 release."
            )

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self.config.model_selection,
            min_detection_confidence=self.config.min_detection_confidence,
        )
        return self

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "MediaPipeFaceDetector":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect faces and return normalized, clipped bounding boxes."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("frame_bgr must be a valid BGR image array.")

        if self._detector is None:
            self.open()

        rgb = np.ascontiguousarray(frame_bgr[:, :, :3][:, :, ::-1])
        result = self._detector.process(rgb)

        faces: list[dict] = []
        if not result.detections:
            return faces

        for detection in result.detections:
            relative_bbox = detection.location_data.relative_bounding_box

            x1 = max(0.0, float(relative_bbox.xmin))
            y1 = max(0.0, float(relative_bbox.ymin))
            x2 = min(1.0, float(relative_bbox.xmin + relative_bbox.width))
            y2 = min(1.0, float(relative_bbox.ymin + relative_bbox.height))

            if x2 <= x1 or y2 <= y1:
                continue

            face = {
                "xmin": x1,
                "ymin": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(detection.score[0]),
            }
            face["area_ratio"] = face_area_ratio(face)
            face["center_distance"] = face_center_distance(face)
            faces.append(face)

        return sorted(faces, key=face_area_ratio, reverse=True)


def detect_faces_mediapipe(
    frame: np.ndarray,
    model_selection: int = 1,
    min_detection_confidence: float = 0.50,
) -> list[dict]:
    """Backward-compatible convenience wrapper for one-off face detection."""
    config = FaceDetectorConfig(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )
    with MediaPipeFaceDetector(config) as detector:
        return detector.detect(frame)
