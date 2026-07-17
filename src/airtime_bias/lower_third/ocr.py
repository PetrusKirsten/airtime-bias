from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from airtime_bias.io.paths import INTERIM_DIR


def normalize_ocr_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _window_similarity(name: str, text: str) -> float:
    if not name or not text:
        return 0.0
    name_tokens = name.split()
    text_tokens = text.split()
    if not name_tokens or not text_tokens:
        return 0.0
    window_sizes = {len(name_tokens), max(1, len(name_tokens) - 1), len(name_tokens) + 1}
    best = 0.0
    for window_size in window_sizes:
        if window_size <= 0:
            continue
        if len(text_tokens) < window_size:
            candidates = [" ".join(text_tokens)]
        else:
            candidates = [
                " ".join(text_tokens[index : index + window_size])
                for index in range(len(text_tokens) - window_size + 1)
            ]
        for candidate in candidates:
            best = max(best, SequenceMatcher(None, name, candidate).ratio())
    return float(best)


def match_participant_name(raw_text: object, participants: list[str]) -> tuple[str | None, float]:
    """Match imperfect OCR output against the closed list of active participants."""
    normalized_text = normalize_ocr_text(raw_text)
    if not normalized_text:
        return None, 0.0

    best_participant: str | None = None
    best_score = 0.0
    for participant in participants:
        normalized_name = normalize_ocr_text(participant)
        if not normalized_name:
            continue
        if normalized_name in normalized_text:
            score = 1.0
        else:
            full_score = SequenceMatcher(None, normalized_name, normalized_text).ratio()
            window_score = _window_similarity(normalized_name, normalized_text)
            first_name = normalized_name.split()[0]
            first_name_score = 0.0
            if len(first_name) >= 4:
                first_name_score = max(
                    SequenceMatcher(None, first_name, token).ratio()
                    for token in normalized_text.split()
                )
                first_name_score *= 0.90
            score = max(full_score, window_score, first_name_score)
        if score > best_score:
            best_participant = str(participant)
            best_score = float(score)
    return best_participant, best_score


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class EasyOCRConfig:
    languages: tuple[str, ...] = ("pt", "en")
    gpu: bool = False
    min_text_confidence: float = 0.15
    model_storage_directory: str | None = None


class EasyOCRReader:
    """Lazy reusable EasyOCR reader for positive lower-third crops only."""

    def __init__(self, config: EasyOCRConfig | None = None) -> None:
        self.config = config or EasyOCRConfig()
        self._reader = None

    def open(self) -> "EasyOCRReader":
        if self._reader is not None:
            return self
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "EasyOCR is optional. Install it with: pip install -e \".[ocr]\""
            ) from exc

        model_dir = (
            Path(self.config.model_storage_directory)
            if self.config.model_storage_directory
            else INTERIM_DIR / "models" / "easyocr"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        self._reader = easyocr.Reader(
            list(self.config.languages),
            gpu=self.config.gpu,
            model_storage_directory=str(model_dir),
            download_enabled=True,
        )
        return self

    def close(self) -> None:
        self._reader = None

    def __enter__(self) -> "EasyOCRReader":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("OCR crop is empty.")
        scale = max(1.0, 1200.0 / max(crop_bgr.shape[1], 1))
        enlarged = cv2.resize(
            crop_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    def read(self, crop_bgr: np.ndarray) -> tuple[str, float, list[dict]]:
        if self._reader is None:
            self.open()
        prepared = self._preprocess(crop_bgr)
        result = self._reader.readtext(prepared, detail=1, paragraph=False)
        rows: list[dict] = []
        texts: list[str] = []
        confidences: list[float] = []
        for box, text, confidence in result:
            confidence = float(confidence)
            if confidence < self.config.min_text_confidence:
                continue
            texts.append(str(text))
            confidences.append(confidence)
            rows.append({"box": box, "text": str(text), "confidence": confidence})
        raw_text = " | ".join(texts)
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        return raw_text, mean_confidence, rows


def annotate_events_with_ocr(
    events: pd.DataFrame,
    *,
    participants: list[str],
    reader: EasyOCRReader,
    minimum_name_similarity: float = 0.68,
) -> pd.DataFrame:
    """Run OCR only on representative positive crops and match names to active cast."""
    if events.empty:
        return events.copy()

    output = events.copy()
    output["ocr_raw_text"] = output.get("ocr_raw_text", None)
    output["ocr_mean_confidence"] = output.get("ocr_mean_confidence", None)
    output["matched_participant"] = output.get("matched_participant", None)
    output["participant_name_similarity"] = output.get("participant_name_similarity", None)
    output["ocr_error"] = output.get("ocr_error", None)

    with reader:
        for index, event in output.iterrows():
            crop_path_value = event.get("representative_roi_crop_path")
            if not isinstance(crop_path_value, str) or not crop_path_value:
                output.at[index, "ocr_error"] = "representative_roi_crop_missing"
                continue
            crop_path = Path(crop_path_value)
            crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
            if crop is None:
                output.at[index, "ocr_error"] = "could_not_read_representative_roi_crop"
                continue
            try:
                raw_text, mean_confidence, _ = reader.read(crop)
                participant, similarity = match_participant_name(raw_text, participants)
                output.at[index, "ocr_raw_text"] = raw_text
                output.at[index, "ocr_mean_confidence"] = mean_confidence
                output.at[index, "participant_name_similarity"] = similarity
                output.at[index, "matched_participant"] = (
                    participant if similarity >= minimum_name_similarity else None
                )
            except Exception as exc:
                output.at[index, "ocr_error"] = f"ocr_failed: {exc}"
    return output
