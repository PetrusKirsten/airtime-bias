from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LowerThirdROI:
    """Normalized lower-left region where participant lower-thirds are expected."""

    xmin: float = 0.02
    ymin: float = 0.70
    xmax: float = 0.68
    ymax: float = 0.96

    def __post_init__(self) -> None:
        values = (self.xmin, self.ymin, self.xmax, self.ymax)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Lower-third ROI coordinates must be between 0 and 1.")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Lower-third ROI xmax/ymax must exceed xmin/ymin.")


@dataclass(frozen=True)
class LowerThirdVisualConfig:
    """Interpretable visual detector calibrated for a copper lower-third layout.

    The detector deliberately combines weak signals instead of relying on one exact
    template. This makes it usable without storing copyrighted screenshots in the
    repository and keeps every score component auditable in the output table.
    """

    roi: LowerThirdROI = LowerThirdROI()
    analysis_width: int = 640
    visual_threshold: float = 0.50

    warm_hue_low_1: int = 0
    warm_hue_high_1: int = 25
    warm_hue_low_2: int = 165
    warm_hue_high_2: int = 179
    warm_saturation_min: int = 35
    warm_saturation_max: int = 230
    warm_value_min: int = 35
    warm_value_max: int = 245

    weight_rectangle: float = 0.30
    weight_edge_density: float = 0.20
    weight_text_density: float = 0.20
    weight_text_components: float = 0.15
    weight_logo_circularity: float = 0.10
    weight_warm_row: float = 0.05

    def __post_init__(self) -> None:
        if self.analysis_width < 160:
            raise ValueError("analysis_width must be at least 160 pixels.")
        if not 0.0 <= self.visual_threshold <= 1.0:
            raise ValueError("visual_threshold must be between 0 and 1.")
        weights = (
            self.weight_rectangle,
            self.weight_edge_density,
            self.weight_text_density,
            self.weight_text_components,
            self.weight_logo_circularity,
            self.weight_warm_row,
        )
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("Lower-third visual score weights must sum to 1.0.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["roi"] = asdict(self.roi)
        return data


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def crop_normalized_roi(frame_bgr: np.ndarray, roi: LowerThirdROI) -> np.ndarray:
    """Crop a normalized ROI from a BGR image."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        raise ValueError("frame_bgr must be a valid BGR image array.")

    height, width = frame_bgr.shape[:2]
    x1 = int(round(roi.xmin * width))
    y1 = int(round(roi.ymin * height))
    x2 = int(round(roi.xmax * width))
    y2 = int(round(roi.ymax * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return frame_bgr[y1:y2, x1:x2].copy()


def _resize_for_analysis(crop: np.ndarray, width: int) -> np.ndarray:
    if crop.shape[1] == width:
        return crop
    scale = width / crop.shape[1]
    height = max(1, int(round(crop.shape[0] * scale)))
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


def _warm_mask(hsv: np.ndarray, config: LowerThirdVisualConfig) -> np.ndarray:
    lower_1 = np.array(
        [config.warm_hue_low_1, config.warm_saturation_min, config.warm_value_min],
        dtype=np.uint8,
    )
    upper_1 = np.array(
        [config.warm_hue_high_1, config.warm_saturation_max, config.warm_value_max],
        dtype=np.uint8,
    )
    lower_2 = np.array(
        [config.warm_hue_low_2, config.warm_saturation_min, config.warm_value_min],
        dtype=np.uint8,
    )
    upper_2 = np.array(
        [config.warm_hue_high_2, config.warm_saturation_max, config.warm_value_max],
        dtype=np.uint8,
    )
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower_1, upper_1),
        cv2.inRange(hsv, lower_2, upper_2),
    )


def _warm_rectangle_score(mask: np.ndarray) -> tuple[float, dict | None]:
    height, width = mask.shape
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    best_box: dict | None = None

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue
        aspect_ratio = box_width / box_height
        area_ratio = (box_width * box_height) / (width * height)
        fill_ratio = cv2.contourArea(contour) / max(box_width * box_height, 1)
        vertical_center = (y + box_height / 2.0) / height

        score = (
            _clamp01((aspect_ratio - 1.5) / 4.5)
            * _clamp01(area_ratio / 0.20)
            * _clamp01(fill_ratio / 0.55)
            * _clamp01((vertical_center - 0.25) / 0.50)
        )
        if score > best_score:
            best_score = score
            best_box = {
                "x": int(x),
                "y": int(y),
                "width": int(box_width),
                "height": int(box_height),
                "aspect_ratio": float(aspect_ratio),
                "area_ratio": float(area_ratio),
                "fill_ratio": float(fill_ratio),
            }

    return float(best_score), best_box


def _text_features(hsv: np.ndarray) -> tuple[float, int, float, float]:
    height, width = hsv.shape[:2]
    y1, y2 = int(0.30 * height), int(0.88 * height)
    x1, x2 = int(0.03 * width), int(0.88 * width)
    band = hsv[y1:y2, x1:x2]

    bright_mask = cv2.inRange(
        band,
        np.array([0, 0, 145], dtype=np.uint8),
        np.array([179, 120, 255], dtype=np.uint8),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    text_area = 0
    text_components = 0
    band_height, band_width = bright_mask.shape

    for index in range(1, component_count):
        _, _, component_width, component_height, area = stats[index]
        if not (2 <= component_height <= 0.25 * band_height):
            continue
        if not (1 <= component_width <= 0.25 * band_width):
            continue
        aspect_ratio = component_width / max(component_height, 1)
        if 0.05 <= aspect_ratio <= 12.0 and area >= 3:
            text_area += int(area)
            text_components += 1

    text_density = text_area / max(band_width * band_height, 1)
    text_density_score = _clamp01((text_density - 0.005) / 0.035)
    text_component_score = _clamp01(text_components / 35.0)
    return (
        float(text_density),
        int(text_components),
        float(text_density_score),
        float(text_component_score),
    )


def _edge_density_score(gray: np.ndarray) -> tuple[float, float]:
    height, width = gray.shape
    band = gray[int(0.30 * height) : int(0.88 * height), int(0.03 * width) : int(0.88 * width)]
    edges = cv2.Canny(band, 60, 150)
    density = float((edges > 0).mean())
    return density, _clamp01((density - 0.02) / 0.10)


def _logo_circularity_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    left = gray[int(0.22 * height) : int(0.95 * height), : int(0.25 * width)]
    edges = cv2.Canny(left, 50, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 30 or perimeter <= 0:
            continue
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        aspect_similarity = min(box_width, box_height) / max(box_width, box_height)
        size_ratio = max(box_width, box_height) / max(min(left.shape[:2]), 1)
        score = (
            _clamp01((circularity - 0.20) / 0.50)
            * _clamp01(aspect_similarity / 0.80)
            * _clamp01(size_ratio / 0.25)
        )
        best = max(best, score)
    return float(best)


def _warm_row_score(mask: np.ndarray) -> tuple[float, float]:
    row_fraction = mask.mean(axis=1) / 255.0
    window = max(3, int(round(0.30 * mask.shape[0])))
    if len(row_fraction) >= window:
        kernel = np.ones(window, dtype=float) / window
        max_fraction = float(np.convolve(row_fraction, kernel, mode="valid").max())
    else:
        max_fraction = float(row_fraction.mean())
    return max_fraction, _clamp01((max_fraction - 0.25) / 0.55)


def analyze_lower_third_frame(
    frame_bgr: np.ndarray,
    config: LowerThirdVisualConfig | None = None,
) -> dict:
    """Calculate lower-third visual features for one frame.

    Returns a flat dictionary suitable for direct storage in a Parquet table. The
    final score was calibrated against the supplied positive/negative screenshots,
    while keeping all thresholds configurable for validation on full episodes.
    """
    cfg = config or LowerThirdVisualConfig()
    original_height, original_width = frame_bgr.shape[:2]
    crop = crop_normalized_roi(frame_bgr, cfg.roi)
    analysis_crop = _resize_for_analysis(crop, cfg.analysis_width)
    hsv = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2GRAY)
    warm_mask = _warm_mask(hsv, cfg)

    rectangle_score, rectangle_box = _warm_rectangle_score(warm_mask)
    edge_density, edge_score = _edge_density_score(gray)
    text_density, text_components, text_density_score, text_component_score = _text_features(hsv)
    logo_score = _logo_circularity_score(gray)
    warm_row_fraction, warm_row_score = _warm_row_score(warm_mask)

    visual_score = (
        cfg.weight_rectangle * rectangle_score
        + cfg.weight_edge_density * edge_score
        + cfg.weight_text_density * text_density_score
        + cfg.weight_text_components * text_component_score
        + cfg.weight_logo_circularity * logo_score
        + cfg.weight_warm_row * warm_row_score
    )
    visual_score = _clamp01(visual_score)

    result = {
        "frame_width": int(original_width),
        "frame_height": int(original_height),
        "roi_xmin": float(cfg.roi.xmin),
        "roi_ymin": float(cfg.roi.ymin),
        "roi_xmax": float(cfg.roi.xmax),
        "roi_ymax": float(cfg.roi.ymax),
        "warm_rectangle_score": round(rectangle_score, 6),
        "edge_density": round(edge_density, 6),
        "edge_density_score": round(edge_score, 6),
        "bright_text_density": round(text_density, 6),
        "bright_text_component_count": int(text_components),
        "bright_text_density_score": round(text_density_score, 6),
        "bright_text_component_score": round(text_component_score, 6),
        "logo_circularity_score": round(logo_score, 6),
        "warm_row_fraction": round(warm_row_fraction, 6),
        "warm_row_score": round(warm_row_score, 6),
        "lower_third_visual_score": round(visual_score, 6),
        "lower_third_detected": bool(visual_score >= cfg.visual_threshold),
        "visual_threshold": float(cfg.visual_threshold),
    }

    if rectangle_box is None:
        result.update(
            {
                "rectangle_x": None,
                "rectangle_y": None,
                "rectangle_width": None,
                "rectangle_height": None,
            }
        )
    else:
        result.update(
            {
                "rectangle_x": rectangle_box["x"],
                "rectangle_y": rectangle_box["y"],
                "rectangle_width": rectangle_box["width"],
                "rectangle_height": rectangle_box["height"],
            }
        )
    return result
