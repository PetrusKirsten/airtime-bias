from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

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
    """Strict, interpretable detector for the participant lower-third graphic.

    Early versions combined generic rectangle, text, edge, and circular-logo cues.
    That was recall-oriented but also promoted sponsor boards, countertop edges, and
    decorative set panels. The current detector requires a conjunction of two signals
    specific to the observed participant graphic:

    1. a horizontally continuous copper/rose band across the expected label area;
    2. two concentrated rows of light, character-sized components inside that band.

    Logo and edge features remain weak supporting evidence, never sufficient alone.
    """

    roi: LowerThirdROI = field(default_factory=LowerThirdROI)
    analysis_width: int = 640
    visual_threshold: float = 0.58
    strict_gate: bool = True

    copper_hue_low_1: int = 0
    copper_hue_high_1: int = 20
    copper_hue_low_2: int = 170
    copper_hue_high_2: int = 179
    copper_saturation_min: int = 45
    copper_saturation_max: int = 185
    copper_value_min: int = 50
    copper_value_max: int = 225

    minimum_copper_density: float = 0.45
    minimum_copper_continuity: float = 0.45
    minimum_text_components: int = 18
    minimum_text_line_support: float = 0.50

    weight_copper_density: float = 0.40
    weight_copper_continuity: float = 0.25
    weight_text_lines: float = 0.15
    weight_text_components: float = 0.10
    weight_logo_circularity: float = 0.05
    weight_edge_density: float = 0.05

    def __post_init__(self) -> None:
        if self.analysis_width < 160:
            raise ValueError("analysis_width must be at least 160 pixels.")
        if not 0.0 <= self.visual_threshold <= 1.0:
            raise ValueError("visual_threshold must be between 0 and 1.")
        for name, value in (
            ("minimum_copper_density", self.minimum_copper_density),
            ("minimum_copper_continuity", self.minimum_copper_continuity),
            ("minimum_text_line_support", self.minimum_text_line_support),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.minimum_text_components < 1:
            raise ValueError("minimum_text_components must be positive.")

        weights = (
            self.weight_copper_density,
            self.weight_copper_continuity,
            self.weight_text_lines,
            self.weight_text_components,
            self.weight_logo_circularity,
            self.weight_edge_density,
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


def _copper_mask(hsv: np.ndarray, config: LowerThirdVisualConfig) -> np.ndarray:
    lower_1 = np.array(
        [
            config.copper_hue_low_1,
            config.copper_saturation_min,
            config.copper_value_min,
        ],
        dtype=np.uint8,
    )
    upper_1 = np.array(
        [
            config.copper_hue_high_1,
            config.copper_saturation_max,
            config.copper_value_max,
        ],
        dtype=np.uint8,
    )
    lower_2 = np.array(
        [
            config.copper_hue_low_2,
            config.copper_saturation_min,
            config.copper_value_min,
        ],
        dtype=np.uint8,
    )
    upper_2 = np.array(
        [
            config.copper_hue_high_2,
            config.copper_saturation_max,
            config.copper_value_max,
        ],
        dtype=np.uint8,
    )
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower_1, upper_1),
        cv2.inRange(hsv, lower_2, upper_2),
    )


def _expected_label_band(array: np.ndarray) -> np.ndarray:
    """Return the stable central band occupied by the participant graphic."""
    height, width = array.shape[:2]
    y1, y2 = int(0.28 * height), int(0.92 * height)
    x1, x2 = int(0.03 * width), int(0.82 * width)
    return array[y1:y2, x1:x2]


def _copper_band_features(mask: np.ndarray) -> dict:
    band = _expected_label_band(mask)
    present = band > 0
    sections = np.array_split(present, 4, axis=1)
    section_densities = [float(section.mean()) for section in sections]

    density = float(np.mean(section_densities))
    continuity = float(min(section_densities[1:]))

    row_fraction = present.mean(axis=1)
    window = max(3, int(round(0.25 * len(row_fraction))))
    if len(row_fraction) >= window:
        row_coherence = float(
            np.convolve(row_fraction, np.ones(window, dtype=float) / window, mode="valid").max()
        )
    else:
        row_coherence = float(row_fraction.mean())

    return {
        "copper_band_density": density,
        "copper_section_1_density": section_densities[0],
        "copper_section_2_density": section_densities[1],
        "copper_section_3_density": section_densities[2],
        "copper_section_4_density": section_densities[3],
        "copper_horizontal_continuity": continuity,
        "copper_row_coherence": row_coherence,
        "copper_density_score": _clamp01((density - 0.30) / 0.35),
        "copper_continuity_score": _clamp01((continuity - 0.35) / 0.30),
    }


def _text_line_features(hsv: np.ndarray) -> dict:
    height, width = hsv.shape[:2]
    band = hsv[
        int(0.30 * height) : int(0.92 * height),
        int(0.14 * width) : int(0.78 * width),
    ]
    bright_mask = cv2.inRange(
        band,
        np.array([0, 0, 145], dtype=np.uint8),
        np.array([179, 135, 255], dtype=np.uint8),
    )

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    band_height, band_width = bright_mask.shape
    centers_y: list[float] = []
    text_area = 0

    for index in range(1, component_count):
        _, y, component_width, component_height, area = stats[index]
        if not (2 <= component_height <= 0.28 * band_height):
            continue
        if not (1 <= component_width <= 0.18 * band_width):
            continue
        aspect_ratio = component_width / max(component_height, 1)
        if 0.05 <= aspect_ratio <= 12.0 and area >= 3:
            text_area += int(area)
            centers_y.append((float(y) + component_height / 2.0) / max(band_height, 1))

    text_components = len(centers_y)
    text_density = text_area / max(band_width * band_height, 1)

    if centers_y:
        histogram, _ = np.histogram(centers_y, bins=12, range=(0.0, 1.0))
        strongest_bins = sorted(histogram.tolist(), reverse=True)[:2]
        two_line_support = float(sum(strongest_bins) / text_components)
    else:
        two_line_support = 0.0

    return {
        "bright_text_density": float(text_density),
        "bright_text_component_count": int(text_components),
        "bright_text_density_score": _clamp01((text_density - 0.04) / 0.16),
        "bright_text_component_score": _clamp01((text_components - 15.0) / 35.0),
        "text_two_line_support": two_line_support,
        "text_two_line_score": _clamp01((two_line_support - 0.40) / 0.25),
    }


def _edge_density_score(gray: np.ndarray) -> tuple[float, float]:
    height, width = gray.shape
    band = gray[
        int(0.30 * height) : int(0.92 * height),
        int(0.14 * width) : int(0.78 * width),
    ]
    edges = cv2.Canny(band, 60, 150)
    density = float((edges > 0).mean())
    return density, _clamp01((density - 0.02) / 0.10)


def _logo_circularity_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    left = gray[int(0.22 * height) : int(0.95 * height), : int(0.23 * width)]
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


def _visual_gate(
    copper: dict,
    text: dict,
    config: LowerThirdVisualConfig,
) -> tuple[bool, str | None]:
    reasons: list[str] = []
    if copper["copper_band_density"] < config.minimum_copper_density:
        reasons.append("insufficient_copper_density")
    if copper["copper_horizontal_continuity"] < config.minimum_copper_continuity:
        reasons.append("insufficient_horizontal_continuity")
    if text["bright_text_component_count"] < config.minimum_text_components:
        reasons.append("insufficient_text_components")
    if text["text_two_line_support"] < config.minimum_text_line_support:
        reasons.append("missing_two_line_text_pattern")
    return not reasons, ";".join(reasons) if reasons else None


def analyze_lower_third_frame(
    frame_bgr: np.ndarray,
    config: LowerThirdVisualConfig | None = None,
) -> dict:
    """Calculate strict lower-third visual features for one full video frame."""
    cfg = config or LowerThirdVisualConfig()
    original_height, original_width = frame_bgr.shape[:2]
    crop = crop_normalized_roi(frame_bgr, cfg.roi)
    analysis_crop = _resize_for_analysis(crop, cfg.analysis_width)
    hsv = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2GRAY)

    copper = _copper_band_features(_copper_mask(hsv, cfg))
    text = _text_line_features(hsv)
    edge_density, edge_score = _edge_density_score(gray)
    logo_score = _logo_circularity_score(gray)

    visual_score = (
        cfg.weight_copper_density * copper["copper_density_score"]
        + cfg.weight_copper_continuity * copper["copper_continuity_score"]
        + cfg.weight_text_lines * text["text_two_line_score"]
        + cfg.weight_text_components * text["bright_text_component_score"]
        + cfg.weight_logo_circularity * logo_score
        + cfg.weight_edge_density * edge_score
    )
    visual_score = _clamp01(visual_score)
    gate_pass, rejection_reason = _visual_gate(copper, text, cfg)
    detected = visual_score >= cfg.visual_threshold and (gate_pass or not cfg.strict_gate)

    label_x = int(round(0.03 * analysis_crop.shape[1]))
    label_y = int(round(0.28 * analysis_crop.shape[0]))
    label_width = int(round(0.79 * analysis_crop.shape[1]))
    label_height = int(round(0.64 * analysis_crop.shape[0]))

    result = {
        "frame_width": int(original_width),
        "frame_height": int(original_height),
        "roi_xmin": float(cfg.roi.xmin),
        "roi_ymin": float(cfg.roi.ymin),
        "roi_xmax": float(cfg.roi.xmax),
        "roi_ymax": float(cfg.roi.ymax),
        **{key: round(value, 6) for key, value in copper.items()},
        **{
            key: (int(value) if key == "bright_text_component_count" else round(value, 6))
            for key, value in text.items()
        },
        "edge_density": round(edge_density, 6),
        "edge_density_score": round(edge_score, 6),
        "logo_circularity_score": round(logo_score, 6),
        "visual_gate_pass": bool(gate_pass),
        "visual_rejection_reason": rejection_reason,
        "lower_third_visual_score": round(visual_score, 6),
        "lower_third_detected": bool(detected),
        "visual_threshold": float(cfg.visual_threshold),
        "strict_gate": bool(cfg.strict_gate),
        "minimum_copper_density": float(cfg.minimum_copper_density),
        "minimum_copper_continuity": float(cfg.minimum_copper_continuity),
        "minimum_text_components": int(cfg.minimum_text_components),
        "minimum_text_line_support": float(cfg.minimum_text_line_support),
        "warm_rectangle_score": round(copper["copper_density_score"], 6),
        "warm_row_fraction": round(copper["copper_row_coherence"], 6),
        "warm_row_score": round(copper["copper_continuity_score"], 6),
        "rectangle_x": label_x,
        "rectangle_y": label_y,
        "rectangle_width": label_width,
        "rectangle_height": label_height,
    }
    return result
