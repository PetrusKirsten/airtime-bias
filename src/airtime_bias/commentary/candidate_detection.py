from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CommentaryScoreConfig:
    """Configuration for the interpretable commentary candidate score."""

    min_duration_seconds: float = 2.0
    min_face_area_ratio: float = 0.04
    max_center_distance: float = 0.35
    candidate_threshold: float = 0.70
    review_threshold: float = 0.50
    weight_face_presence: float = 0.30
    weight_single_face: float = 0.20
    weight_face_area: float = 0.20
    weight_centering: float = 0.15
    weight_stability: float = 0.10
    weight_duration: float = 0.05

    def __post_init__(self) -> None:
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive.")
        if self.min_face_area_ratio <= 0:
            raise ValueError("min_face_area_ratio must be positive.")
        if self.max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive.")
        if not 0 <= self.review_threshold <= self.candidate_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy 0 <= review_threshold <= candidate_threshold <= 1."
            )

        total_weight = sum(
            (
                self.weight_face_presence,
                self.weight_single_face,
                self.weight_face_area,
                self.weight_centering,
                self.weight_stability,
                self.weight_duration,
            )
        )
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Commentary score weights must sum to 1.0, got {total_weight:.4f}.")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _geometry_stability(face_rows: pd.DataFrame, max_center_distance: float) -> float:
    """Score stability of face size and position across available frames."""
    if len(face_rows) == 0:
        return 0.0
    if len(face_rows) == 1:
        return 0.50

    areas = pd.to_numeric(face_rows["main_face_area_ratio"], errors="coerce").dropna()
    centers = pd.to_numeric(face_rows["main_face_center_distance"], errors="coerce").dropna()

    if areas.empty or centers.empty:
        return 0.0

    mean_area = float(areas.mean())
    area_cv = float(areas.std(ddof=0) / mean_area) if mean_area > 0 else 1.0
    area_stability = _clamp01(1.0 - area_cv)

    center_spread = float(centers.std(ddof=0))
    center_stability = _clamp01(1.0 - center_spread / max_center_distance)

    return float((area_stability + center_stability) / 2.0)


def build_commentary_candidate_table(
    frame_features: pd.DataFrame,
    config: CommentaryScoreConfig | None = None,
) -> pd.DataFrame:
    """Aggregate frame-level face features into one row per scene segment.

    The score is intentionally heuristic and decomposed into explicit components.
    It is designed to reduce review workload while preserving uncertain cases for
    human validation, not to serve as a definitive classifier.
    """
    if frame_features.empty:
        return pd.DataFrame()

    required_columns = {
        "segment_id",
        "frame_position",
        "analysis_success",
        "skipped_by_prefilter",
        "prefilter_pass",
        "face_count",
        "main_face_area_ratio",
        "main_face_center_distance",
    }
    missing = required_columns - set(frame_features.columns)
    if missing:
        raise ValueError(f"Face feature table is missing required columns: {sorted(missing)}")

    cfg = config or CommentaryScoreConfig()
    rows: list[dict] = []

    for segment_id, group in frame_features.groupby("segment_id", sort=False):
        expected_frames = int(group["frame_position"].notna().sum())
        analysis_success = group["analysis_success"].fillna(False).astype(bool)
        analyzed = group.loc[analysis_success].copy()

        face_counts = pd.to_numeric(analyzed.get("face_count"), errors="coerce").fillna(0)
        face_presence_count = int((face_counts >= 1).sum())
        single_face_count = int((face_counts == 1).sum())

        face_rows = analyzed.loc[face_counts >= 1].copy()
        area_values = pd.to_numeric(
            face_rows.get("main_face_area_ratio"), errors="coerce"
        ).dropna()
        center_values = pd.to_numeric(
            face_rows.get("main_face_center_distance"), errors="coerce"
        ).dropna()

        median_area = float(area_values.median()) if not area_values.empty else 0.0
        median_center = float(center_values.median()) if not center_values.empty else np.nan

        face_presence_ratio = _safe_ratio(face_presence_count, expected_frames)
        single_face_ratio = _safe_ratio(single_face_count, expected_frames)
        face_area_score = _clamp01(median_area / cfg.min_face_area_ratio)
        centering_score = (
            _clamp01(1.0 - median_center / cfg.max_center_distance)
            if not np.isnan(median_center)
            else 0.0
        )
        stability_score = _geometry_stability(face_rows, cfg.max_center_distance)

        duration_values = pd.to_numeric(group.get("segment_duration"), errors="coerce").dropna()
        segment_duration = float(duration_values.iloc[0]) if not duration_values.empty else 0.0
        duration_score = _clamp01(segment_duration / cfg.min_duration_seconds)

        commentary_score = (
            cfg.weight_face_presence * face_presence_ratio
            + cfg.weight_single_face * single_face_ratio
            + cfg.weight_face_area * face_area_score
            + cfg.weight_centering * centering_score
            + cfg.weight_stability * stability_score
            + cfg.weight_duration * duration_score
        )
        commentary_score = _clamp01(commentary_score)

        if commentary_score >= cfg.candidate_threshold:
            tier = "candidate"
        elif commentary_score >= cfg.review_threshold:
            tier = "review"
        else:
            tier = "reject"

        first = group.iloc[0]
        rows.append(
            {
                "episode_id": first.get("episode_id"),
                "part_id": first.get("part_id"),
                "part_order": first.get("part_order"),
                "segment_id": segment_id,
                "segment_duration": segment_duration,
                "expected_frames": expected_frames,
                "analyzed_frames": int(analysis_success.sum()),
                "skipped_frames": int(group["skipped_by_prefilter"].fillna(False).sum()),
                "prefilter_pass": bool(group["prefilter_pass"].fillna(False).any()),
                "face_presence_count": face_presence_count,
                "face_presence_ratio": face_presence_ratio,
                "single_face_count": single_face_count,
                "single_face_ratio": single_face_ratio,
                "median_face_area_ratio": median_area,
                "median_face_center_distance": (
                    median_center if not np.isnan(median_center) else None
                ),
                "face_area_score": face_area_score,
                "centering_score": centering_score,
                "geometry_stability_score": stability_score,
                "duration_score": duration_score,
                "commentary_score": round(commentary_score, 4),
                "candidate_tier": tier,
                "is_candidate": tier == "candidate",
                "review_required": tier == "review",
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            [
                column
                for column in ("part_order", "commentary_score", "segment_id")
                if column in result.columns
            ],
            ascending=[True, False, True][: len(
                [
                    column
                    for column in ("part_order", "commentary_score", "segment_id")
                    if column in result.columns
                ]
            )],
        ).reset_index(drop=True)
    return result
