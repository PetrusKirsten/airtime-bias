from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .visual_detection import (
    LowerThirdVisualConfig,
    analyze_lower_third_frame,
    crop_normalized_roi,
)


def _safe_token(value: object) -> str:
    token = "".join(character if str(character).isalnum() or character in "-_" else "_" for character in str(value))
    token = token.strip("_")
    return token or "unknown"


def _read_frame(cap: cv2.VideoCapture, timestamp_seconds: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_seconds)) * 1000.0)
    success, frame = cap.read()
    return frame if success else None


def scan_lower_thirds_in_video(
    video_path: str | Path,
    *,
    episode_id: str,
    part_id: str,
    part_order: int | None = None,
    global_offset_seconds: float = 0.0,
    sample_interval_seconds: float = 0.50,
    start_time_seconds: float = 0.0,
    end_time_seconds: float | None = None,
    visual_config: LowerThirdVisualConfig | None = None,
    output_dir: str | Path | None = None,
    save_score_margin: float = 0.08,
    jpeg_quality: int = 85,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Scan one video part using a small lower-left crop at a fixed interval."""
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive.")
    if start_time_seconds < 0:
        raise ValueError("start_time_seconds cannot be negative.")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")

    cfg = visual_config or LowerThirdVisualConfig()
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not video_path.exists() or not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration <= 0:
        duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
    end_time = duration if end_time_seconds is None else min(float(end_time_seconds), duration)
    if end_time <= start_time_seconds:
        cap.release()
        raise ValueError("The requested lower-third scan interval is empty.")

    timestamps = np.arange(
        float(start_time_seconds),
        float(end_time) + sample_interval_seconds * 0.25,
        float(sample_interval_seconds),
    )
    timestamps = timestamps[timestamps < end_time + 1e-6]
    total = len(timestamps)
    rows: list[dict] = []

    image_dir: Path | None = None
    if output_dir is not None:
        image_dir = Path(output_dir) / _safe_token(episode_id) / _safe_token(part_id)
        image_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, timestamp in enumerate(timestamps, start=1):
            frame = _read_frame(cap, float(timestamp))
            base_row = {
                "episode_id": str(episode_id),
                "part_id": str(part_id),
                "part_order": part_order,
                "source_video_path": str(video_path),
                "part_timestamp_seconds": float(timestamp),
                "global_timestamp_seconds": float(global_offset_seconds + timestamp),
                "sample_interval_seconds": float(sample_interval_seconds),
                "source_frame_index": int(round(timestamp * fps)) if fps > 0 else None,
                "analysis_success": False,
                "full_frame_path": None,
                "roi_crop_path": None,
                "error": None,
            }

            if frame is None:
                base_row["error"] = "could_not_read_frame"
                rows.append(base_row)
            else:
                try:
                    features = analyze_lower_third_frame(frame, cfg)
                    base_row.update(features)
                    base_row["analysis_success"] = True

                    save_threshold = max(0.0, cfg.visual_threshold - float(save_score_margin))
                    if image_dir is not None and features["lower_third_visual_score"] >= save_threshold:
                        time_token = f"{timestamp:010.3f}".replace(".", "p")
                        full_path = image_dir / f"{time_token}_full.jpg"
                        crop_path = image_dir / f"{time_token}_roi.jpg"
                        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                        crop = crop_normalized_roi(frame, cfg.roi)
                        if cv2.imwrite(str(full_path), frame, params):
                            base_row["full_frame_path"] = str(full_path)
                        if cv2.imwrite(str(crop_path), crop, params):
                            base_row["roi_crop_path"] = str(crop_path)
                except Exception as exc:
                    base_row["error"] = f"lower_third_analysis_failed: {exc}"
                rows.append(base_row)

            if progress_callback is not None:
                progress_callback(index, total, f"{part_id} @ {timestamp:.2f}s")
    finally:
        cap.release()

    return pd.DataFrame(rows)


def group_lower_third_events(
    frame_detections: pd.DataFrame,
    *,
    max_gap_seconds: float = 0.80,
    min_positive_frames: int = 2,
    event_padding_seconds: float = 0.25,
    strong_single_frame_threshold: float = 0.72,
) -> pd.DataFrame:
    """Group consecutive positive sampled frames into temporal lower-third events."""
    if frame_detections.empty:
        return pd.DataFrame()
    if max_gap_seconds < 0 or event_padding_seconds < 0:
        raise ValueError("Gap and padding values must be non-negative.")
    if min_positive_frames < 1:
        raise ValueError("min_positive_frames must be at least 1.")

    required = {
        "episode_id",
        "part_id",
        "part_timestamp_seconds",
        "global_timestamp_seconds",
        "lower_third_visual_score",
        "lower_third_detected",
    }
    missing = required - set(frame_detections.columns)
    if missing:
        raise ValueError(f"Lower-third frame table is missing columns: {sorted(missing)}")

    working = frame_detections.copy()
    working = working[
        working["analysis_success"].fillna(False).astype(bool)
        & working["lower_third_detected"].fillna(False).astype(bool)
    ].copy()
    if working.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    event_counter = 0

    group_columns = [column for column in ("episode_id", "part_id") if column in working.columns]
    for _, part_rows in working.groupby(group_columns, sort=False):
        part_rows = part_rows.sort_values("part_timestamp_seconds").reset_index(drop=True)
        clusters: list[list[int]] = []
        current_cluster: list[int] = []
        previous_time: float | None = None

        for row_index, row in part_rows.iterrows():
            current_time = float(row["part_timestamp_seconds"])
            if previous_time is None or current_time - previous_time <= max_gap_seconds + 1e-9:
                current_cluster.append(row_index)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [row_index]
            previous_time = current_time
        if current_cluster:
            clusters.append(current_cluster)

        for cluster_indexes in clusters:
            cluster = part_rows.loc[cluster_indexes].copy()
            positive_frames = len(cluster)
            max_score = float(cluster["lower_third_visual_score"].max())
            if positive_frames < min_positive_frames and max_score < strong_single_frame_threshold:
                continue

            event_counter += 1
            representative = cluster.loc[cluster["lower_third_visual_score"].idxmax()]
            interval = float(cluster["sample_interval_seconds"].dropna().median()) if "sample_interval_seconds" in cluster.columns else 0.5
            part_start = max(0.0, float(cluster["part_timestamp_seconds"].min()) - event_padding_seconds)
            part_end = float(cluster["part_timestamp_seconds"].max()) + interval + event_padding_seconds
            global_start = max(
                0.0,
                float(cluster["global_timestamp_seconds"].min()) - event_padding_seconds,
            )
            global_end = float(cluster["global_timestamp_seconds"].max()) + interval + event_padding_seconds
            episode_id = str(representative["episode_id"])
            part_id = str(representative["part_id"])

            rows.append(
                {
                    "episode_id": episode_id,
                    "part_id": part_id,
                    "part_order": representative.get("part_order"),
                    "lower_third_event_id": f"{part_id}_lt_{event_counter:05d}",
                    "part_start_time": part_start,
                    "part_end_time": part_end,
                    "global_start_time": global_start,
                    "global_end_time": global_end,
                    "duration": max(0.0, part_end - part_start),
                    "positive_frame_count": int(positive_frames),
                    "mean_visual_score": float(cluster["lower_third_visual_score"].mean()),
                    "max_visual_score": max_score,
                    "representative_part_timestamp_seconds": float(
                        representative["part_timestamp_seconds"]
                    ),
                    "representative_global_timestamp_seconds": float(
                        representative["global_timestamp_seconds"]
                    ),
                    "representative_full_frame_path": representative.get("full_frame_path"),
                    "representative_roi_crop_path": representative.get("roi_crop_path"),
                    "source_video_path": representative.get("source_video_path"),
                    "ocr_raw_text": None,
                    "ocr_mean_confidence": None,
                    "matched_participant": None,
                    "participant_name_similarity": None,
                    "ocr_error": None,
                }
            )

    return pd.DataFrame(rows)
