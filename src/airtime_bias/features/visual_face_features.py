from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import pandas as pd

from airtime_bias.vision.face_detection import (
    FaceDetectorConfig,
    MediaPipeFaceDetector,
    select_main_face,
)


def _empty_feature_row(sample: pd.Series) -> dict:
    """Create a stable output schema from one frame-sample row."""
    return {
        "episode_id": sample.get("episode_id"),
        "part_id": sample.get("part_id"),
        "part_order": sample.get("part_order"),
        "segment_id": sample.get("segment_id"),
        "segment_duration": sample.get("segment_duration"),
        "frame_position": sample.get("frame_position"),
        "frame_path": sample.get("frame_path"),
        "source_video_path": sample.get("source_video_path"),
        "part_timestamp_seconds": sample.get("part_timestamp_seconds"),
        "global_timestamp_seconds": sample.get("global_timestamp_seconds"),
        "analysis_success": False,
        "skipped_by_prefilter": False,
        "prefilter_pass": False,
        "face_count": None,
        "max_face_confidence": None,
        "main_face_xmin": None,
        "main_face_ymin": None,
        "main_face_width": None,
        "main_face_height": None,
        "main_face_area_ratio": None,
        "main_face_center_distance": None,
        "error": None,
    }


def _analyze_sample(sample: pd.Series, detector: MediaPipeFaceDetector) -> dict:
    """Read one sampled image and calculate face-level visual features."""
    row = _empty_feature_row(sample)

    if not bool(sample.get("success", True)):
        row["error"] = sample.get("error") or "input_frame_extraction_failed"
        return row

    frame_path_value = sample.get("frame_path")
    if not isinstance(frame_path_value, str) or not frame_path_value:
        row["error"] = "frame_path_missing"
        return row

    frame_path = Path(frame_path_value)
    if not frame_path.exists():
        row["error"] = "frame_file_not_found"
        return row

    frame = cv2.imread(str(frame_path))
    if frame is None:
        row["error"] = "could_not_read_frame_image"
        return row

    try:
        faces = detector.detect(frame)
    except Exception as exc:
        row["error"] = f"face_detection_failed: {exc}"
        return row

    row["analysis_success"] = True
    row["face_count"] = len(faces)
    row["max_face_confidence"] = (
        max(float(face.get("confidence", 0.0)) for face in faces) if faces else None
    )

    main_face = select_main_face(faces)
    if main_face is not None:
        row.update(
            {
                "main_face_xmin": float(main_face["xmin"]),
                "main_face_ymin": float(main_face["ymin"]),
                "main_face_width": float(main_face["width"]),
                "main_face_height": float(main_face["height"]),
                "main_face_area_ratio": float(main_face["area_ratio"]),
                "main_face_center_distance": float(main_face["center_distance"]),
            }
        )

    return row


def middle_frame_passes_prefilter(
    feature_row: dict,
    min_face_area_ratio: float = 0.02,
    max_center_distance: float = 0.55,
    max_faces: int | None = 4,
) -> bool:
    """Apply a deliberately recall-oriented filter to the middle frame.

    The prefilter only rejects obvious negatives. It does not require exactly one
    face because reaction shots and partial group shots may still deserve review.
    """
    if not feature_row.get("analysis_success"):
        return False

    face_count = int(feature_row.get("face_count") or 0)
    if face_count < 1:
        return False
    if max_faces is not None and face_count > int(max_faces):
        return False

    area_ratio = feature_row.get("main_face_area_ratio")
    center_distance = feature_row.get("main_face_center_distance")
    if area_ratio is None or center_distance is None:
        return False

    return bool(
        float(area_ratio) >= float(min_face_area_ratio)
        and float(center_distance) <= float(max_center_distance)
    )


def extract_face_features_from_samples(
    frame_samples: pd.DataFrame,
    detector_config: FaceDetectorConfig | None = None,
    use_middle_prefilter: bool = True,
    prefilter_min_face_area_ratio: float = 0.02,
    prefilter_max_center_distance: float = 0.55,
    prefilter_max_faces: int | None = 4,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Extract frame-level face features with an optional two-stage cascade.

    When the middle-frame prefilter is enabled, the middle image is analyzed first.
    Start/end images are only analyzed when the middle frame contains at least one
    sufficiently large and reasonably central face. Skipped rows are retained in
    the output so segment-level aggregation remains transparent and auditable.
    """
    required_columns = {
        "segment_id",
        "frame_position",
        "frame_path",
        "success",
    }
    missing = required_columns - set(frame_samples.columns)
    if missing:
        raise ValueError(f"Frame sample table is missing required columns: {sorted(missing)}")
    if frame_samples.empty:
        return pd.DataFrame()

    working = frame_samples.copy()
    working["segment_id"] = working["segment_id"].astype(str)
    working["frame_position"] = working["frame_position"].astype(str)

    position_order = {"middle": 0, "start": 1, "end": 2}
    working["_position_order"] = working["frame_position"].map(position_order).fillna(9)
    sort_columns = [
        column
        for column in ("part_order", "global_timestamp_seconds", "segment_id", "_position_order")
        if column in working.columns
    ]
    working = working.sort_values(sort_columns)

    grouped = list(working.groupby("segment_id", sort=False))
    total_segments = len(grouped)
    output_rows: list[dict] = []

    with MediaPipeFaceDetector(detector_config or FaceDetectorConfig()) as detector:
        for index, (segment_id, segment_rows) in enumerate(grouped, start=1):
            segment_rows = segment_rows.sort_values("_position_order")
            middle_rows = segment_rows[segment_rows["frame_position"] == "middle"]
            anchor_sample = middle_rows.iloc[0] if not middle_rows.empty else segment_rows.iloc[0]

            anchor_features = _analyze_sample(anchor_sample, detector)
            if use_middle_prefilter:
                prefilter_pass = middle_frame_passes_prefilter(
                    anchor_features,
                    min_face_area_ratio=prefilter_min_face_area_ratio,
                    max_center_distance=prefilter_max_center_distance,
                    max_faces=prefilter_max_faces,
                )
            else:
                prefilter_pass = True

            anchor_features["prefilter_pass"] = prefilter_pass
            output_rows.append(anchor_features)

            anchor_index = anchor_sample.name
            for sample_index, sample in segment_rows.iterrows():
                if sample_index == anchor_index:
                    continue

                if use_middle_prefilter and not prefilter_pass:
                    skipped = _empty_feature_row(sample)
                    skipped["skipped_by_prefilter"] = True
                    skipped["prefilter_pass"] = False
                    skipped["error"] = "skipped_after_middle_frame_prefilter"
                    output_rows.append(skipped)
                    continue

                features = _analyze_sample(sample, detector)
                features["prefilter_pass"] = prefilter_pass
                output_rows.append(features)

            if progress_callback is not None:
                progress_callback(index, total_segments, segment_id)

    result = pd.DataFrame(output_rows)
    if not result.empty:
        result = result.sort_values(
            [
                column
                for column in ("part_order", "global_timestamp_seconds", "segment_id", "frame_position")
                if column in result.columns
            ]
        ).reset_index(drop=True)
    return result
