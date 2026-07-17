import cv2
import numpy as np
import pandas as pd

from airtime_bias.commentary.candidate_detection import (
    CommentaryScoreConfig,
    build_commentary_candidate_table,
)
from airtime_bias.lower_third.integration import attach_lower_third_signals
from airtime_bias.lower_third.ocr import match_participant_name
from airtime_bias.lower_third.temporal_events import group_lower_third_events
from airtime_bias.lower_third.visual_detection import (
    LowerThirdVisualConfig,
    analyze_lower_third_frame,
)
from airtime_bias.validation.review_queue import build_or_refresh_review_queue


def _synthetic_label_frame() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (45, 55, 70)
    cv2.rectangle(frame, (60, 520), (790, 660), (55, 95, 145), -1)
    cv2.circle(frame, (105, 590), 48, (80, 125, 185), 8)
    cv2.putText(
        frame,
        "LARISSA, 22",
        (175, 585),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (245, 245, 245),
        4,
    )
    cv2.putText(
        frame,
        "SAO BERNARDO DO CAMPO/SP",
        (175, 630),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
    )
    return frame


def test_visual_detector_finds_synthetic_lower_third():
    result = analyze_lower_third_frame(
        _synthetic_label_frame(), LowerThirdVisualConfig(visual_threshold=0.40)
    )
    assert result["lower_third_visual_score"] >= 0.40
    assert result["lower_third_detected"] is True


def test_visual_detector_rejects_plain_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (30, 40, 50)
    result = analyze_lower_third_frame(frame)
    assert result["lower_third_visual_score"] < 0.50
    assert result["lower_third_detected"] is False


def test_temporal_grouping_keeps_short_two_sample_event():
    rows = []
    for time, detected, score in [
        (0.0, False, 0.1),
        (0.5, True, 0.7),
        (1.0, True, 0.8),
        (1.5, False, 0.2),
    ]:
        rows.append(
            {
                "episode_id": "e1",
                "part_id": "p1",
                "part_order": 1,
                "part_timestamp_seconds": time,
                "global_timestamp_seconds": time,
                "sample_interval_seconds": 0.5,
                "lower_third_visual_score": score,
                "lower_third_detected": detected,
                "analysis_success": True,
                "full_frame_path": None,
                "roi_crop_path": None,
                "source_video_path": "video.mp4",
            }
        )
    events = group_lower_third_events(
        pd.DataFrame(rows), max_gap_seconds=0.6, min_positive_frames=2
    )
    assert len(events) == 1
    assert events.iloc[0]["positive_frame_count"] == 2
    assert events.iloc[0]["duration"] < 2.0


def test_ocr_name_matching_handles_minor_error():
    participant, score = match_participant_name(
        "LARlSSA 22 mc13 larissa", ["Larissa", "Nuri", "Maria Eduarda"]
    )
    assert participant == "Larissa"
    assert score >= 0.75


def test_duration_does_not_penalize_short_comment_by_default():
    rows = []
    for position in ["start", "middle", "end"]:
        rows.append(
            {
                "episode_id": "e1",
                "part_id": "p1",
                "part_order": 1,
                "segment_id": "s1",
                "segment_duration": 1.2,
                "frame_position": position,
                "analysis_success": True,
                "skipped_by_prefilter": False,
                "prefilter_pass": True,
                "face_count": 1,
                "main_face_area_ratio": 0.08,
                "main_face_center_distance": 0.1,
            }
        )
    result = build_commentary_candidate_table(
        pd.DataFrame(rows), CommentaryScoreConfig()
    )
    assert result.iloc[0]["is_shorter_than_two_seconds"]
    assert result.iloc[0]["duration_weight"] == 0.0
    assert result.iloc[0]["candidate_tier"] == "candidate"


def test_lower_third_promotes_overlapping_face_reject():
    candidates = pd.DataFrame(
        [
            {
                "episode_id": "e1",
                "part_id": "p1",
                "part_order": 1,
                "segment_id": "s1",
                "segment_duration": 1.2,
                "commentary_score": 0.35,
                "candidate_tier": "reject",
                "is_candidate": False,
                "review_required": False,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "segment_id": "s1",
                "episode_id": "e1",
                "part_id": "p1",
                "part_order": 1,
                "segment_duration": 1.2,
                "sample_offset_seconds": 0.2,
                "global_timestamp_seconds": 10.2,
                "part_timestamp_seconds": 10.2,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "episode_id": "e1",
                "part_id": "p1",
                "lower_third_event_id": "p1_lt_1",
                "global_start_time": 10.0,
                "global_end_time": 11.0,
                "max_visual_score": 0.8,
                "mean_visual_score": 0.75,
                "participant_name_similarity": 0.9,
                "matched_participant": "Larissa",
                "representative_full_frame_path": "frame.jpg",
                "representative_roi_crop_path": "crop.jpg",
            }
        ]
    )
    output = attach_lower_third_signals(
        candidates,
        frame_samples=samples,
        lower_third_events=events,
    )
    assert output.iloc[0]["candidate_tier"] == "candidate"
    assert output.iloc[0]["candidate_source"] == "lower_third"
    assert output.iloc[0]["lower_third_matched_participant"] == "Larissa"


def test_refresh_review_queue_preserves_manual_labels():
    candidates = pd.DataFrame(
        [
            {"segment_id": "s1", "candidate_tier": "candidate", "commentary_score": 0.9},
            {"segment_id": "s2", "candidate_tier": "review", "commentary_score": 0.6},
        ]
    )
    existing = pd.DataFrame(
        [
            {
                "segment_id": "s1",
                "candidate_tier": "candidate",
                "commentary_score": 0.8,
                "manual_review_status": "reviewed",
                "manual_scene_type": "participant_closeup",
                "manual_is_commentary": False,
                "manual_identity": "Larissa",
                "manual_notes": "clean close-up",
            }
        ]
    )
    refreshed = build_or_refresh_review_queue(
        candidates,
        existing_queue=existing,
        source_candidate_table="c.parquet",
        source_frame_table="f.parquet",
        review_run_id="run1",
    )
    row = refreshed[refreshed["segment_id"] == "s1"].iloc[0]
    assert row["manual_scene_type"] == "participant_closeup"
    assert row["manual_identity"] == "Larissa"
    assert row["manual_notes"] == "clean close-up"
