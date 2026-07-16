import pandas as pd

from airtime_bias.commentary.candidate_detection import (
    CommentaryScoreConfig,
    build_commentary_candidate_table,
)


def _frame_row(
    segment_id: str,
    frame_position: str,
    face_count: int,
    area: float | None,
    center: float | None,
    duration: float = 4.0,
    prefilter_pass: bool = True,
) -> dict:
    return {
        "episode_id": "s01e01",
        "part_id": "s01e01_p01",
        "part_order": 1,
        "segment_id": segment_id,
        "segment_duration": duration,
        "frame_position": frame_position,
        "analysis_success": True,
        "skipped_by_prefilter": False,
        "prefilter_pass": prefilter_pass,
        "face_count": face_count,
        "main_face_area_ratio": area,
        "main_face_center_distance": center,
    }


def test_stable_single_face_scene_becomes_candidate() -> None:
    features = pd.DataFrame(
        [
            _frame_row("seg_good", "start", 1, 0.08, 0.08),
            _frame_row("seg_good", "middle", 1, 0.09, 0.07),
            _frame_row("seg_good", "end", 1, 0.085, 0.09),
        ]
    )

    result = build_commentary_candidate_table(features, CommentaryScoreConfig())

    assert len(result) == 1
    assert result.loc[0, "candidate_tier"] == "candidate"
    assert bool(result.loc[0, "is_candidate"])
    assert result.loc[0, "face_presence_ratio"] == 1.0
    assert result.loc[0, "single_face_ratio"] == 1.0


def test_scene_without_faces_is_rejected() -> None:
    features = pd.DataFrame(
        [
            _frame_row("seg_empty", "middle", 0, None, None, prefilter_pass=False),
            {
                **_frame_row("seg_empty", "start", 0, None, None, prefilter_pass=False),
                "analysis_success": False,
                "skipped_by_prefilter": True,
            },
            {
                **_frame_row("seg_empty", "end", 0, None, None, prefilter_pass=False),
                "analysis_success": False,
                "skipped_by_prefilter": True,
            },
        ]
    )

    result = build_commentary_candidate_table(features, CommentaryScoreConfig())

    assert len(result) == 1
    assert result.loc[0, "candidate_tier"] == "reject"
    assert not bool(result.loc[0, "is_candidate"])
    assert result.loc[0, "face_presence_ratio"] == 0.0
    assert int(result.loc[0, "skipped_frames"]) == 2
