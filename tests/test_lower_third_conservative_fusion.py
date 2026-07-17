import pandas as pd

from airtime_bias.lower_third.integration import attach_lower_third_signals


def test_visual_only_lower_third_does_not_auto_accept_face_reject():
    candidates = pd.DataFrame(
        [
            {
                "episode_id": "e1",
                "part_id": "p1",
                "part_order": 1,
                "segment_id": "s1",
                "segment_duration": 2.0,
                "commentary_score": 0.25,
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
                "segment_duration": 2.0,
                "sample_offset_seconds": 0.5,
                "part_timestamp_seconds": 10.5,
                "global_timestamp_seconds": 10.5,
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
                "global_end_time": 12.0,
                "max_visual_score": 0.90,
                "mean_visual_score": 0.86,
                "participant_name_similarity": 0.0,
                "matched_participant": None,
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

    assert output.iloc[0]["candidate_tier"] == "review"
    assert not bool(output.iloc[0]["is_candidate"])
    assert output.iloc[0]["candidate_source"] == "lower_third"
