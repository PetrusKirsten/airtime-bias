from __future__ import annotations

import pandas as pd


def derive_segment_intervals(frame_samples: pd.DataFrame) -> pd.DataFrame:
    """Derive one local/global interval per sampled scene."""
    if frame_samples.empty:
        return pd.DataFrame()
    required = {"segment_id", "part_id", "segment_duration"}
    missing = required - set(frame_samples.columns)
    if missing:
        raise ValueError(f"Frame sample table is missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for segment_id, group in frame_samples.groupby("segment_id", sort=False):
        first = group.iloc[0]
        duration_values = pd.to_numeric(group["segment_duration"], errors="coerce").dropna()
        duration = float(duration_values.iloc[0]) if not duration_values.empty else 0.0

        part_start_values: list[float] = []
        global_start_values: list[float] = []
        if "sample_offset_seconds" in group.columns:
            offsets = pd.to_numeric(group["sample_offset_seconds"], errors="coerce")
            part_times = pd.to_numeric(group.get("part_timestamp_seconds"), errors="coerce")
            global_times = pd.to_numeric(group.get("global_timestamp_seconds"), errors="coerce")
            part_start_values = (part_times - offsets).dropna().tolist()
            global_start_values = (global_times - offsets).dropna().tolist()

        if part_start_values:
            part_start = float(pd.Series(part_start_values).median())
        else:
            available = pd.to_numeric(group.get("part_timestamp_seconds"), errors="coerce").dropna()
            part_start = max(0.0, float(available.min()) - 0.25) if not available.empty else 0.0

        if global_start_values:
            global_start = float(pd.Series(global_start_values).median())
        else:
            available = pd.to_numeric(group.get("global_timestamp_seconds"), errors="coerce").dropna()
            global_start = max(0.0, float(available.min()) - 0.25) if not available.empty else part_start

        rows.append(
            {
                "segment_id": str(segment_id),
                "episode_id": first.get("episode_id"),
                "part_id": str(first.get("part_id")),
                "part_order": first.get("part_order"),
                "segment_part_start_time": part_start,
                "segment_part_end_time": part_start + duration,
                "segment_global_start_time": global_start,
                "segment_global_end_time": global_start + duration,
            }
        )
    return pd.DataFrame(rows)


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def attach_lower_third_signals(
    candidates: pd.DataFrame,
    *,
    frame_samples: pd.DataFrame,
    lower_third_events: pd.DataFrame,
    review_threshold: float = 0.50,
    candidate_threshold: float = 0.70,
    matched_name_threshold: float = 0.68,
    strong_visual_candidate_threshold: float = 0.82,
    minimum_overlap_ratio_for_candidate: float = 0.25,
) -> pd.DataFrame:
    """Attach label events while keeping visual-only promotion conservative.

    A cast-name OCR match is strong enough to create a candidate. Visual evidence by
    itself can rescue a face reject into manual review, but cannot automatically turn
    that reject into a high-confidence candidate. This prevents sponsor boards and set
    graphics from propagating directly into airtime totals.
    """
    if candidates.empty:
        return candidates.copy()

    output = candidates.copy()
    output["face_commentary_score"] = pd.to_numeric(
        output["commentary_score"], errors="coerce"
    ).fillna(0.0)
    output["face_candidate_tier"] = output["candidate_tier"].astype(str)

    default_columns = {
        "lower_third_detected": False,
        "lower_third_event_ids": None,
        "lower_third_overlap_seconds": 0.0,
        "lower_third_overlap_ratio": 0.0,
        "lower_third_max_visual_score": 0.0,
        "lower_third_mean_visual_score": 0.0,
        "lower_third_matched_participant": None,
        "lower_third_name_similarity": 0.0,
        "lower_third_ocr_confirmed": False,
        "lower_third_representative_frame_path": None,
        "lower_third_representative_crop_path": None,
        "candidate_source": "face",
    }
    for column, value in default_columns.items():
        output[column] = value

    intervals = derive_segment_intervals(frame_samples)
    output = output.merge(intervals, on=["segment_id"], how="left", suffixes=("", "_interval"))

    if lower_third_events.empty:
        return output

    events = lower_third_events.copy()
    for index, candidate in output.iterrows():
        part_id = str(candidate.get("part_id"))
        start = candidate.get("segment_global_start_time")
        end = candidate.get("segment_global_end_time")
        if pd.isna(start) or pd.isna(end):
            continue

        matching = events[events["part_id"].astype(str) == part_id].copy()
        if "episode_id" in events.columns and pd.notna(candidate.get("episode_id")):
            matching = matching[
                matching["episode_id"].astype(str) == str(candidate.get("episode_id"))
            ].copy()
        if matching.empty:
            continue

        matching["_overlap"] = matching.apply(
            lambda row: _overlap_seconds(
                float(start),
                float(end),
                float(row["global_start_time"]),
                float(row["global_end_time"]),
            ),
            axis=1,
        )
        matching = matching[matching["_overlap"] > 0].copy()
        if matching.empty:
            continue

        best = matching.sort_values(["_overlap", "max_visual_score"], ascending=False).iloc[0]
        total_overlap = float(matching["_overlap"].sum())
        duration = max(float(end) - float(start), 1e-9)
        overlap_ratio = min(1.0, total_overlap / duration)
        visual_score = float(pd.to_numeric(matching["max_visual_score"], errors="coerce").max())
        mean_score = float(pd.to_numeric(matching["mean_visual_score"], errors="coerce").max())

        similarities = pd.to_numeric(
            matching.get("participant_name_similarity", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        name_similarity = float(similarities.max()) if not similarities.empty else 0.0
        matched_rows = (
            matching[matching["matched_participant"].notna()]
            if "matched_participant" in matching.columns
            else pd.DataFrame()
        )
        matched_participant = (
            str(
                matched_rows.sort_values(
                    "participant_name_similarity", ascending=False
                ).iloc[0]["matched_participant"]
            )
            if not matched_rows.empty
            else None
        )
        ocr_confirmed = (
            matched_participant is not None and name_similarity >= matched_name_threshold
        )

        face_score = float(candidate.get("face_commentary_score") or 0.0)
        face_tier = str(candidate.get("face_candidate_tier", "reject"))
        lower_signal = max(visual_score, name_similarity)
        combined_score = 1.0 - (1.0 - face_score) * (1.0 - lower_signal)
        combined_score = max(0.0, min(1.0, combined_score))
        strong_visual_overlap = (
            visual_score >= strong_visual_candidate_threshold
            and overlap_ratio >= minimum_overlap_ratio_for_candidate
        )

        if ocr_confirmed:
            tier = "candidate"
        elif face_tier == "candidate":
            tier = "candidate"
        elif face_tier == "review":
            tier = "candidate" if strong_visual_overlap and combined_score >= candidate_threshold else "review"
        elif strong_visual_overlap or combined_score >= review_threshold:
            tier = "review"
        else:
            tier = face_tier

        source = "face+lower_third" if face_tier in {"candidate", "review"} else "lower_third"
        output.at[index, "lower_third_detected"] = True
        output.at[index, "lower_third_event_ids"] = ",".join(
            matching["lower_third_event_id"].astype(str).tolist()
        )
        output.at[index, "lower_third_overlap_seconds"] = total_overlap
        output.at[index, "lower_third_overlap_ratio"] = overlap_ratio
        output.at[index, "lower_third_max_visual_score"] = visual_score
        output.at[index, "lower_third_mean_visual_score"] = mean_score
        output.at[index, "lower_third_matched_participant"] = matched_participant
        output.at[index, "lower_third_name_similarity"] = name_similarity
        output.at[index, "lower_third_ocr_confirmed"] = bool(ocr_confirmed)
        output.at[index, "lower_third_representative_frame_path"] = best.get(
            "representative_full_frame_path"
        )
        output.at[index, "lower_third_representative_crop_path"] = best.get(
            "representative_roi_crop_path"
        )
        output.at[index, "candidate_source"] = source
        output.at[index, "commentary_score"] = round(combined_score, 4)
        output.at[index, "candidate_tier"] = tier
        output.at[index, "is_candidate"] = tier == "candidate"
        output.at[index, "review_required"] = tier == "review"

    return output
