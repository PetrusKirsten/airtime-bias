from __future__ import annotations

import pandas as pd


SCENE_TYPES = ("pending", "commentary", "participant_closeup", "other", "uncertain")
MANUAL_COLUMNS = (
    "manual_review_status",
    "manual_scene_type",
    "manual_is_commentary",
    "manual_identity",
    "manual_notes",
)


def ensure_review_schema(review_queue: pd.DataFrame) -> pd.DataFrame:
    """Add the current manual-review schema while preserving older decisions."""
    queue = review_queue.copy()
    if "manual_review_status" not in queue.columns:
        queue["manual_review_status"] = "pending"
    queue["manual_review_status"] = queue["manual_review_status"].fillna("pending").astype(str)

    if "manual_is_commentary" not in queue.columns:
        queue["manual_is_commentary"] = pd.Series(pd.NA, index=queue.index, dtype="boolean")
    else:
        queue["manual_is_commentary"] = queue["manual_is_commentary"].astype("boolean")

    if "manual_identity" not in queue.columns:
        queue["manual_identity"] = pd.Series(pd.NA, index=queue.index, dtype="string")
    else:
        queue["manual_identity"] = queue["manual_identity"].astype("string")

    if "manual_notes" not in queue.columns:
        queue["manual_notes"] = ""
    queue["manual_notes"] = queue["manual_notes"].fillna("").astype(str)

    if "manual_scene_type" not in queue.columns:
        inferred: list[str] = []
        for _, row in queue.iterrows():
            status = str(row.get("manual_review_status", "pending"))
            manual_value = row.get("manual_is_commentary")
            identity = row.get("manual_identity")
            if status == "uncertain":
                scene_type = "uncertain"
            elif pd.notna(manual_value) and bool(manual_value):
                scene_type = "commentary"
            elif pd.notna(manual_value) and not bool(manual_value) and pd.notna(identity):
                scene_type = "participant_closeup"
            elif pd.notna(manual_value) and not bool(manual_value):
                scene_type = "other"
            else:
                scene_type = "pending"
            inferred.append(scene_type)
        queue["manual_scene_type"] = inferred
    else:
        queue["manual_scene_type"] = queue["manual_scene_type"].fillna("pending").astype(str)

    return queue


def build_or_refresh_review_queue(
    candidates: pd.DataFrame,
    *,
    existing_queue: pd.DataFrame | None = None,
    source_candidate_table: str,
    source_frame_table: str,
    review_run_id: str,
) -> pd.DataFrame:
    """Create a queue and preserve all previously saved manual decisions by segment ID."""
    queue = candidates[candidates["candidate_tier"].isin(["candidate", "review"])].copy()
    queue["source_candidate_table"] = source_candidate_table
    queue["source_frame_table"] = source_frame_table
    queue["review_run_id"] = review_run_id
    queue = ensure_review_schema(queue)

    if existing_queue is None or existing_queue.empty:
        return queue

    existing = ensure_review_schema(existing_queue)
    preserved_columns = ["segment_id", *MANUAL_COLUMNS]
    preserved = existing[[column for column in preserved_columns if column in existing.columns]].copy()
    queue = queue.drop(columns=[column for column in MANUAL_COLUMNS if column in queue.columns])
    queue = queue.merge(preserved, on="segment_id", how="left")
    return ensure_review_schema(queue)


def apply_manual_decision(
    review_queue: pd.DataFrame,
    *,
    segment_id: str,
    scene_type: str,
    identity: str | None,
    notes: str,
) -> pd.DataFrame:
    if scene_type not in SCENE_TYPES:
        raise ValueError(f"Unsupported manual scene type: {scene_type}")
    queue = ensure_review_schema(review_queue)
    mask = queue["segment_id"].astype(str) == str(segment_id)
    if not mask.any():
        raise KeyError(f"Segment not found in review queue: {segment_id}")

    if scene_type == "commentary":
        status, is_commentary = "reviewed", True
    elif scene_type in {"participant_closeup", "other"}:
        status, is_commentary = "reviewed", False
    elif scene_type == "uncertain":
        status, is_commentary = "uncertain", pd.NA
    else:
        status, is_commentary = "pending", pd.NA

    queue.loc[mask, "manual_review_status"] = status
    queue.loc[mask, "manual_scene_type"] = scene_type
    queue.loc[mask, "manual_is_commentary"] = is_commentary
    queue.loc[mask, "manual_identity"] = identity or pd.NA
    queue.loc[mask, "manual_notes"] = notes or ""
    return queue


def build_identity_label_table(
    review_queue: pd.DataFrame,
    frame_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Expand named manual decisions into frame-level labels for future face recognition."""
    queue = ensure_review_schema(review_queue)
    named = queue[queue["manual_identity"].notna()].copy()
    if named.empty or frame_samples.empty:
        return pd.DataFrame()

    columns = [
        "segment_id",
        "manual_identity",
        "manual_scene_type",
        "manual_is_commentary",
        "source_candidate_table",
        "review_run_id",
    ]
    columns = [column for column in columns if column in named.columns]
    labels = named[columns].merge(frame_samples, on="segment_id", how="inner")
    if "success" in labels.columns:
        labels = labels[labels["success"].fillna(False).astype(bool)].copy()
    labels["identity_label_source"] = "manual_candidate_review"
    return labels.reset_index(drop=True)
