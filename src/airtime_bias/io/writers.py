from __future__ import annotations

from pathlib import Path

import pandas as pd


_MANUAL_REVIEW_COLUMNS = (
    "manual_review_status",
    "manual_scene_type",
    "manual_is_commentary",
    "manual_identity",
    "manual_notes",
)


def _looks_like_fresh_review_queue(df: pd.DataFrame) -> bool:
    """Return True when an automatic queue rebuild contains only blank defaults.

    The manual review page saves the complete queue, including reviewed rows. An
    automatic "create or refresh" action, in contrast, initializes every manual
    field to pending/blank. Distinguishing these cases prevents an automatic refresh
    from erasing work while still allowing new manual decisions to be written.
    """
    if df.empty or "segment_id" not in df.columns:
        return False

    statuses = df.get("manual_review_status", pd.Series("pending", index=df.index))
    statuses_blank = statuses.fillna("pending").astype(str).eq("pending").all()

    scene_types = df.get("manual_scene_type", pd.Series("pending", index=df.index))
    scene_types_blank = scene_types.fillna("pending").astype(str).eq("pending").all()

    commentary = df.get("manual_is_commentary", pd.Series(pd.NA, index=df.index))
    commentary_blank = commentary.isna().all()

    identities = df.get("manual_identity", pd.Series(pd.NA, index=df.index))
    identities_blank = identities.isna().all() | identities.fillna("").astype(str).str.strip().eq("").all()

    notes = df.get("manual_notes", pd.Series("", index=df.index))
    notes_blank = notes.fillna("").astype(str).str.strip().eq("").all()

    return bool(
        statuses_blank
        and scene_types_blank
        and commentary_blank
        and identities_blank
        and notes_blank
    )


def _preserve_review_annotations(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Preserve existing manual fields when an automatic queue is refreshed."""
    if (
        not path.name.endswith("_review_queue.parquet")
        or not path.exists()
        or not _looks_like_fresh_review_queue(df)
    ):
        return df

    try:
        existing = pd.read_parquet(path)
    except Exception:
        return df
    if existing.empty or "segment_id" not in existing.columns:
        return df

    preserved_columns = [
        "segment_id",
        *[column for column in _MANUAL_REVIEW_COLUMNS if column in existing.columns],
    ]
    preserved = existing[preserved_columns].drop_duplicates("segment_id", keep="last")
    refreshed = df.drop(
        columns=[column for column in _MANUAL_REVIEW_COLUMNS if column in df.columns]
    )
    refreshed = refreshed.merge(preserved, on="segment_id", how="left")

    if "manual_review_status" not in refreshed.columns:
        refreshed["manual_review_status"] = "pending"
    refreshed["manual_review_status"] = refreshed["manual_review_status"].fillna("pending")

    if "manual_scene_type" not in refreshed.columns:
        refreshed["manual_scene_type"] = "pending"
    refreshed["manual_scene_type"] = refreshed["manual_scene_type"].fillna("pending")

    if "manual_is_commentary" not in refreshed.columns:
        refreshed["manual_is_commentary"] = pd.Series(
            pd.NA, index=refreshed.index, dtype="boolean"
        )
    else:
        refreshed["manual_is_commentary"] = refreshed["manual_is_commentary"].astype(
            "boolean"
        )

    if "manual_identity" not in refreshed.columns:
        refreshed["manual_identity"] = pd.Series(pd.NA, index=refreshed.index, dtype="string")
    if "manual_notes" not in refreshed.columns:
        refreshed["manual_notes"] = ""
    refreshed["manual_notes"] = refreshed["manual_notes"].fillna("")
    return refreshed


def save_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        output = _preserve_review_annotations(df.copy(), path)
        output.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path.suffix}")
