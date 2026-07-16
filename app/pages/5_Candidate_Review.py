from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from airtime_bias.io.artifact_history import artifact_label, list_artifacts
from airtime_bias.io.loaders import load_table
from airtime_bias.io.paths import VALIDATION_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table

ensure_project_dirs()

st.set_page_config(page_title="Candidate Review", page_icon="✅", layout="wide")
st.title("✅ Manual Commentary Candidate Review")
st.markdown(
    """
Validate retained commentary candidates before participant identity matching. Decisions are
saved directly to the selected review queue and remain available after leaving the page.
"""
)

REVIEW_QUEUE_DIR = VALIDATION_DIR / "commentary_reviews"
review_files = list_artifacts(REVIEW_QUEUE_DIR, "*_review_queue.parquet")

if not review_files:
    st.warning(
        "No review queue is available. Open Commentary Candidates and create a queue from "
        "a saved analysis first."
    )
    st.page_link(
        "pages/4_Commentary_Candidates.py",
        label="Return to Commentary Candidates",
        icon="🎯",
    )
    st.stop()

selected_review_path = st.selectbox(
    "Review queue",
    options=review_files,
    format_func=lambda path: artifact_label(path, REVIEW_QUEUE_DIR),
)

review_queue = load_table(selected_review_path)
if review_queue.empty:
    st.error("The selected review queue is empty.")
    st.stop()

required_columns = {"segment_id", "candidate_tier", "commentary_score"}
missing_columns = required_columns - set(review_queue.columns)
if missing_columns:
    st.error(f"Review queue is missing required columns: {sorted(missing_columns)}")
    st.stop()

review_queue = review_queue.copy()
if "manual_review_status" not in review_queue.columns:
    review_queue["manual_review_status"] = "pending"
if "manual_is_commentary" not in review_queue.columns:
    review_queue["manual_is_commentary"] = pd.Series(
        pd.NA,
        index=review_queue.index,
        dtype="boolean",
    )
else:
    review_queue["manual_is_commentary"] = review_queue["manual_is_commentary"].astype(
        "boolean"
    )
if "manual_notes" not in review_queue.columns:
    review_queue["manual_notes"] = ""
if "manual_identity" not in review_queue.columns:
    review_queue["manual_identity"] = pd.Series(
        pd.NA,
        index=review_queue.index,
        dtype="string",
    )

review_queue["segment_id"] = review_queue["segment_id"].astype(str)
status_series = review_queue["manual_review_status"].fillna("pending").astype(str)
confirmed_mask = review_queue["manual_is_commentary"].eq(True).fillna(False)
rejected_mask = review_queue["manual_is_commentary"].eq(False).fillna(False)
uncertain_mask = status_series.eq("uncertain")

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Queue size", f"{len(review_queue):,}")
metric_2.metric("Pending", f"{int((status_series == 'pending').sum()):,}")
metric_3.metric("Confirmed commentary", f"{int(confirmed_mask.sum()):,}")
metric_4.metric("Rejected", f"{int(rejected_mask.sum()):,}")
metric_5.metric("Uncertain", f"{int(uncertain_mask.sum()):,}")

source_frame_table: Path | None = None
if "source_frame_table" in review_queue.columns:
    source_values = review_queue["source_frame_table"].dropna()
    if not source_values.empty:
        source_frame_table = Path(str(source_values.iloc[0]))

source_samples = load_table(source_frame_table) if source_frame_table is not None else pd.DataFrame()
if not source_samples.empty:
    source_samples = source_samples.copy()
    source_samples["segment_id"] = source_samples["segment_id"].astype(str)
    source_samples["frame_position"] = source_samples["frame_position"].astype(str)

st.subheader("Review filters")
filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

with filter_col_1:
    status_filter = st.multiselect(
        "Review status",
        options=["pending", "reviewed", "uncertain"],
        default=["pending", "uncertain"],
    )

with filter_col_2:
    tier_filter = st.multiselect(
        "Candidate tier",
        options=["candidate", "review"],
        default=["candidate", "review"],
    )

with filter_col_3:
    sort_mode = st.selectbox(
        "Order",
        options=["Highest score first", "Lowest score first", "Timeline"],
    )

filtered_queue = review_queue[
    review_queue["manual_review_status"].fillna("pending").astype(str).isin(status_filter)
    & review_queue["candidate_tier"].astype(str).isin(tier_filter)
].copy()

if sort_mode == "Highest score first":
    filtered_queue = filtered_queue.sort_values("commentary_score", ascending=False)
elif sort_mode == "Lowest score first":
    filtered_queue = filtered_queue.sort_values("commentary_score", ascending=True)
else:
    sort_columns = [
        column
        for column in ("part_order", "global_start_time", "segment_id")
        if column in filtered_queue.columns
    ]
    if sort_columns:
        filtered_queue = filtered_queue.sort_values(sort_columns)

if filtered_queue.empty:
    st.success("No scenes match the current filters.")
    st.dataframe(review_queue, width="stretch")
    st.stop()

segment_options = filtered_queue["segment_id"].tolist()
selected_segment_id = st.selectbox(
    "Scene to review",
    options=segment_options,
    format_func=lambda segment_id: (
        f"{segment_id} · score "
        f"{float(filtered_queue.loc[filtered_queue['segment_id'] == segment_id, 'commentary_score'].iloc[0]):.3f}"
    ),
)

selected_row = review_queue[review_queue["segment_id"] == selected_segment_id].iloc[0]

st.markdown(
    f"### `{selected_segment_id}` · score `{float(selected_row['commentary_score']):.3f}` "
    f"· tier `{selected_row['candidate_tier']}`"
)

if source_samples.empty:
    st.warning("The source frame table could not be loaded for this queue.")
else:
    segment_samples = source_samples[source_samples["segment_id"] == selected_segment_id].copy()
    segment_samples = segment_samples.set_index("frame_position")
    image_columns = st.columns(3)
    for column, position in zip(image_columns, ["start", "middle", "end"]):
        with column:
            if position not in segment_samples.index:
                st.caption(f"{position}: unavailable")
                continue
            sample = segment_samples.loc[position]
            if isinstance(sample, pd.DataFrame):
                sample = sample.iloc[0]
            frame_path = sample.get("frame_path")
            if isinstance(frame_path, str) and Path(frame_path).exists():
                st.image(frame_path, caption=position, width="stretch")
            else:
                st.caption(f"{position}: file not found")

feature_columns = [
    column
    for column in (
        "face_presence_ratio",
        "single_face_ratio",
        "median_face_area_ratio",
        "median_face_center_distance",
        "geometry_stability_score",
        "segment_duration",
    )
    if column in review_queue.columns
]
if feature_columns:
    with st.expander("Candidate features", expanded=False):
        st.dataframe(pd.DataFrame([selected_row[feature_columns]]), width="stretch")

current_status = str(selected_row.get("manual_review_status", "pending"))
manual_value = selected_row.get("manual_is_commentary")
current_decision = "Pending"
if current_status == "uncertain":
    current_decision = "Uncertain"
elif pd.notna(manual_value):
    current_decision = "Commentary" if bool(manual_value) else "Not commentary"

identity_value = selected_row.get("manual_identity")
identity_text = "" if pd.isna(identity_value) else str(identity_value)
notes_value = selected_row.get("manual_notes")
notes_text = "" if pd.isna(notes_value) else str(notes_value)

review_col_1, review_col_2 = st.columns(2)
with review_col_1:
    decision = st.radio(
        "Manual decision",
        options=["Pending", "Commentary", "Not commentary", "Uncertain"],
        index=["Pending", "Commentary", "Not commentary", "Uncertain"].index(
            current_decision
        ),
        horizontal=True,
        key=f"decision_{selected_review_path.stem}_{selected_segment_id}",
    )
    manual_identity = st.text_input(
        "Participant identity (optional for now)",
        value=identity_text,
        key=f"identity_{selected_review_path.stem}_{selected_segment_id}",
    )

with review_col_2:
    manual_notes = st.text_area(
        "Review notes",
        value=notes_text,
        key=f"notes_{selected_review_path.stem}_{selected_segment_id}",
    )

if st.button("Save decision", type="primary"):
    segment_mask = review_queue["segment_id"] == selected_segment_id

    if decision == "Commentary":
        review_queue.loc[segment_mask, "manual_review_status"] = "reviewed"
        review_queue.loc[segment_mask, "manual_is_commentary"] = True
    elif decision == "Not commentary":
        review_queue.loc[segment_mask, "manual_review_status"] = "reviewed"
        review_queue.loc[segment_mask, "manual_is_commentary"] = False
    elif decision == "Uncertain":
        review_queue.loc[segment_mask, "manual_review_status"] = "uncertain"
        review_queue.loc[segment_mask, "manual_is_commentary"] = pd.NA
    else:
        review_queue.loc[segment_mask, "manual_review_status"] = "pending"
        review_queue.loc[segment_mask, "manual_is_commentary"] = pd.NA

    review_queue.loc[segment_mask, "manual_identity"] = manual_identity or pd.NA
    review_queue.loc[segment_mask, "manual_notes"] = manual_notes
    save_table(review_queue, selected_review_path)
    st.success("Decision saved to the review queue.")
    st.rerun()

st.divider()
with st.expander("Complete persisted review table", expanded=False):
    st.dataframe(review_queue, width="stretch")

st.info(
    "After enough scenes have been manually validated, the next methodological step is "
    "calibrating recall/precision and then restricting participant identity matching to "
    "confirmed commentary scenes."
)
