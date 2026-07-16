from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from airtime_bias.io.artifact_history import artifact_label, list_artifacts
from airtime_bias.io.loaders import load_table
from airtime_bias.io.paths import VALIDATION_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table
from airtime_bias.validation.review_queue import (
    apply_manual_decision,
    build_identity_label_table,
    ensure_review_schema,
)

ensure_project_dirs()
st.set_page_config(page_title="Candidate Review", page_icon="✅", layout="wide")
st.title("✅ Manual Commentary and Identity Review")
st.markdown(
    """
Classify the scene type and identity as separate questions. A participant close-up can be a
negative commentary example while still becoming a valuable labeled face for future identity
matching. Every saved decision persists in the selected Parquet review queue.
"""
)

REVIEW_QUEUE_DIR = VALIDATION_DIR / "commentary_reviews"
IDENTITY_LABEL_DIR = VALIDATION_DIR / "identity_labels"
review_files = list_artifacts(REVIEW_QUEUE_DIR, "*_review_queue.parquet")
if not review_files:
    st.warning("No review queue exists yet. Create one from a saved candidate or fused analysis.")
    st.page_link("pages/5_Commentary_Candidates.py", label="Return to Commentary Candidates", icon="🎯")
    st.stop()

selected_review_path = st.selectbox(
    "Review queue", review_files, format_func=lambda path: artifact_label(path, REVIEW_QUEUE_DIR)
)
review_queue = ensure_review_schema(load_table(selected_review_path))
if review_queue.empty:
    st.error("The selected review queue is empty.")
    st.stop()
review_queue["segment_id"] = review_queue["segment_id"].astype(str)

source_frame_table: Path | None = None
source_values = review_queue.get("source_frame_table", pd.Series(dtype=str)).dropna()
if not source_values.empty:
    source_frame_table = Path(str(source_values.iloc[0]))
source_samples = load_table(source_frame_table) if source_frame_table is not None else pd.DataFrame()
if not source_samples.empty:
    source_samples = source_samples.copy()
    source_samples["segment_id"] = source_samples["segment_id"].astype(str)
    source_samples["frame_position"] = source_samples["frame_position"].astype(str)

scene_types = review_queue["manual_scene_type"].fillna("pending").astype(str)
metrics = st.columns(6)
metrics[0].metric("Queue size", f"{len(review_queue):,}")
metrics[1].metric("Pending", f"{int((scene_types == 'pending').sum()):,}")
metrics[2].metric("Commentary", f"{int((scene_types == 'commentary').sum()):,}")
metrics[3].metric(
    "Participant close-up", f"{int((scene_types == 'participant_closeup').sum()):,}"
)
metrics[4].metric("Other", f"{int((scene_types == 'other').sum()):,}")
metrics[5].metric("Uncertain", f"{int((scene_types == 'uncertain').sum()):,}")

st.subheader("Review filters")
filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
with filter_col_1:
    scene_type_filter = st.multiselect(
        "Scene type",
        ["pending", "commentary", "participant_closeup", "other", "uncertain"],
        ["pending", "uncertain"],
    )
with filter_col_2:
    tier_filter = st.multiselect(
        "Candidate tier", ["candidate", "review"], ["candidate", "review"]
    )
with filter_col_3:
    sort_mode = st.selectbox("Order", ["Highest score first", "Lowest score first", "Timeline"])

filtered = review_queue[
    review_queue["manual_scene_type"].astype(str).isin(scene_type_filter)
    & review_queue["candidate_tier"].astype(str).isin(tier_filter)
].copy()
if sort_mode == "Highest score first":
    filtered = filtered.sort_values("commentary_score", ascending=False)
elif sort_mode == "Lowest score first":
    filtered = filtered.sort_values("commentary_score", ascending=True)
else:
    sort_columns = [
        column
        for column in ("part_order", "segment_global_start_time", "global_start_time", "segment_id")
        if column in filtered.columns
    ]
    if sort_columns:
        filtered = filtered.sort_values(sort_columns)

if filtered.empty:
    st.success("No scenes match the current filters. Your saved progress remains intact.")
else:
    segment_options = filtered["segment_id"].tolist()
    selected_segment_id = st.selectbox(
        "Scene to review",
        segment_options,
        format_func=lambda segment_id: (
            f"{segment_id} · score "
            f"{float(filtered.loc[filtered['segment_id'] == segment_id, 'commentary_score'].iloc[0]):.3f}"
        ),
    )
    selected_row = review_queue[review_queue["segment_id"] == selected_segment_id].iloc[0]
    st.markdown(
        f"### `{selected_segment_id}` · score `{float(selected_row['commentary_score']):.3f}` · "
        f"tier `{selected_row['candidate_tier']}` · "
        f"source `{selected_row.get('candidate_source', 'face')}`"
    )

    segment_samples = (
        source_samples[source_samples["segment_id"] == selected_segment_id].copy()
        if not source_samples.empty
        else pd.DataFrame()
    )
    label_path = selected_row.get("lower_third_representative_frame_path")
    has_label_image = isinstance(label_path, str) and Path(label_path).exists()
    image_count = 4 if has_label_image else 3
    image_columns = st.columns(image_count)
    if not segment_samples.empty:
        segment_samples = segment_samples.set_index("frame_position")
        for column, position in zip(image_columns[:3], ["start", "middle", "end"]):
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
    if has_label_image:
        with image_columns[3]:
            st.image(label_path, caption="lower-third signal", width="stretch")

    feature_columns = [
        column
        for column in (
            "face_commentary_score",
            "commentary_score",
            "face_presence_ratio",
            "single_face_ratio",
            "median_face_area_ratio",
            "median_face_center_distance",
            "geometry_stability_score",
            "lower_third_max_visual_score",
            "lower_third_matched_participant",
            "segment_duration",
        )
        if column in review_queue.columns
    ]
    if feature_columns:
        with st.expander("Candidate features", expanded=False):
            st.dataframe(pd.DataFrame([selected_row[feature_columns]]), width="stretch")

    label_map = {
        "Pending": "pending",
        "Commentary": "commentary",
        "Participant close-up": "participant_closeup",
        "Other": "other",
        "Uncertain": "uncertain",
    }
    reverse_map = {value: label for label, value in label_map.items()}
    current_scene_type = str(selected_row.get("manual_scene_type", "pending"))
    decision_col_1, decision_col_2 = st.columns(2)
    with decision_col_1:
        decision_label = st.radio(
            "Manual scene type",
            list(label_map),
            index=list(label_map).index(reverse_map.get(current_scene_type, "Pending")),
            horizontal=True,
            key=f"scene_type_{selected_review_path.stem}_{selected_segment_id}",
        )
        identity_value = selected_row.get("manual_identity")
        identity = st.text_input(
            "Participant identity (optional)",
            "" if pd.isna(identity_value) else str(identity_value),
            key=f"identity_{selected_review_path.stem}_{selected_segment_id}",
            help="Use this for both commentary and participant close-up scenes.",
        )
    with decision_col_2:
        notes_value = selected_row.get("manual_notes")
        notes = st.text_area(
            "Review notes",
            "" if pd.isna(notes_value) else str(notes_value),
            key=f"notes_{selected_review_path.stem}_{selected_segment_id}",
        )

    if st.button("Save decision", type="primary"):
        updated = apply_manual_decision(
            review_queue,
            segment_id=selected_segment_id,
            scene_type=label_map[decision_label],
            identity=identity or None,
            notes=notes,
        )
        save_table(updated, selected_review_path)
        st.success("Decision saved. You may leave the page and continue later.")
        st.rerun()

st.divider()
st.subheader("Identity-label dataset")
named_count = int(review_queue["manual_identity"].notna().sum())
st.write(
    f"The current queue contains **{named_count:,}** named scenes. Commentary and participant "
    "close-ups can both contribute frames to a future identity model while retaining their scene type."
)
identity_output_path = IDENTITY_LABEL_DIR / f"{selected_review_path.stem}_identity_labels.parquet"
if st.button("Build or refresh identity-label table"):
    labels = build_identity_label_table(review_queue, source_samples)
    save_table(labels, identity_output_path)
    st.success(f"Saved {len(labels):,} labeled frames to `{identity_output_path}`")

with st.expander("Complete persisted review table", expanded=False):
    st.dataframe(review_queue, width="stretch")
