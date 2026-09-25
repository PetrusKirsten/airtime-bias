from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from airtime_bias.io.artifact_history import (
    artifact_label,
    list_artifacts,
    load_manifest,
    parameter_fingerprint,
    preferred_artifact_index,
    write_manifest,
)
from airtime_bias.io.loaders import load_config, load_table
from airtime_bias.io.paths import INTERIM_DIR, PROCESSED_DIR, VALIDATION_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table
from airtime_bias.lower_third.integration import attach_lower_third_signals
from airtime_bias.validation.review_queue import build_or_refresh_review_queue
from airtime_bias.visualization.candidate_plots import (
    plot_candidate_tier_counts,
    plot_commentary_score_distribution,
)

ensure_project_dirs()
st.set_page_config(page_title="Signal Fusion", page_icon="🧩", layout="wide")
st.title("🧩 Commentary Signal Fusion")
st.markdown(
    """
Combine the face-based candidate analysis with independent lower-third events. The union route
can promote short or visually difficult commentary scenes while keeping face-only, label-only,
and combined evidence explicit in the output table.
"""
)

config = load_config()
commentary_defaults = config.get("commentary_detection", {})
FACE_CANDIDATE_DIR = PROCESSED_DIR / "commentary_candidates"
FUSED_DIR = FACE_CANDIDATE_DIR / "fused"
LOWER_EVENT_DIR = INTERIM_DIR / "lower_thirds" / "events"
REVIEW_QUEUE_DIR = VALIDATION_DIR / "commentary_reviews"

face_files = [
    path
    for path in list_artifacts(FACE_CANDIDATE_DIR, "*_commentary_candidates.parquet")
    if FUSED_DIR not in path.parents
]
lower_files = list_artifacts(LOWER_EVENT_DIR, "*_lower_third_events.parquet")

if not face_files:
    st.warning("No face-based candidate analysis is available. Run Commentary Candidates first.")
    st.stop()
if not lower_files:
    st.warning("No lower-third event table is available. Run Lower-Third Detection first.")
    st.stop()

selection_col_1, selection_col_2 = st.columns(2)
with selection_col_1:
    selected_face_path = st.selectbox(
        "Face candidate analysis",
        face_files,
        format_func=lambda path: artifact_label(path, FACE_CANDIDATE_DIR),
    )
with selection_col_2:
    selected_lower_path = st.selectbox(
        "Lower-third event analysis",
        lower_files,
        format_func=lambda path: artifact_label(path, LOWER_EVENT_DIR),
    )

face_candidates = load_table(selected_face_path)
lower_events = load_table(selected_lower_path)
if face_candidates.empty:
    st.error("The selected face candidate table is empty.")
    st.stop()

source_values = face_candidates.get("source_frame_table", pd.Series(dtype=str)).dropna()
if source_values.empty:
    st.error(
        "The face candidate table does not record its source frame table. Re-run Commentary "
        "Candidates with the current persisted-run implementation."
    )
    st.stop()
source_frame_path = Path(str(source_values.iloc[0]))
frame_samples = load_table(source_frame_path)
if frame_samples.empty:
    st.error(f"The source frame table could not be loaded: `{source_frame_path}`")
    st.stop()

face_episode_values = face_candidates.get("episode_id", pd.Series(dtype=str)).dropna().astype(str).unique()
lower_episode_values = (
    lower_events.get("episode_id", pd.Series(dtype=str)).dropna().astype(str).unique()
    if not lower_events.empty
    else []
)
if len(face_episode_values) and len(lower_episode_values):
    if set(face_episode_values).isdisjoint(set(lower_episode_values)):
        st.error("The selected face candidates and lower-third events belong to different episodes.")
        st.stop()

st.subheader("Fusion thresholds")
threshold_col_1, threshold_col_2, threshold_col_3 = st.columns(3)
with threshold_col_1:
    review_threshold = st.slider(
        "Review threshold",
        min_value=0.10,
        max_value=0.85,
        value=float(commentary_defaults.get("review_threshold", 0.50)),
        step=0.01,
    )
with threshold_col_2:
    candidate_threshold = st.slider(
        "Candidate threshold",
        min_value=0.20,
        max_value=0.98,
        value=max(
            float(commentary_defaults.get("candidate_threshold", 0.70)),
            review_threshold,
        ),
        step=0.01,
    )
with threshold_col_3:
    matched_name_threshold = st.slider(
        "OCR name-match threshold",
        min_value=0.40,
        max_value=1.00,
        value=float(commentary_defaults.get("lower_third_name_threshold", 0.68)),
        step=0.01,
        help="A lower-third with a sufficiently strong participant-name match is promoted directly.",
    )

if review_threshold > candidate_threshold:
    st.error("The review threshold cannot exceed the candidate threshold.")
    st.stop()

run_parameters = {
    "face_candidate_table": str(selected_face_path),
    "source_frame_table": str(source_frame_path),
    "lower_third_event_table": str(selected_lower_path),
    "review_threshold": float(review_threshold),
    "candidate_threshold": float(candidate_threshold),
    "matched_name_threshold": float(matched_name_threshold),
}
config_token = parameter_fingerprint(run_parameters)
face_run_token = selected_face_path.name.removesuffix("_commentary_candidates.parquet")
lower_run_token = selected_lower_path.name.removesuffix("_lower_third_events.parquet")
episode_token = str(face_episode_values[0]) if len(face_episode_values) else "episode"
run_id = f"{episode_token}__fusion__{config_token}"
fused_path = FUSED_DIR / f"{run_id}_commentary_candidates.parquet"
manifest_path = fused_path.with_suffix(".manifest.json")
review_path = REVIEW_QUEUE_DIR / f"{run_id}_review_queue.parquet"

with st.expander("Execution details", expanded=False):
    st.write(f"**Face run:** `{face_run_token}`")
    st.write(f"**Lower-third run:** `{lower_run_token}`")
    st.write(f"**Source frames:** `{source_frame_path}`")
    st.write(f"**Fused candidates:** `{fused_path}`")
    st.json(run_parameters)


def _render_fused_results(fused: pd.DataFrame, *, run_key: str, output_path: Path) -> None:
    if fused.empty:
        st.info("No fused candidate rows were produced.")
        return

    tiers = fused["candidate_tier"].astype(str).value_counts()
    source_counts = fused.get("candidate_source", pd.Series("face", index=fused.index)).astype(str)
    lower_supported = fused.get(
        "lower_third_detected", pd.Series(False, index=fused.index)
    ).fillna(False).astype(bool)
    short_mask = pd.to_numeric(fused.get("segment_duration"), errors="coerce").fillna(0) < 2.0
    promoted = (
        fused.get("face_candidate_tier", fused["candidate_tier"]).astype(str).eq("reject")
        & fused["candidate_tier"].astype(str).isin(["candidate", "review"])
    )

    metrics = st.columns(6)
    metrics[0].metric("Candidates", f"{int(tiers.get('candidate', 0)):,}")
    metrics[1].metric("Review", f"{int(tiers.get('review', 0)):,}")
    metrics[2].metric("Rejects", f"{int(tiers.get('reject', 0)):,}")
    metrics[3].metric("Lower-third supported", f"{int(lower_supported.sum()):,}")
    metrics[4].metric("Promoted by label route", f"{int(promoted.sum()):,}")
    metrics[5].metric(
        "Sub-2s retained",
        f"{int((short_mask & fused['candidate_tier'].isin(['candidate', 'review'])).sum()):,}",
    )

    diagnostics_tab, gallery_tab, table_tab = st.tabs(
        ["Diagnostics", "Lower-third-supported gallery", "Fused table"]
    )
    with diagnostics_tab:
        plot_col_1, plot_col_2 = st.columns(2)
        with plot_col_1:
            figure = plot_commentary_score_distribution(fused)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
        with plot_col_2:
            figure = plot_candidate_tier_counts(fused)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")

        source_summary = (
            source_counts.value_counts().rename_axis("candidate_source").reset_index(name="scene_count")
        )
        st.markdown("#### Evidence source")
        st.dataframe(source_summary, width="stretch")

    with gallery_tab:
        supported = fused[lower_supported].sort_values("commentary_score", ascending=False)
        if supported.empty:
            st.info("No fused scenes overlap a lower-third event.")
        else:
            preview_count = st.slider(
                "Scenes to preview",
                min_value=1,
                max_value=min(40, len(supported)),
                value=min(12, len(supported)),
                key=f"fusion_gallery_{run_key}",
            )
            for _, row in supported.head(preview_count).iterrows():
                st.markdown(
                    f"**{row['segment_id']}** · score `{row['commentary_score']:.3f}` · "
                    f"tier `{row['candidate_tier']}` · source `{row.get('candidate_source', 'face')}`"
                )
                image_col, info_col = st.columns([3, 1])
                with image_col:
                    image_path = row.get("lower_third_representative_frame_path")
                    if isinstance(image_path, str) and Path(image_path).exists():
                        st.image(image_path, width="stretch")
                    else:
                        st.caption("Representative lower-third frame unavailable.")
                with info_col:
                    st.metric(
                        "Label score",
                        f"{float(row.get('lower_third_max_visual_score', 0)):.3f}",
                    )
                    st.metric(
                        "Overlap",
                        f"{float(row.get('lower_third_overlap_seconds', 0)):.2f}s",
                    )
                    participant = row.get("lower_third_matched_participant")
                    if pd.notna(participant):
                        st.success(str(participant))
                    if float(row.get("segment_duration", 0) or 0) < 2.0:
                        st.warning("Sub-2-second scene")

    with table_tab:
        st.dataframe(fused, width="stretch")

    st.markdown("### Persisted manual review queue")
    st.write(
        "Refreshing this queue preserves all existing manual scene types, identities, and notes "
        "by `segment_id`."
    )
    if st.button(
        "Create or safely refresh review queue", type="primary", key=f"queue_{run_key}"
    ):
        existing = load_table(review_path)
        queue = build_or_refresh_review_queue(
            fused,
            existing_queue=existing,
            source_candidate_table=str(output_path),
            source_frame_table=str(source_frame_path),
            review_run_id=run_key,
        )
        save_table(queue, review_path)
        st.success(
            f"Saved {len(queue):,} retained scenes to `{review_path}` without erasing prior decisions."
        )

    if review_path.exists():
        queue = load_table(review_path)
        reviewed_count = int(
            queue.get("manual_review_status", pd.Series(dtype=str))
            .astype(str)
            .eq("reviewed")
            .sum()
        )
        st.info(f"Review queue available: {len(queue):,} scenes · {reviewed_count:,} reviewed.")
        st.page_link("pages/5_Candidate_Review.py", label="Continue manual review", icon="✅")


if st.button("Fuse face and lower-third signals", type="primary"):
    with st.status("Combining commentary signals...", expanded=True) as status:
        try:
            fused = attach_lower_third_signals(
                face_candidates,
                frame_samples=frame_samples,
                lower_third_events=lower_events,
                review_threshold=float(review_threshold),
                candidate_threshold=float(candidate_threshold),
                matched_name_threshold=float(matched_name_threshold),
            )
            fused["source_face_candidate_table"] = str(selected_face_path)
            fused["source_lower_third_event_table"] = str(selected_lower_path)
            fused["source_frame_table"] = str(source_frame_path)
            fused["fusion_run_id"] = run_id
            fused["fusion_config_id"] = config_token
            save_table(fused, fused_path)
            write_manifest(
                manifest_path,
                {
                    "stage": "commentary_signal_fusion",
                    "run_id": run_id,
                    "output_table": str(fused_path),
                    "face_run": face_run_token,
                    "lower_third_run": lower_run_token,
                    "scene_count": int(len(fused)),
                    "lower_third_supported_count": int(
                        fused["lower_third_detected"].fillna(False).sum()
                    ),
                    "parameters": run_parameters,
                },
            )
            st.session_state["signal_fusion_last_output"] = str(fused_path)
        except Exception as exc:
            status.update(label="Signal fusion failed.", state="error", expanded=True)
            st.exception(exc)
            st.stop()
        status.update(label="Signal fusion complete.", state="complete", expanded=False)
    st.success("The fused analysis was saved and remains available below.")

st.divider()
st.subheader("Saved fused analyses")
history = list_artifacts(FUSED_DIR, "*_commentary_candidates.parquet")
if not history:
    st.info("No fused analysis has been saved yet.")
else:
    preferred = st.session_state.get("signal_fusion_last_output")
    selected_fused_path = st.selectbox(
        "Fused analysis to visualize",
        history,
        index=preferred_artifact_index(history, preferred),
        format_func=lambda path: artifact_label(path, FUSED_DIR),
        key="signal_fusion_history_selector",
    )
    selected_manifest = load_manifest(selected_fused_path.with_suffix(".manifest.json"))
    if selected_manifest:
        with st.expander("Saved execution parameters", expanded=False):
            st.json(selected_manifest)
    saved_fused = load_table(selected_fused_path)
    selected_run_id = selected_fused_path.name.removesuffix("_commentary_candidates.parquet")
    st.caption(f"Loaded `{selected_fused_path}`")
    _render_fused_results(saved_fused, run_key=selected_run_id, output_path=selected_fused_path)
