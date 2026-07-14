from __future__ import annotations

import pandas as pd
import streamlit as st

from airtime_bias.io.artifact_history import (
    artifact_label,
    list_artifacts,
    load_manifest,
    preferred_artifact_index,
    write_manifest,
)
from airtime_bias.io.loaders import load_config, load_table
from airtime_bias.io.metadata import (
    add_video_paths,
    load_active_cast,
    load_episode_parts,
    load_episodes,
    validate_metadata_tables,
)
from airtime_bias.io.paths import INTERIM_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table
from airtime_bias.video.clip_utils import get_video_duration_seconds
from airtime_bias.video.scene_detection import detect_scenes
from airtime_bias.visualization.plots import (
    plot_duration_bucket_share,
    plot_scene_count_by_part,
    plot_scene_duration_by_part,
    plot_scene_duration_histogram,
    summarize_scene_durations,
)

ensure_project_dirs()

st.set_page_config(page_title="Scene Detection", page_icon="🎞️", layout="wide")
st.title("🎞️ Scene Detection")
st.markdown(
    """
Detect raw video shots and compare saved segmentation experiments. Every execution is
persisted locally and remains available after navigating away from this page.
"""
)

SCENE_DIR = INTERIM_DIR / "scenes"
config = load_config()
scene_defaults = config.get("scene_detection", {})

episodes = load_episodes()
episode_parts = load_episode_parts()
active_cast = load_active_cast()
problems = validate_metadata_tables(episodes, episode_parts, active_cast)

if problems:
    st.warning("Metadata is not ready yet. Go to Episode Setup first.")
    for problem in problems:
        st.write(f"- {problem}")
    st.stop()

parts_with_paths = add_video_paths(episode_parts, prefer_proxy=True)
if not parts_with_paths["selected_video_exists"].all():
    st.error("Some video files were not found. Fix video paths in Episode Setup first.")
    st.dataframe(
        parts_with_paths.loc[
            ~parts_with_paths["selected_video_exists"],
            ["episode_id", "part_id", "video_filename", "selected_video_path"],
        ],
        width="stretch",
    )
    st.stop()

episode_ids = sorted(episodes["episode_id"].astype(str).unique().tolist())
episode_id = st.selectbox("Select episode", episode_ids)
selected_parts_all = (
    parts_with_paths[parts_with_paths["episode_id"].astype(str) == episode_id]
    .copy()
    .sort_values("part_order")
)

st.subheader("Scene detection parameters")
col1, col2, col3 = st.columns(3)

with col1:
    threshold = st.slider(
        "PySceneDetect content threshold",
        min_value=10.0,
        max_value=80.0,
        value=float(scene_defaults.get("threshold", 44.0)),
        step=1.0,
        help="Higher values usually create fewer, longer segments.",
    )

with col2:
    min_scene_len_seconds = st.slider(
        "Minimum scene length between cuts (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=float(scene_defaults.get("min_scene_len_seconds", 2.0)),
        step=0.5,
        help=(
            "Converted to frames and passed to PySceneDetect. It constrains how close "
            "detected cuts may be; it does not delete short rows afterward."
        ),
    )

with col3:
    processing_mode = st.radio(
        "Processing mode",
        ["Single part calibration", "All episode parts"],
        index=0,
        help="Calibrate on one part before processing the complete episode.",
    )

if processing_mode == "Single part calibration":
    part_options = selected_parts_all["part_id"].astype(str).tolist()
    selected_part_id = st.selectbox("Part to process", part_options)
    selected_parts = selected_parts_all[
        selected_parts_all["part_id"].astype(str) == selected_part_id
    ].copy()
    output_scope = selected_part_id
else:
    selected_parts = selected_parts_all.copy()
    output_scope = "all_parts"

st.subheader("Parts to process")
st.dataframe(
    selected_parts[["episode_id", "part_id", "part_order", "selected_video_path"]],
    width="stretch",
)

safe_threshold = str(threshold).replace(".", "p")
safe_min_len = str(min_scene_len_seconds).replace(".", "p")
experiment_output_path = (
    SCENE_DIR
    / f"{episode_id}_{output_scope}_thr{safe_threshold}_min{safe_min_len}_segments.parquet"
)
latest_output_path = SCENE_DIR / f"{episode_id}_segments.parquet"
manifest_path = experiment_output_path.with_suffix(".manifest.json")


def _compute_offsets(parts: pd.DataFrame) -> dict[str, float]:
    """Compute global offsets for all parts in an episode."""
    offsets: dict[str, float] = {}
    offset_seconds = 0.0
    for _, part in parts.sort_values("part_order").iterrows():
        part_id = str(part["part_id"])
        offsets[part_id] = offset_seconds
        offset_seconds += get_video_duration_seconds(part["selected_video_path"])
    return offsets


def _render_scene_diagnostics(scenes: pd.DataFrame) -> None:
    """Render descriptive statistics and plots for detected scenes."""
    if scenes.empty:
        st.info("No scenes detected.")
        return

    summary = summarize_scene_durations(scenes)
    if not summary.empty:
        row = summary.iloc[0]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Scenes", f"{int(row['n_scenes']):,}")
        metric_columns[1].metric("Median duration", f"{row['median_duration']:.2f}s")
        metric_columns[2].metric("Mean duration", f"{row['mean_duration']:.2f}s")
        metric_columns[3].metric("Under 2s", f"{row['share_under_2s']:.1%}")

        detail_columns = st.columns(4)
        detail_columns[0].metric("Min", f"{row['min_duration']:.2f}s")
        detail_columns[1].metric("P25", f"{row['p25_duration']:.2f}s")
        detail_columns[2].metric("P75", f"{row['p75_duration']:.2f}s")
        detail_columns[3].metric("Max", f"{row['max_duration']:.2f}s")

        with st.expander("Full duration summary table", expanded=False):
            st.dataframe(summary.round(4), width="stretch")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Duration histogram", "Duration buckets", "Scenes by part", "Duration by part"]
    )
    with tab1:
        figure = plot_scene_duration_histogram(scenes)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
    with tab2:
        figure = plot_duration_bucket_share(scenes)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
    with tab3:
        figure = plot_scene_count_by_part(scenes)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
    with tab4:
        figure = plot_scene_duration_by_part(scenes)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")


if st.button("Run scene detection", type="primary"):
    scene_tables: list[pd.DataFrame] = []
    total_parts = len(selected_parts)
    all_part_offsets = _compute_offsets(selected_parts_all)

    status_rows = selected_parts[["part_id", "part_order", "selected_video_path"]].copy()
    status_rows["status"] = "pending"
    status_rows["segments_detected"] = 0
    status_rows["duration_seconds"] = None

    progress_bar = st.progress(0, text="Preparing scene detection...")
    status_table_placeholder = st.empty()

    with st.status("Running scene detection...", expanded=True) as status:
        status_table_placeholder.dataframe(status_rows, width="stretch")

        for index, (_, part) in enumerate(selected_parts.iterrows(), start=1):
            part_id = str(part["part_id"])
            part_order = int(part["part_order"])
            video_path = part["selected_video_path"]
            global_offset_seconds = all_part_offsets.get(part_id, 0.0)

            progress_bar.progress(
                int(((index - 1) / total_parts) * 100),
                text=f"Processing part {index}/{total_parts}: {part_id}",
            )
            status_rows.loc[status_rows["part_id"].astype(str) == part_id, "status"] = "running"
            status_table_placeholder.dataframe(status_rows, width="stretch")

            try:
                part_duration = get_video_duration_seconds(video_path)
                part_scenes = detect_scenes(
                    video_path=video_path,
                    episode_id=episode_id,
                    threshold=threshold,
                    min_scene_len_seconds=min_scene_len_seconds,
                    part_id=part_id,
                    part_order=part_order,
                    global_offset_seconds=global_offset_seconds,
                )
                scene_tables.append(part_scenes)
                part_mask = status_rows["part_id"].astype(str) == part_id
                status_rows.loc[part_mask, "status"] = "done"
                status_rows.loc[part_mask, "segments_detected"] = len(part_scenes)
                status_rows.loc[part_mask, "duration_seconds"] = round(part_duration, 2)
            except Exception as exc:
                status_rows.loc[
                    status_rows["part_id"].astype(str) == part_id, "status"
                ] = "error"
                status_table_placeholder.dataframe(status_rows, width="stretch")
                status.update(label="Scene detection failed.", state="error", expanded=True)
                st.exception(exc)
                st.stop()

            progress_bar.progress(
                int((index / total_parts) * 100),
                text=f"Completed part {index}/{total_parts}: {part_id}",
            )
            status_table_placeholder.dataframe(status_rows, width="stretch")

        scenes = pd.concat(scene_tables, ignore_index=True) if scene_tables else pd.DataFrame()
        save_table(scenes, experiment_output_path)
        if processing_mode == "All episode parts":
            save_table(scenes, latest_output_path)

        write_manifest(
            manifest_path,
            {
                "stage": "scene_detection",
                "output_table": str(experiment_output_path),
                "latest_episode_table": (
                    str(latest_output_path) if processing_mode == "All episode parts" else None
                ),
                "episode_id": episode_id,
                "scope": output_scope,
                "processing_mode": processing_mode,
                "threshold": float(threshold),
                "min_scene_len_seconds": float(min_scene_len_seconds),
                "parts": selected_parts["part_id"].astype(str).tolist(),
                "scene_count": int(len(scenes)),
            },
        )
        st.session_state["scene_detection_last_output"] = str(experiment_output_path)
        status.update(label="Scene detection complete.", state="complete", expanded=False)

    progress_bar.progress(100, text="Scene detection complete.")
    st.success(f"Detected {len(scenes):,} scenes and saved the execution for later review.")

st.divider()
st.subheader("Saved scene analyses")
scene_history = list_artifacts(SCENE_DIR, "*_segments.parquet")

if not scene_history:
    st.info("No saved scene analyses are available yet.")
else:
    preferred = st.session_state.get("scene_detection_last_output")
    selected_history_path = st.selectbox(
        "Analysis to visualize",
        options=scene_history,
        index=preferred_artifact_index(scene_history, preferred),
        format_func=lambda path: artifact_label(path, SCENE_DIR),
        key="scene_detection_history_selector",
    )
    selected_manifest = load_manifest(selected_history_path.with_suffix(".manifest.json"))
    if selected_manifest:
        with st.expander("Saved execution parameters", expanded=False):
            st.json(selected_manifest)

    saved_scenes = load_table(selected_history_path)
    st.caption(f"Loaded `{selected_history_path}`")
    _render_scene_diagnostics(saved_scenes)
    with st.expander("Detected scene table", expanded=False):
        st.dataframe(saved_scenes, width="stretch")
