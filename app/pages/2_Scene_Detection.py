from __future__ import annotations

import pandas as pd
import streamlit as st

from airtime_bias.io.loaders import load_table
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
This page detects raw video shots/scenes and helps you evaluate whether the segmentation
is useful for later commentary/talking-head detection.
"""
)

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
    st.error("Some video files were not found. Fix video paths in Episode Setup before scene detection.")
    st.dataframe(
        parts_with_paths.loc[
            ~parts_with_paths["selected_video_exists"],
            ["episode_id", "part_id", "video_filename", "selected_video_path"],
        ],
        use_container_width=True,
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
        value=40.0,
        step=1.0,
        help="Higher values usually create fewer, longer segments. Lower values detect more cuts.",
    )

with col2:
    min_scene_len_seconds = st.slider(
        "Minimum scene length between cuts (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help=(
            "Converted to frames and passed to PySceneDetect. "
            "This reduces very dense cut detection, but does not post-delete rows."
        ),
    )

with col3:
    processing_mode = st.radio(
        "Processing mode",
        ["Single part calibration", "All episode parts"],
        index=0,
        help="Use single part calibration while tuning thresholds. Process all parts only after parameters look reasonable.",
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

st.caption(
    "Suggestion for calibration: try threshold 35, 40, 45 with min scene length 1.5–2.5 seconds."
)

st.subheader("Parts to process")
st.dataframe(
    selected_parts[["episode_id", "part_id", "part_order", "selected_video_path"]],
    use_container_width=True,
)

safe_threshold = str(threshold).replace(".", "p")
safe_min_len = str(min_scene_len_seconds).replace(".", "p")

experiment_output_path = (
    INTERIM_DIR
    / "scenes"
    / f"{episode_id}_{output_scope}_thr{safe_threshold}_min{safe_min_len}_segments.parquet"
)

latest_output_path = INTERIM_DIR / "scenes" / f"{episode_id}_segments.parquet"


def _compute_offsets(parts: pd.DataFrame) -> dict[str, float]:
    """Compute global offsets for all parts in an episode."""
    offsets = {}
    offset_seconds = 0.0

    ordered_parts = parts.sort_values("part_order")

    for _, part in ordered_parts.iterrows():
        part_id = str(part["part_id"])
        video_path = part["selected_video_path"]

        offsets[part_id] = offset_seconds
        offset_seconds += get_video_duration_seconds(video_path)

    return offsets


def _render_scene_diagnostics(scenes: pd.DataFrame) -> None:
    """Render descriptive statistics and plots for detected scenes."""
    if scenes.empty:
        st.info("No scenes detected.")
        return

    st.subheader("Segmentation diagnostics")

    summary = summarize_scene_durations(scenes)

    if not summary.empty:
        row = summary.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scenes", f"{int(row['n_scenes'])}")
        col2.metric("Median duration", f"{row['median_duration']:.2f}s")
        col3.metric("Mean duration", f"{row['mean_duration']:.2f}s")
        col4.metric("Under 2s", f"{row['share_under_2s'] * 100:.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Min", f"{row['min_duration']:.2f}s")
        col6.metric("P25", f"{row['p25_duration']:.2f}s")
        col7.metric("P75", f"{row['p75_duration']:.2f}s")
        col8.metric("Max", f"{row['max_duration']:.2f}s")

        with st.expander("Full duration summary table", expanded=False):
            st.dataframe(summary.round(4), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Duration histogram",
            "Duration buckets",
            "Scenes by part",
            "Duration by part",
        ]
    )

    with tab1:
        fig = plot_scene_duration_histogram(scenes)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_duration_bucket_share(scenes)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_scene_count_by_part(scenes)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_scene_duration_by_part(scenes)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


if st.button("Run scene detection"):
    scene_tables = []
    total_parts = len(selected_parts)

    all_part_offsets = _compute_offsets(selected_parts_all)

    status_rows = selected_parts[["part_id", "part_order", "selected_video_path"]].copy()
    status_rows["status"] = "pending"
    status_rows["segments_detected"] = 0
    status_rows["duration_seconds"] = None

    progress_bar = st.progress(0, text="Preparing scene detection...")
    status_table_placeholder = st.empty()

    with st.status("Running scene detection...", expanded=True) as status:
        status_table_placeholder.dataframe(status_rows, use_container_width=True)

        for index, (_, part) in enumerate(selected_parts.iterrows(), start=1):
            part_id = str(part["part_id"])
            part_order = int(part["part_order"])
            video_path = part["selected_video_path"]
            global_offset_seconds = all_part_offsets.get(part_id, 0.0)

            st.write(f"Processing part {index}/{total_parts}: `{part_id}`")

            progress_bar.progress(
                int(((index - 1) / total_parts) * 100),
                text=f"Processing part {index}/{total_parts}: {part_id}",
            )

            status_rows.loc[
                status_rows["part_id"].astype(str) == part_id,
                "status",
            ] = "running"

            status_table_placeholder.dataframe(status_rows, use_container_width=True)

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

                status_rows.loc[
                    status_rows["part_id"].astype(str) == part_id,
                    "status",
                ] = "done"

                status_rows.loc[
                    status_rows["part_id"].astype(str) == part_id,
                    "segments_detected",
                ] = len(part_scenes)

                status_rows.loc[
                    status_rows["part_id"].astype(str) == part_id,
                    "duration_seconds",
                ] = round(part_duration, 2)

                st.write(f"Detected {len(part_scenes)} scenes in `{part_id}`.")

            except Exception as exc:
                status_rows.loc[
                    status_rows["part_id"].astype(str) == part_id,
                    "status",
                ] = "error"

                status_table_placeholder.dataframe(status_rows, use_container_width=True)

                status.update(
                    label="Scene detection failed.",
                    state="error",
                    expanded=True,
                )

                st.error(f"Error while processing `{part_id}`: {exc}")
                st.stop()

            progress_bar.progress(
                int((index / total_parts) * 100),
                text=f"Completed part {index}/{total_parts}: {part_id}",
            )

            status_table_placeholder.dataframe(status_rows, use_container_width=True)

        if scene_tables:
            scenes = pd.concat(scene_tables, ignore_index=True)
        else:
            scenes = pd.DataFrame()

        save_table(scenes, experiment_output_path)

        if processing_mode == "All episode parts":
            save_table(scenes, latest_output_path)

        status.update(
            label="Scene detection complete.",
            state="complete",
            expanded=False,
        )

    progress_bar.progress(100, text="Scene detection complete.")

    st.success(f"Detected {len(scenes)} scenes across {total_parts} video part(s).")

    if processing_mode == "All episode parts":
        st.info(f"Saved latest full-episode scene table to: `{latest_output_path}`")

    st.info(f"Saved experiment output to: `{experiment_output_path}`")

    _render_scene_diagnostics(scenes)

    st.subheader("Detected scenes")
    st.dataframe(scenes, use_container_width=True)

else:
    existing = load_table(latest_output_path)

    if not existing.empty:
        st.subheader("Existing latest full-episode scene table")
        st.dataframe(existing, use_container_width=True)
        _render_scene_diagnostics(existing)
    else:
        st.info("No latest full-episode scene table found yet for this episode.")