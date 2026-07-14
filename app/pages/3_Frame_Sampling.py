from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from airtime_bias.io.loaders import load_table
from airtime_bias.io.paths import INTERIM_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table
from airtime_bias.video.frame_sampling import (
    SUPPORTED_FRAME_POSITIONS,
    sample_frames_from_scenes,
)

ensure_project_dirs()

st.set_page_config(
    page_title="Frame Sampling",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Frame Sampling")
st.markdown(
    """
Extract representative frames from each detected scene before face detection and
commentary candidate scoring. By default, Airtime Bias samples frames near the
**start**, **middle**, and **end** of every scene while avoiding exact cut boundaries.
"""
)

SCENE_DIR = INTERIM_DIR / "scenes"
SAMPLED_FRAMES_DIR = INTERIM_DIR / "sampled_frames"
TABLE_DIR = SAMPLED_FRAMES_DIR / "tables"

scene_files = sorted(SCENE_DIR.glob("*_segments.parquet"))
if not scene_files:
    st.warning("No scene tables found. Run Scene Detection first.")
    st.stop()

selected_scene_path = st.selectbox(
    "Scene table",
    options=scene_files,
    format_func=lambda path: path.name,
)

scenes = load_table(selected_scene_path)
if scenes.empty:
    st.error("The selected scene table is empty.")
    st.stop()

required_columns = {
    "episode_id",
    "part_id",
    "segment_id",
    "video_path",
    "part_start_time",
    "part_end_time",
    "global_start_time",
    "duration",
}
missing_columns = required_columns - set(scenes.columns)
if missing_columns:
    st.error(f"Scene table is missing required columns: {sorted(missing_columns)}")
    st.stop()

scenes = scenes.copy()
scenes["part_id"] = scenes["part_id"].astype(str)
scenes["episode_id"] = scenes["episode_id"].astype(str)

st.subheader("Sampling scope")
col_scope_1, col_scope_2 = st.columns(2)

with col_scope_1:
    episode_options = sorted(scenes["episode_id"].unique().tolist())
    episode_id = st.selectbox("Episode", episode_options)

with col_scope_2:
    scope_mode = st.radio(
        "Scope",
        options=["All parts", "Single part"],
        horizontal=True,
        help="Use a single part for quick validation before processing the full episode.",
    )

selected_scenes = scenes[scenes["episode_id"] == episode_id].copy()
selected_part_id: str | None = None

if scope_mode == "Single part":
    part_options = sorted(selected_scenes["part_id"].unique().tolist())
    selected_part_id = st.selectbox("Episode part", part_options)
    selected_scenes = selected_scenes[selected_scenes["part_id"] == selected_part_id].copy()

selected_scenes = selected_scenes.sort_values(
    [column for column in ("part_order", "part_start_time") if column in selected_scenes.columns]
)

st.subheader("Sampling parameters")
param_col_1, param_col_2, param_col_3, param_col_4 = st.columns(4)

with param_col_1:
    frame_positions = st.multiselect(
        "Frame positions",
        options=list(SUPPORTED_FRAME_POSITIONS),
        default=list(SUPPORTED_FRAME_POSITIONS),
        help="Start and end samples are shifted inward by the selected edge margin.",
    )

with param_col_2:
    edge_margin_seconds = st.slider(
        "Edge margin (seconds)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Reduces transition, black-frame, and motion-blur captures near cuts.",
    )

with param_col_3:
    jpeg_quality = st.slider(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=85,
        step=5,
    )

with param_col_4:
    resize_mode = st.selectbox(
        "Maximum frame width",
        options=[None, 640, 960, 1280],
        index=0,
        format_func=lambda value: "Keep source resolution" if value is None else f"{value}px",
        help="Optional downscaling saves storage while preserving aspect ratio.",
    )

option_col_1, option_col_2 = st.columns(2)
with option_col_1:
    overwrite = st.toggle(
        "Overwrite existing frames",
        value=False,
        help="When disabled, existing files are reused and recorded in the output table.",
    )
with option_col_2:
    development_mode = st.toggle(
        "Development mode",
        value=True,
        help="Process only the first N scenes while validating the workflow.",
    )

if development_mode:
    max_scenes = st.number_input(
        "Maximum scenes",
        min_value=1,
        max_value=max(1, len(selected_scenes)),
        value=min(50, len(selected_scenes)),
        step=10,
    )
    scenes_to_process = selected_scenes.head(int(max_scenes)).copy()
else:
    scenes_to_process = selected_scenes.copy()

if not frame_positions:
    st.warning("Select at least one frame position.")
    st.stop()

estimated_frame_count = len(scenes_to_process) * len(frame_positions)
median_duration = pd.to_numeric(scenes_to_process["duration"], errors="coerce").median()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Scenes selected", f"{len(scenes_to_process):,}")
metric_2.metric("Frames to extract", f"{estimated_frame_count:,}")
metric_3.metric("Episode parts", f"{scenes_to_process['part_id'].nunique():,}")
metric_4.metric("Median scene duration", f"{median_duration:.2f}s")

scope_token = selected_part_id if selected_part_id is not None else "all_parts"
mode_token = f"dev{len(scenes_to_process)}" if development_mode else "full"
position_token = "-".join(frame_positions)
experiment_name = (
    f"{Path(selected_scene_path).stem}__{scope_token}__{position_token}__{mode_token}"
)
experiment_dir = SAMPLED_FRAMES_DIR / "images" / experiment_name
output_table_path = TABLE_DIR / f"{experiment_name}_frame_samples.parquet"

with st.expander("Execution details", expanded=False):
    st.write(f"**Input:** `{selected_scene_path}`")
    st.write(f"**Image directory:** `{experiment_dir}`")
    st.write(f"**Output table:** `{output_table_path}`")
    st.dataframe(
        scenes_to_process[
            [
                column
                for column in (
                    "episode_id",
                    "part_id",
                    "segment_id",
                    "part_start_time",
                    "part_end_time",
                    "duration",
                )
                if column in scenes_to_process.columns
            ]
        ].head(100),
        width="stretch",
    )


def _render_results(frame_samples: pd.DataFrame) -> None:
    """Render execution metrics, errors, table, and a compact image gallery."""
    if frame_samples.empty:
        st.info("No frame sample rows were generated.")
        return

    success_mask = frame_samples["success"].fillna(False).astype(bool)
    reused_mask = frame_samples.get("reused_existing", pd.Series(False, index=frame_samples.index))
    reused_mask = reused_mask.fillna(False).astype(bool)
    error_mask = ~success_mask

    total_size = pd.to_numeric(
        frame_samples.get("file_size_bytes", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).sum()

    result_col_1, result_col_2, result_col_3, result_col_4 = st.columns(4)
    result_col_1.metric("Successful frames", f"{int(success_mask.sum()):,}")
    result_col_2.metric("Failed frames", f"{int(error_mask.sum()):,}")
    result_col_3.metric("Reused files", f"{int(reused_mask.sum()):,}")
    result_col_4.metric("Storage", f"{total_size / (1024 ** 2):.1f} MB")

    if error_mask.any():
        st.warning("Some frames could not be extracted. Review the error table below.")
        st.dataframe(
            frame_samples.loc[
                error_mask,
                [
                    column
                    for column in (
                        "episode_id",
                        "part_id",
                        "segment_id",
                        "frame_position",
                        "source_video_path",
                        "error",
                    )
                    if column in frame_samples.columns
                ],
            ],
            width="stretch",
        )

    table_tab, gallery_tab = st.tabs(["Frame sample table", "Visual quality check"])

    with table_tab:
        st.dataframe(frame_samples, width="stretch")

    with gallery_tab:
        successful_samples = frame_samples.loc[
            success_mask & frame_samples["frame_path"].notna()
        ].copy()

        if successful_samples.empty:
            st.info("No successful frames are available for preview.")
            return

        segment_options = successful_samples["segment_id"].drop_duplicates().tolist()
        default_preview_count = min(12, len(segment_options))
        preview_count = st.slider(
            "Segments to preview",
            min_value=1,
            max_value=min(30, len(segment_options)),
            value=default_preview_count,
        )

        preview_segments = segment_options[:preview_count]
        preview_rows = successful_samples[
            successful_samples["segment_id"].isin(preview_segments)
        ].copy()

        for segment_id, segment_rows in preview_rows.groupby("segment_id", sort=False):
            segment_rows = segment_rows.set_index("frame_position").reindex(frame_positions)
            st.markdown(f"**{segment_id}**")
            gallery_columns = st.columns(len(frame_positions))

            for column, frame_position in zip(gallery_columns, frame_positions):
                with column:
                    if frame_position not in segment_rows.index:
                        st.caption(f"{frame_position}: unavailable")
                        continue
                    row = segment_rows.loc[frame_position]
                    frame_path = row.get("frame_path")
                    if isinstance(frame_path, str) and Path(frame_path).exists():
                        st.image(
                            frame_path,
                            caption=(
                                f"{frame_position} · "
                                f"{float(row['global_timestamp_seconds']):.2f}s"
                            ),
                            width="stretch",
                        )
                    else:
                        st.caption(f"{frame_position}: file not found")


if st.button("Extract representative frames", type="primary"):
    progress_bar = st.progress(0, text="Preparing frame extraction...")
    current_segment = st.empty()

    def update_progress(current: int, total: int, segment_id: str) -> None:
        percentage = int((current / max(total, 1)) * 100)
        progress_bar.progress(
            percentage,
            text=f"Extracting frames: {current:,}/{total:,} scenes",
        )
        current_segment.caption(f"Current scene: `{segment_id}`")

    with st.status("Extracting representative frames...", expanded=True) as status:
        try:
            frame_samples = sample_frames_from_scenes(
                scenes=scenes_to_process,
                output_dir=experiment_dir,
                frame_positions=frame_positions,
                edge_margin_seconds=edge_margin_seconds,
                jpeg_quality=jpeg_quality,
                max_width=resize_mode,
                overwrite=overwrite,
                progress_callback=update_progress,
            )
            save_table(frame_samples, output_table_path)
        except Exception as exc:
            status.update(label="Frame extraction failed.", state="error", expanded=True)
            st.exception(exc)
            st.stop()

        status.update(label="Frame extraction complete.", state="complete", expanded=False)

    progress_bar.progress(100, text="Frame extraction complete.")
    st.success(f"Saved frame metadata to `{output_table_path}`")
    _render_results(frame_samples)
else:
    existing_samples = load_table(output_table_path)
    if not existing_samples.empty:
        st.subheader("Existing extraction results")
        st.caption(f"Loaded `{output_table_path}`")
        _render_results(existing_samples)
    else:
        st.info("Configure the extraction parameters and run a development sample first.")
