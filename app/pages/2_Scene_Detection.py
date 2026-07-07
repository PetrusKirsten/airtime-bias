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


ensure_project_dirs()
st.set_page_config(page_title="Scene Detection", page_icon="🎞️", layout="wide")
st.title("🎞️ Scene Detection")

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
        width='stretch',
    )
    st.stop()

episode_ids = sorted(episodes["episode_id"].astype(str).unique().tolist())
episode_id = st.selectbox("Select episode", episode_ids)
threshold = st.slider("PySceneDetect content threshold", 10.0, 60.0, 27.0, 1.0)

st.caption("Higher thresholds usually create fewer, longer segments. Lower thresholds detect more cuts.")

selected_parts = parts_with_paths[parts_with_paths["episode_id"].astype(str) == episode_id].sort_values("part_order")
st.subheader("Parts to process")
st.dataframe(selected_parts[["part_id", "part_order", "selected_video_path"]], width='stretch')

output_path = INTERIM_DIR / "scenes" / f"{episode_id}_segments.parquet"

if st.button("Run scene detection for all parts"):
    scene_tables = []
    offset_seconds = 0.0
    total_parts = len(selected_parts)

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

            st.write(f"Processing part {index}/{total_parts}: `{part_id}`")
            progress_bar.progress(
                int(((index - 1) / total_parts) * 100),
                text=f"Processing part {index}/{total_parts}: {part_id}",
            )

            status_rows.loc[status_rows["part_id"].astype(str) == part_id, "status"] = "running"
            status_table_placeholder.dataframe(status_rows, use_container_width=True)

            try:
                part_duration = get_video_duration_seconds(video_path)

                part_scenes = detect_scenes(
                    video_path=video_path,
                    episode_id=episode_id,
                    threshold=threshold,
                    part_id=part_id,
                    part_order=part_order,
                    global_offset_seconds=offset_seconds,
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

                offset_seconds += part_duration

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

        save_table(scenes, output_path)

        status.update(
            label="Scene detection complete.",
            state="complete",
            expanded=False,
        )

    progress_bar.progress(100, text="Scene detection complete.")

    st.success(
        f"Detected {len(scenes)} scenes across {total_parts} video parts."
    )
    st.dataframe(scenes, use_container_width=True)

else:
    existing = load_table(output_path)
    if not existing.empty:
        st.subheader("Existing scene table")
        st.dataframe(existing, use_container_width=True)
    
    else:
        st.info("No scene table found yet for this episode.")
