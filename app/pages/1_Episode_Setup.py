import pandas as pd
import streamlit as st

from airtime_bias.io.metadata import (
    ACTIVE_CAST_PATH,
    EPISODE_PARTS_PATH,
    EPISODES_PATH,
    add_video_paths,
    load_active_cast,
    load_episode_parts,
    load_episodes,
    validate_metadata_tables,
)
from airtime_bias.io.paths import METADATA_DIR, RAW_VIDEO_DIR, PROXY_VIDEO_DIR, ensure_project_dirs
from airtime_bias.video.clip_utils import get_video_duration_seconds, seconds_to_timestamp

ensure_project_dirs()
st.set_page_config(page_title="Episode Setup", page_icon="🗂️", layout="wide")
st.title("🗂️ Episode Setup")

st.markdown(
    """
Use this page to check the local metadata files used by the app. For now, Airtime Bias reads manually created CSV files from `data/interim/metadata/`.
"""
)

st.info(
    f"Expected metadata folder: `{METADATA_DIR}`\n\n"
    f"Expected video folder: `{RAW_VIDEO_DIR}`\n\n"
    f"Optional proxy folder: `{PROXY_VIDEO_DIR}`"
)

episodes = load_episodes()
episode_parts = load_episode_parts()
active_cast = load_active_cast()
problems = validate_metadata_tables(episodes, episode_parts, active_cast)

if problems:
    st.warning("Metadata is not ready yet.")
    for problem in problems:
        st.write(f"- {problem}")
    st.markdown(
        f"""
Create these files exactly:

- `{EPISODES_PATH}`
- `{EPISODE_PARTS_PATH}`
- `{ACTIVE_CAST_PATH}`
        """
    )
else:
    st.success("Metadata files detected successfully.")

with st.expander("Expected CSV schemas", expanded=bool(problems)):
    st.code(
        """# episodes.csv
episode_id,season,episode_number,episode_title,eliminated_participant
s01e05,1,5,,ana

# episode_parts.csv
episode_id,part_id,part_order,video_filename
s01e05,s01e05_p01,1,s01e05_part01.mp4
s01e05,s01e05_p02,2,s01e05_part02.mp4
s01e05,s01e05_p03,3,s01e05_part03.mp4
s01e05,s01e05_p04,4,s01e05_part04.mp4
s01e05,s01e05_p05,5,s01e05_part05.mp4

# active_cast.csv
episode_id,participant
s01e05,ana
s01e05,bruno
s01e05,carla
""",
        language="csv",
    )

st.subheader("Episodes")
st.dataframe(episodes, width='stretch')

st.subheader("Episode parts")
parts_with_paths = add_video_paths(episode_parts, prefer_proxy=True)
st.dataframe(parts_with_paths, width='stretch')

if not parts_with_paths.empty:
    missing_videos = parts_with_paths[~parts_with_paths["selected_video_exists"]]
    if not missing_videos.empty:
        st.error("Some video files were not found. Check filenames and folders.")
        st.dataframe(
            missing_videos[["episode_id", "part_id", "video_filename", "selected_video_path"]],
            width='stretch',
        )

st.subheader("Active cast")
st.dataframe(active_cast, width='stretch')

st.subheader("Episode timeline check")
if parts_with_paths.empty or "selected_video_exists" not in parts_with_paths.columns:
    st.caption("Add episode_parts.csv to enable timeline checking.")
elif not parts_with_paths["selected_video_exists"].all():
    st.caption("Fix missing video paths before computing durations.")
else:
    episode_options = sorted(parts_with_paths["episode_id"].astype(str).unique().tolist())
    selected_episode = st.selectbox("Select episode for duration/offset check", episode_options)
    selected_parts = parts_with_paths[parts_with_paths["episode_id"].astype(str) == selected_episode].copy()
    selected_parts = selected_parts.sort_values("part_order")

    if st.button("Compute durations and global offsets"):
        rows = []
        offset = 0.0
        for _, row in selected_parts.iterrows():
            duration = get_video_duration_seconds(row["selected_video_path"])
            rows.append(
                {
                    "episode_id": row["episode_id"],
                    "part_id": row["part_id"],
                    "part_order": row["part_order"],
                    "selected_video_path": row["selected_video_path"],
                    "duration_seconds": duration,
                    "duration_timestamp": seconds_to_timestamp(duration),
                    "global_offset_seconds": offset,
                    "global_offset_timestamp": seconds_to_timestamp(offset),
                }
            )
            offset += duration
        timeline = pd.DataFrame(rows)
        st.dataframe(timeline, width='stretch')
        st.success(f"Total episode duration: {seconds_to_timestamp(offset)}")
