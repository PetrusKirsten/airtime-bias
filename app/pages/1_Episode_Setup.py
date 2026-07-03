import pandas as pd
import streamlit as st
from airtime_bias.io.paths import PROCESSED_DIR, RAW_VIDEO_DIR, ensure_project_dirs
from airtime_bias.io.loaders import load_table
from airtime_bias.io.writers import save_table

ensure_project_dirs()
st.set_page_config(page_title="Episode Setup", page_icon="🗂️", layout="wide")
st.title("🗂️ Episode Setup")

metadata_path = PROCESSED_DIR / "episodes_metadata.parquet"
metadata = load_table(metadata_path)

st.markdown(
    """
Register only lightweight metadata here. Keep the actual videos local in `data/raw/videos/`.
"""
)

with st.form("episode_form"):
    col1, col2 = st.columns(2)
    with col1:
        episode_id = st.text_input("Episode ID", placeholder="s01e01")
        season = st.text_input("Season", placeholder="season_01")
        episode_number = st.number_input("Episode number", min_value=1, step=1)
    with col2:
        video_filename = st.text_input("Local video filename", placeholder="episode_01.mp4")
        eliminated_participant = st.text_input("Eliminated participant", placeholder="Ana")
        active_participants = st.text_area(
            "Active participants, comma-separated", placeholder="Ana, Bruno, Carla"
        )

    submitted = st.form_submit_button("Save episode metadata")

if submitted:
    row = {
        "episode_id": episode_id,
        "season": season,
        "episode_number": int(episode_number),
        "video_path": str(RAW_VIDEO_DIR / video_filename),
        "eliminated_participant": eliminated_participant,
        "active_participants": active_participants,
    }
    metadata = pd.concat([metadata, pd.DataFrame([row])], ignore_index=True)
    save_table(metadata, metadata_path)
    st.success("Episode metadata saved.")

st.subheader("Registered episodes")
st.dataframe(metadata, use_container_width=True)
st.caption(f"Place local videos in: {RAW_VIDEO_DIR}")
