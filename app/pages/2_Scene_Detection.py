from pathlib import Path
import streamlit as st
from airtime_bias.io.paths import PROCESSED_DIR, INTERIM_DIR
from airtime_bias.io.loaders import load_table
from airtime_bias.io.writers import save_table
from airtime_bias.video.scene_detection import detect_scenes

st.set_page_config(page_title="Scene Detection", page_icon="🎞️", layout="wide")
st.title("🎞️ Scene Detection")

metadata = load_table(PROCESSED_DIR / "episodes_metadata.parquet")
if metadata.empty:
    st.warning("No episodes registered yet. Go to Episode Setup first.")
    st.stop()

episode_id = st.selectbox("Select episode", metadata["episode_id"].tolist())
episode = metadata[metadata["episode_id"] == episode_id].iloc[0]
threshold = st.slider("PySceneDetect content threshold", 10.0, 60.0, 27.0, 1.0)

st.caption("Higher thresholds usually create fewer, longer segments. Lower thresholds detect more cuts.")

if st.button("Run scene detection"):
    scenes = detect_scenes(Path(episode["video_path"]), episode_id=episode_id, threshold=threshold)
    output_path = INTERIM_DIR / "scenes" / f"{episode_id}_segments.parquet"
    save_table(scenes, output_path)
    st.success(f"Detected {len(scenes)} scenes.")
    st.dataframe(scenes, use_container_width=True)
else:
    existing = load_table(INTERIM_DIR / "scenes" / f"{episode_id}_segments.parquet")
    if not existing.empty:
        st.subheader("Existing scene table")
        st.dataframe(existing, use_container_width=True)
