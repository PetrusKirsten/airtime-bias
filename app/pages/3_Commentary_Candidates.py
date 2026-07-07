import streamlit as st
from airtime_bias.io.paths import INTERIM_DIR
from airtime_bias.io.loaders import load_table

st.set_page_config(page_title="Commentary Candidates", page_icon="🎙️", layout="wide")
st.title("🎙️ Commentary Candidates")

st.markdown(
    """
This page will score candidate **participant commentary/talking-head segments** using visual cues such as face count, face size, centering, duration and frame stability.
"""
)

scene_files = sorted((INTERIM_DIR / "scenes").glob("*_segments.parquet"))
if not scene_files:
    st.warning("No scene files found yet. Run Scene Detection first.")
    st.stop()

selected = st.selectbox("Scene table", scene_files, format_func=lambda p: p.name)
scenes = load_table(selected)
st.dataframe(scenes, width='stretch')

st.info("Next implementation: sample frames, detect faces and compute commentary_score.")
