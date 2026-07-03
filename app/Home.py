import streamlit as st
from airtime_bias.io.paths import ensure_project_dirs

ensure_project_dirs()

st.set_page_config(
    page_title="Airtime Bias",
    page_icon="📺",
    layout="wide",
)

st.title("📺 Airtime Bias")
st.subheader("Computer vision for narrative airtime analysis in reality TV")

st.markdown(
    """
Airtime Bias is an interactive, human-in-the-loop application for investigating how reality TV distributes narrative exposure among contestants.

### Research question

> Do eliminated contestants receive a different pattern of narrative airtime in the episode they leave?

### Initial technical focus

The MVP focuses on **participant commentary/talking-head segments**: individual, direct-to-camera or interview-style scenes that usually have a stable frame, one dominant face and high narrative value.

### Pipeline

1. Episode setup  
2. Scene detection  
3. Commentary candidate detection  
4. Participant identity review  
5. Exposure analysis  
6. Pipeline validation  
7. Story mode  

The app is designed to keep videos, frames, reference images and embeddings local.
"""
)

st.info("Start with **Episode Setup** in the sidebar.")
