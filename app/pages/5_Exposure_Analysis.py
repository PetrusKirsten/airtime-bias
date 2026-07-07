import streamlit as st
from airtime_bias.io.paths import PROCESSED_DIR
from airtime_bias.io.loaders import load_table
from airtime_bias.features.exposure_metrics import build_exposure_metrics
from airtime_bias.features.elimination_features import add_episode_zscores
from airtime_bias.visualization.plots import plot_exposure_ranking

st.set_page_config(page_title="Exposure Analysis", page_icon="📊", layout="wide")
st.title("📊 Exposure Analysis")

segments = load_table(PROCESSED_DIR / "commentary_segments.parquet")
metadata = load_table(PROCESSED_DIR / "episodes_metadata.parquet")

if segments.empty:
    st.warning("No commentary segment data found yet.")
    st.stop()

metrics = build_exposure_metrics(segments, metadata)
metrics = add_episode_zscores(metrics)

st.dataframe(metrics, width='stretch')
fig = plot_exposure_ranking(metrics)
if fig:
    st.plotly_chart(fig, width='stretch')
