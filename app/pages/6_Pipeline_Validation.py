import streamlit as st
from airtime_bias.io.paths import PROCESSED_DIR
from airtime_bias.io.loaders import load_table
from airtime_bias.validation.evaluate_pipeline import detection_precision, identity_accuracy

st.set_page_config(page_title="Pipeline Validation", page_icon="✅", layout="wide")
st.title("✅ Pipeline Validation")

review = load_table(PROCESSED_DIR / "commentary_segments.parquet")
if review.empty:
    st.warning("No reviewed data available yet.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Commentary detection precision", detection_precision(review) or "N/A")
col2.metric("Participant identity accuracy", identity_accuracy(review) or "N/A")

st.dataframe(review, width='stretch')
