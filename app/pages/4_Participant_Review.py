import streamlit as st
from airtime_bias.io.paths import PROCESSED_DIR
from airtime_bias.io.loaders import load_table
from airtime_bias.io.writers import save_table

st.set_page_config(page_title="Participant Review", page_icon="🧑", layout="wide")
st.title("🧑 Participant Review")

path = PROCESSED_DIR / "commentary_segments.parquet"
df = load_table(path)

if df.empty:
    st.warning("No commentary candidate table found yet.")
    st.stop()

st.markdown(
    """
Review low-confidence predictions, correct participant identities and mark false positives. This human-in-the-loop layer is part of the method, not a shortcut.
"""
)

edited = st.data_editor(df, width='stretch', num_rows="dynamic")
if st.button("Save reviewed table"):
    save_table(edited, path)
    st.success("Reviewed table saved.")
