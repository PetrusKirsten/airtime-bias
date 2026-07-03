import pandas as pd
import plotly.express as px


def plot_exposure_ranking(metrics: pd.DataFrame):
    if metrics.empty:
        return None
    return px.bar(
        metrics.sort_values("commentary_time_total", ascending=True),
        x="commentary_time_total",
        y="participant",
        color="was_eliminated" if "was_eliminated" in metrics.columns else None,
        orientation="h",
        title="Narrative airtime by participant",
        labels={
            "commentary_time_total": "Commentary/talking-head time (s)",
            "participant": "Participant",
            "was_eliminated": "Was eliminated",
        },
    )
