from __future__ import annotations

import pandas as pd
import plotly.express as px

from airtime_bias.visualization.plots import (
    ACCENT_GOLD,
    ACCENT_RED,
    PRIMARY_BLUE,
    _apply_professional_style,
)


def plot_lower_third_score_distribution(frame_results: pd.DataFrame):
    if frame_results.empty or "lower_third_visual_score" not in frame_results.columns:
        return None
    figure = px.histogram(
        frame_results,
        x="lower_third_visual_score",
        color="lower_third_detected",
        nbins=40,
        labels={
            "lower_third_visual_score": "Lower-third visual score",
            "lower_third_detected": "Detected",
            "count": "Sampled frames",
        },
        color_discrete_map={True: ACCENT_GOLD, False: PRIMARY_BLUE},
    )
    return _apply_professional_style(figure, "Lower-third visual score distribution")


def plot_lower_third_timeline(frame_results: pd.DataFrame):
    if frame_results.empty:
        return None
    x_column = (
        "global_timestamp_seconds"
        if "global_timestamp_seconds" in frame_results.columns
        else "part_timestamp_seconds"
    )
    figure = px.scatter(
        frame_results,
        x=x_column,
        y="lower_third_visual_score",
        color="lower_third_detected",
        hover_data=[column for column in ("part_id", "roi_crop_path") if column in frame_results.columns],
        labels={
            x_column: "Timeline (seconds)",
            "lower_third_visual_score": "Visual score",
            "lower_third_detected": "Detected",
        },
        color_discrete_map={True: ACCENT_GOLD, False: PRIMARY_BLUE},
    )
    figure.update_traces(marker={"size": 7, "opacity": 0.80})
    return _apply_professional_style(figure, "Lower-third detections over time")


def plot_lower_third_event_durations(events: pd.DataFrame):
    if events.empty or "duration" not in events.columns:
        return None
    figure = px.histogram(
        events,
        x="duration",
        nbins=30,
        labels={"duration": "Event duration (seconds)", "count": "Events"},
    )
    figure.update_traces(marker_color=ACCENT_RED, opacity=0.90)
    return _apply_professional_style(figure, "Lower-third event duration distribution")
