from __future__ import annotations

import pandas as pd
import plotly.express as px

from airtime_bias.visualization.plots import (
    ACCENT_GOLD,
    ACCENT_RED,
    BAR_BORDER,
    PRIMARY_BLUE,
    _apply_professional_style,
)

TIER_COLORS = {
    "candidate": PRIMARY_BLUE,
    "review": ACCENT_GOLD,
    "reject": ACCENT_RED,
}


def plot_commentary_score_distribution(candidates: pd.DataFrame):
    """Histogram of commentary candidate scores by tier."""
    if candidates.empty or "commentary_score" not in candidates.columns:
        return None

    fig = px.histogram(
        candidates,
        x="commentary_score",
        color="candidate_tier",
        nbins=30,
        barmode="overlay",
        color_discrete_map=TIER_COLORS,
        labels={
            "commentary_score": "Commentary score",
            "count": "Number of scenes",
            "candidate_tier": "Decision tier",
        },
    )
    fig.update_traces(
        marker_line_color=BAR_BORDER,
        marker_line_width=0.8,
        opacity=0.82,
    )
    fig.update_xaxes(range=[0, 1])
    return _apply_professional_style(fig, "Commentary score distribution")


def plot_candidate_tier_counts(candidates: pd.DataFrame):
    """Bar chart showing how many scenes remain in each decision tier."""
    if candidates.empty or "candidate_tier" not in candidates.columns:
        return None

    order = ["candidate", "review", "reject"]
    counts = (
        candidates["candidate_tier"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("candidate_tier")
        .reset_index(name="scene_count")
    )

    fig = px.bar(
        counts,
        x="candidate_tier",
        y="scene_count",
        color="candidate_tier",
        text="scene_count",
        category_orders={"candidate_tier": order},
        color_discrete_map=TIER_COLORS,
        labels={
            "candidate_tier": "Decision tier",
            "scene_count": "Number of scenes",
        },
    )
    fig.update_traces(
        marker_line_color=BAR_BORDER,
        marker_line_width=1.0,
        textposition="outside",
        showlegend=False,
    )
    return _apply_professional_style(fig, "Candidate reduction by decision tier")


def plot_face_geometry_map(candidates: pd.DataFrame):
    """Scatter plot of face area versus distance from frame center."""
    required = {
        "median_face_area_ratio",
        "median_face_center_distance",
        "candidate_tier",
    }
    if candidates.empty or not required.issubset(candidates.columns):
        return None

    data = candidates.dropna(
        subset=["median_face_area_ratio", "median_face_center_distance"]
    ).copy()
    if data.empty:
        return None

    fig = px.scatter(
        data,
        x="median_face_center_distance",
        y="median_face_area_ratio",
        color="candidate_tier",
        size="commentary_score",
        hover_data=[
            "segment_id",
            "segment_duration",
            "face_presence_ratio",
            "single_face_ratio",
            "commentary_score",
        ],
        color_discrete_map=TIER_COLORS,
        labels={
            "median_face_center_distance": "Median face distance from center",
            "median_face_area_ratio": "Median face area ratio",
            "candidate_tier": "Decision tier",
            "commentary_score": "Commentary score",
        },
    )
    fig.update_traces(marker_line_color=BAR_BORDER, marker_line_width=0.6, opacity=0.82)
    return _apply_professional_style(fig, "Face geometry of detected scenes")
