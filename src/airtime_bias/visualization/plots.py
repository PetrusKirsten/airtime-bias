from __future__ import annotations

import pandas as pd
import plotly.express as px


# ============================================================
# Airtime Bias - Plot styling system
# ============================================================

PRIMARY_BLUE = "#5FACF0"
PRIMARY_BLUE_DARK = "#2F5D8A"
PRIMARY_BLUE_LIGHT = "#8FBCE6"
ACCENT_RED = "#E45756"
ACCENT_GOLD = "#F3A712"

GRID_COLOR = "rgba(255, 255, 255, 0.10)"
AXIS_LINE_COLOR = "rgba(255, 255, 255, 0.22)"
BAR_BORDER = "rgba(255, 255, 255, 0.70)"

PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(0,0,0,0)"

FONT_COLOR = "#EDEDED"
TITLE_COLOR = "#FFFFFF"
SUBTLE_TEXT = "#BFBFBF"


# ============================================================
# Shared style helpers
# ============================================================

def _apply_professional_style(fig, title: str | None = None):
    """Apply a consistent professional dark-theme style to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "color": TITLE_COLOR},
        } if title else None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"size": 14, "color": FONT_COLOR},
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        bargap=0.08,
        hoverlabel=dict(
            bgcolor="rgba(20,20,20,0.95)",
            bordercolor="rgba(255,255,255,0.15)",
            font_size=13,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=12),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE_COLOR,
        ticks="outside",
        tickfont=dict(size=12),
        title_font=dict(size=15),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE_COLOR,
        ticks="outside",
        tickfont=dict(size=12),
        title_font=dict(size=15),
    )

    return fig


def _add_histogram_reference_lines(fig, scenes: pd.DataFrame):
    """Add mean and median vertical lines + annotations to histogram."""
    if scenes.empty or "duration" not in scenes.columns:
        return fig

    durations = pd.to_numeric(scenes["duration"], errors="coerce").dropna()
    if durations.empty:
        return fig

    mean_val = float(durations.mean())
    median_val = float(durations.median())
    share_under_2 = float((durations < 2).mean())
    n_scenes = int(durations.shape[0])

    # Mean line
    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dash",
        line_color=ACCENT_GOLD,
        opacity=0.95,
    )

    # Median line
    fig.add_vline(
        x=median_val,
        line_width=2,
        line_dash="dot",
        line_color=ACCENT_RED,
        opacity=0.95,
    )

    # Summary annotation
    annotation_text = (
        f"<b>N scenes:</b> {n_scenes}"
        f"<br><b>Mean:</b> {mean_val:.2f}s"
        f"<br><b>Median:</b> {median_val:.2f}s"
        f"<br><b>Under 2s:</b> {share_under_2:.1%}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.985,
        y=0.98,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        text=annotation_text,
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        bgcolor="rgba(20,20,20,0.82)",
        font=dict(size=12, color=FONT_COLOR),
    )

    # Mean label
    fig.add_annotation(
        x=mean_val,
        yref="paper",
        y=1.02,
        text=f"Mean = {mean_val:.2f}s",
        showarrow=False,
        font=dict(size=12, color=ACCENT_GOLD),
        bgcolor="rgba(0,0,0,0.0)",
    )

    # Median label
    fig.add_annotation(
        x=median_val,
        yref="paper",
        y=0.93,
        text=f"Median = {median_val:.2f}s",
        showarrow=False,
        font=dict(size=12, color=ACCENT_RED),
        bgcolor="rgba(0,0,0,0.0)",
    )

    return fig


# ============================================================
# Summary / transformations
# ============================================================

def summarize_scene_durations(scenes: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for scene durations."""
    if scenes.empty or "duration" not in scenes.columns:
        return pd.DataFrame()

    durations = pd.to_numeric(scenes["duration"], errors="coerce").dropna()
    if durations.empty:
        return pd.DataFrame()

    summary = {
        "n_scenes": int(durations.shape[0]),
        "total_duration_seconds": float(durations.sum()),
        "mean_duration": float(durations.mean()),
        "median_duration": float(durations.median()),
        "std_duration": float(durations.std(ddof=1)) if durations.shape[0] > 1 else 0.0,
        "min_duration": float(durations.min()),
        "p10_duration": float(durations.quantile(0.10)),
        "p25_duration": float(durations.quantile(0.25)),
        "p75_duration": float(durations.quantile(0.75)),
        "p90_duration": float(durations.quantile(0.90)),
        "max_duration": float(durations.max()),
        "share_under_1s": float((durations < 1).mean()),
        "share_under_2s": float((durations < 2).mean()),
        "share_1_to_3s": float(((durations >= 1) & (durations < 3)).mean()),
        "share_3_to_5s": float(((durations >= 3) & (durations < 5)).mean()),
        "share_over_5s": float((durations >= 5).mean()),
    }

    return pd.DataFrame([summary])


def add_duration_buckets(scenes: pd.DataFrame) -> pd.DataFrame:
    """Add categorical duration buckets to a scene table."""
    if scenes.empty or "duration" not in scenes.columns:
        return scenes.copy()

    df = scenes.copy()
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    bins = [0, 1, 3, 5, 10, 20, float("inf")]
    labels = ["<1s", "1–3s", "3–5s", "5–10s", "10–20s", "20s+"]

    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return df


# ============================================================
# Core charts
# ============================================================

def plot_scene_duration_histogram(scenes: pd.DataFrame):
    """Histogram of scene durations with reference lines and annotations."""
    if scenes.empty or "duration" not in scenes.columns:
        return None

    fig = px.histogram(
        scenes,
        x="duration",
        nbins=40,
        labels={
            "duration": "Scene duration (seconds)",
            "count": "Number of scenes",
        },
    )

    fig.update_traces(
        marker_color=PRIMARY_BLUE,
        marker_line_color=BAR_BORDER,
        marker_line_width=1.0,
        opacity=0.94,
        hovertemplate="Duration bin: %{x:.2f}s<br>Scenes: %{y}<extra></extra>",
    )

    fig = _apply_professional_style(fig, "Scene duration distribution")
    fig = _add_histogram_reference_lines(fig, scenes)

    return fig


def plot_scene_duration_by_part(scenes: pd.DataFrame):
    """Box plot of scene duration by episode part."""
    if scenes.empty or "duration" not in scenes.columns or "part_id" not in scenes.columns:
        return None

    fig = px.box(
        scenes,
        x="part_id",
        y="duration",
        points="outliers",
        labels={
            "part_id": "Episode part",
            "duration": "Scene duration (seconds)",
        },
    )

    fig.update_traces(
        marker_color=PRIMARY_BLUE,
        line_color=PRIMARY_BLUE_DARK,
        fillcolor="rgba(79, 140, 201, 0.42)",
        opacity=0.96,
        hovertemplate="Part: %{x}<br>Duration: %{y:.2f}s<extra></extra>",
    )

    return _apply_professional_style(fig, "Scene duration by episode part")


def plot_scene_count_by_part(scenes: pd.DataFrame):
    """Bar chart of number of scenes detected per episode part."""
    if scenes.empty or "part_id" not in scenes.columns:
        return None

    counts = (
        scenes.groupby("part_id", as_index=False)
        .size()
        .rename(columns={"size": "scene_count"})
    )

    fig = px.bar(
        counts,
        x="part_id",
        y="scene_count",
        text="scene_count",
        labels={
            "part_id": "Episode part",
            "scene_count": "Number of scenes",
        },
    )

    fig.update_traces(
        marker_color=PRIMARY_BLUE,
        marker_line_color=BAR_BORDER,
        marker_line_width=1.0,
        opacity=0.95,
        textposition="outside",
        hovertemplate="Part: %{x}<br>Scenes: %{y}<extra></extra>",
    )

    return _apply_professional_style(fig, "Scenes detected by episode part")


def plot_duration_bucket_share(scenes: pd.DataFrame):
    """Bar chart of duration bucket shares."""
    if scenes.empty or "duration" not in scenes.columns:
        return None

    df = add_duration_buckets(scenes)

    bucket_counts = (
        df.groupby("duration_bucket", observed=False)
        .size()
        .reset_index(name="scene_count")
    )

    bucket_counts["share"] = bucket_counts["scene_count"] / bucket_counts["scene_count"].sum()

    fig = px.bar(
        bucket_counts,
        x="duration_bucket",
        y="share",
        text="scene_count",
        labels={
            "duration_bucket": "Duration bucket",
            "share": "Share of scenes",
            "scene_count": "Number of scenes",
        },
    )

    fig.update_traces(
        marker_color=PRIMARY_BLUE,
        marker_line_color=BAR_BORDER,
        marker_line_width=1.0,
        opacity=0.95,
        textposition="outside",
        hovertemplate=(
            "Bucket: %{x}<br>"
            "Share: %{y:.1%}<br>"
            "Scenes: %{text}<extra></extra>"
        ),
    )

    fig.update_yaxes(tickformat=".0%")

    return _apply_professional_style(fig, "Share of scenes by duration bucket")


def plot_exposure_ranking(metrics: pd.DataFrame):
    """Horizontal ranking chart for participant narrative airtime."""
    if metrics.empty:
        return None

    color_col = "was_eliminated" if "was_eliminated" in metrics.columns else None

    fig = px.bar(
        metrics.sort_values("commentary_time_total", ascending=True),
        x="commentary_time_total",
        y="participant",
        color=color_col,
        orientation="h",
        labels={
            "commentary_time_total": "Commentary / talking-head time (s)",
            "participant": "Participant",
            "was_eliminated": "Eliminated",
        },
        color_discrete_map={True: ACCENT_RED, False: PRIMARY_BLUE} if color_col else None,
    )

    fig.update_traces(
        marker_line_color=BAR_BORDER,
        marker_line_width=1.2,
        opacity=0.95,
        hovertemplate=(
            "Participant: %{y}<br>"
            "Airtime: %{x:.2f}s<extra></extra>"
        ),
    )

    return _apply_professional_style(fig, "Narrative airtime by participant")
    