from __future__ import annotations

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


def plot_scene_duration_histogram(scenes: pd.DataFrame):
    """Histogram of scene durations."""
    if scenes.empty or "duration" not in scenes.columns:
        return None

    return px.histogram(
        scenes,
        x="duration",
        nbins=60,
        title="Scene duration distribution",
        labels={"duration": "Scene duration (seconds)", "count": "Number of scenes"},
    )


def plot_scene_duration_by_part(scenes: pd.DataFrame):
    """Box plot of scene duration by episode part."""
    if scenes.empty or "duration" not in scenes.columns or "part_id" not in scenes.columns:
        return None

    return px.box(
        scenes,
        x="part_id",
        y="duration",
        points="outliers",
        title="Scene duration by episode part",
        labels={
            "part_id": "Episode part",
            "duration": "Scene duration (seconds)",
        },
    )


def plot_scene_count_by_part(scenes: pd.DataFrame):
    """Bar chart of number of scenes detected per episode part."""
    if scenes.empty or "part_id" not in scenes.columns:
        return None

    counts = (
        scenes.groupby("part_id", as_index=False)
        .size()
        .rename(columns={"size": "scene_count"})
    )

    return px.bar(
        counts,
        x="part_id",
        y="scene_count",
        title="Scenes detected by episode part",
        labels={
            "part_id": "Episode part",
            "scene_count": "Number of scenes",
        },
    )


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

    return px.bar(
        bucket_counts,
        x="duration_bucket",
        y="share",
        text="scene_count",
        title="Share of scenes by duration bucket",
        labels={
            "duration_bucket": "Duration bucket",
            "share": "Share of scenes",
            "scene_count": "Number of scenes",
        },
    )