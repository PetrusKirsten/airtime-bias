import pandas as pd


def add_episode_zscores(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add within-episode z-scores for narrative airtime."""
    if metrics.empty:
        return metrics
    df = metrics.copy()
    grouped = df.groupby("episode_id")["commentary_time_total"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, pd.NA)
    df["zscore_vs_cast_in_episode"] = ((df["commentary_time_total"] - mean) / std).fillna(0)
    return df
