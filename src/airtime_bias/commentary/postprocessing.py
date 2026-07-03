import pandas as pd


def remove_short_segments(df: pd.DataFrame, min_duration: float = 1.0) -> pd.DataFrame:
    """Remove candidate segments below a minimum duration threshold."""
    if df.empty or "duration" not in df.columns:
        return df
    return df[df["duration"] >= min_duration].reset_index(drop=True)
