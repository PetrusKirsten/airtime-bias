import pandas as pd


def build_exposure_metrics(segments: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate narrative exposure metrics by episode and participant."""
    if segments.empty:
        return pd.DataFrame()

    df = segments.copy()
    if "final_identity" not in df.columns:
        df["final_identity"] = df.get("predicted_identity", "unknown")

    metrics = (
        df.groupby(["episode_id", "final_identity"], dropna=False)
        .agg(
            commentary_time_total=("duration", "sum"),
            commentary_segments_count=("segment_id", "count"),
            avg_commentary_segment_duration=("duration", "mean"),
            median_commentary_segment_duration=("duration", "median"),
            confidence_mean=(
                "identity_confidence",
                "mean",
            )
            if "identity_confidence" in df.columns
            else ("duration", "size"),
        )
        .reset_index()
        .rename(columns={"final_identity": "participant"})
    )

    totals = metrics.groupby("episode_id")["commentary_time_total"].transform("sum")
    metrics["commentary_time_share"] = metrics["commentary_time_total"] / totals

    if metadata is not None and not metadata.empty and "eliminated_participant" in metadata.columns:
        metrics = metrics.merge(
            metadata[["episode_id", "eliminated_participant"]], on="episode_id", how="left"
        )
        metrics["was_eliminated"] = metrics["participant"] == metrics["eliminated_participant"]

    return metrics
