import pandas as pd


def detection_precision(review_df: pd.DataFrame) -> float | None:
    """Precision among automatically predicted commentary candidates."""
    if review_df.empty or "is_candidate" not in review_df or "manual_is_commentary" not in review_df:
        return None
    predicted = review_df[review_df["is_candidate"] == True]
    if predicted.empty:
        return None
    return float((predicted["manual_is_commentary"] == True).mean())


def identity_accuracy(review_df: pd.DataFrame) -> float | None:
    """Accuracy among manually labeled participant identities."""
    required = {"predicted_identity", "manual_identity"}
    if review_df.empty or not required.issubset(review_df.columns):
        return None
    labeled = review_df[review_df["manual_identity"].notna()]
    if labeled.empty:
        return None
    return float((labeled["predicted_identity"] == labeled["manual_identity"]).mean())
