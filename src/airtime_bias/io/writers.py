from pathlib import Path
import pandas as pd


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path.suffix}")
