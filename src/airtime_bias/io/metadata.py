from __future__ import annotations

from pathlib import Path

import pandas as pd

from airtime_bias.io.loaders import load_table
from airtime_bias.io.paths import METADATA_DIR, RAW_VIDEO_DIR, PROXY_VIDEO_DIR

EPISODES_PATH = METADATA_DIR / "episodes.csv"
EPISODE_PARTS_PATH = METADATA_DIR / "episode_parts.csv"
ACTIVE_CAST_PATH = METADATA_DIR / "active_cast.csv"

REQUIRED_EPISODE_COLUMNS = ["episode_id", "season", "episode_number", "episode_title", "eliminated_participant"]
REQUIRED_PART_COLUMNS = ["episode_id", "part_id", "part_order", "video_filename"]
REQUIRED_CAST_COLUMNS = ["episode_id", "participant"]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Metadata CSVs may be created in Excel/LibreOffice with either comma or semicolon separators.
    # sep=None lets pandas infer the delimiter.
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return load_table(path)


def load_episodes() -> pd.DataFrame:
    """Load episode-level metadata from data/interim/metadata/episodes.csv."""
    return _read_csv_if_exists(EPISODES_PATH)


def load_episode_parts() -> pd.DataFrame:
    """Load episode part metadata from data/interim/metadata/episode_parts.csv."""
    parts = _read_csv_if_exists(EPISODE_PARTS_PATH)
    if not parts.empty and "part_order" in parts.columns:
        parts = parts.sort_values(["episode_id", "part_order"]).reset_index(drop=True)
    return parts


def load_active_cast() -> pd.DataFrame:
    """Load active cast metadata from data/interim/metadata/active_cast.csv."""
    return _read_csv_if_exists(ACTIVE_CAST_PATH)


def validate_metadata_tables(
    episodes: pd.DataFrame,
    episode_parts: pd.DataFrame,
    active_cast: pd.DataFrame,
) -> list[str]:
    """Return human-readable metadata problems. Empty list means no structural issues found."""
    problems: list[str] = []
    checks = [
        ("episodes.csv", episodes, REQUIRED_EPISODE_COLUMNS),
        ("episode_parts.csv", episode_parts, REQUIRED_PART_COLUMNS),
        ("active_cast.csv", active_cast, REQUIRED_CAST_COLUMNS),
    ]
    for filename, df, required_cols in checks:
        if df.empty:
            problems.append(f"{filename} was not found or is empty.")
            continue
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            problems.append(f"{filename} is missing columns: {', '.join(missing)}")
    return problems


def add_video_paths(episode_parts: pd.DataFrame, prefer_proxy: bool = True) -> pd.DataFrame:
    """Add raw/proxy video paths and file-existence flags to an episode_parts table."""
    if episode_parts.empty:
        return episode_parts

    parts = episode_parts.copy()
    parts["raw_video_path"] = parts["video_filename"].apply(lambda name: str(RAW_VIDEO_DIR / str(name)))
    parts["raw_video_exists"] = parts["raw_video_path"].apply(lambda p: Path(p).exists())

    def proxy_name(filename: str) -> str:
        path = Path(str(filename))
        return f"{path.stem}_720p{path.suffix}"

    parts["proxy_video_filename"] = parts["video_filename"].apply(proxy_name)
    parts["proxy_video_path"] = parts["proxy_video_filename"].apply(lambda name: str(PROXY_VIDEO_DIR / str(name)))
    parts["proxy_video_exists"] = parts["proxy_video_path"].apply(lambda p: Path(p).exists())

    if prefer_proxy:
        parts["selected_video_path"] = parts.apply(
            lambda row: row["proxy_video_path"] if row["proxy_video_exists"] else row["raw_video_path"],
            axis=1,
        )
    else:
        parts["selected_video_path"] = parts["raw_video_path"]
    parts["selected_video_exists"] = parts["selected_video_path"].apply(lambda p: Path(p).exists())
    return parts


def get_episode_options() -> list[str]:
    episodes = load_episodes()
    if episodes.empty or "episode_id" not in episodes.columns:
        return []
    return sorted(episodes["episode_id"].dropna().astype(str).unique().tolist())
