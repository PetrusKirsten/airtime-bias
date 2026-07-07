from __future__ import annotations

from pathlib import Path

import pandas as pd


def detect_scenes(
    video_path: Path,
    episode_id: str,
    threshold: float = 27.0,
    part_id: str | None = None,
    part_order: int | None = None,
    global_offset_seconds: float = 0.0,
) -> pd.DataFrame:
    """Detect video scenes with PySceneDetect.

    Returns a table with local part times and global episode times.
    """
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:
        raise ImportError("Install scenedetect with: pip install 'scenedetect[opencv]'") from exc

    video_path = Path(video_path)
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))

    rows = []
    part_token = part_id or episode_id
    for i, (start, end) in enumerate(scene_list, start=1):
        part_start = float(start.get_seconds())
        part_end = float(end.get_seconds())
        global_start = global_offset_seconds + part_start
        global_end = global_offset_seconds + part_end
        rows.append(
            {
                "episode_id": episode_id,
                "part_id": part_id,
                "part_order": part_order,
                "segment_id": f"{part_token}_seg_{i:05d}",
                "video_path": str(video_path),
                "part_start_time": part_start,
                "part_end_time": part_end,
                "global_start_time": global_start,
                "global_end_time": global_end,
                "duration": part_end - part_start,
            }
        )
    return pd.DataFrame(rows)


def detect_episode_parts_scenes(
    episode_parts: pd.DataFrame,
    episode_id: str,
    threshold: float = 27.0,
) -> pd.DataFrame:
    """Run scene detection for all selected video parts of an episode."""
    rows = []
    offset = 0.0

    from airtime_bias.video.clip_utils import get_video_duration_seconds

    parts = episode_parts[episode_parts["episode_id"].astype(str) == str(episode_id)].copy()
    parts = parts.sort_values("part_order")

    for _, part in parts.iterrows():
        video_path = Path(part["selected_video_path"])
        duration = get_video_duration_seconds(video_path)
        scenes = detect_scenes(
            video_path=video_path,
            episode_id=episode_id,
            part_id=str(part["part_id"]),
            part_order=int(part["part_order"]),
            threshold=threshold,
            global_offset_seconds=offset,
        )
        rows.append(scenes)
        offset += duration

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
