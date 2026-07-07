from __future__ import annotations

from pathlib import Path

import pandas as pd

from airtime_bias.video.clip_utils import get_video_duration_seconds, get_video_fps


def detect_scenes(
    video_path: Path,
    episode_id: str,
    threshold: float = 27.0,
    min_scene_len_seconds: float = 1.0,
    part_id: str | None = None,
    part_order: int | None = None,
    global_offset_seconds: float = 0.0,
) -> pd.DataFrame:
    """Detect video scenes with PySceneDetect.

    Parameters
    ----------
    video_path:
        Path to the local video file.
    episode_id:
        Logical episode identifier.
    threshold:
        PySceneDetect content threshold. Higher values create fewer cuts.
    min_scene_len_seconds:
        Minimum time between scene cuts, converted internally to frames.
        This does not remove short rows after detection; it constrains cut detection.
    part_id:
        Video part identifier.
    part_order:
        Ordering of the part within the episode.
    global_offset_seconds:
        Start offset of this part within the full episode timeline.

    Returns
    -------
    pd.DataFrame
        Scene table with local and global timestamps.
    """
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:
        raise ImportError("Install scenedetect with: pip install 'scenedetect[opencv]'") from exc

    video_path = Path(video_path)

    fps = get_video_fps(video_path)
    min_scene_len_frames = max(1, int(round(min_scene_len_seconds * fps)))

    detector = ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len_frames,
    )

    scene_list = detect(str(video_path), detector)

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
                "threshold": float(threshold),
                "min_scene_len_seconds": float(min_scene_len_seconds),
                "min_scene_len_frames": int(min_scene_len_frames),
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
    min_scene_len_seconds: float = 1.0,
) -> pd.DataFrame:
    """Run scene detection for all selected video parts of an episode."""
    scene_tables = []
    offset_seconds = 0.0

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
            min_scene_len_seconds=min_scene_len_seconds,
            global_offset_seconds=offset_seconds,
        )

        scene_tables.append(scenes)
        offset_seconds += duration

    if not scene_tables:
        return pd.DataFrame()

    return pd.concat(scene_tables, ignore_index=True)