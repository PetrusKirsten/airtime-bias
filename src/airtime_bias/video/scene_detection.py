from pathlib import Path
import pandas as pd


def detect_scenes(video_path: Path, episode_id: str, threshold: float = 27.0) -> pd.DataFrame:
    """Detect video scenes with PySceneDetect.

    Returns a table with episode_id, segment_id, start_time, end_time and duration.
    """
    try:
        from scenedetect import detect, ContentDetector
    except ImportError as exc:
        raise ImportError("Install scenedetect with: pip install 'scenedetect[opencv]'") from exc

    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    rows = []
    for i, (start, end) in enumerate(scene_list, start=1):
        start_seconds = start.get_seconds()
        end_seconds = end.get_seconds()
        rows.append(
            {
                "episode_id": episode_id,
                "segment_id": f"{episode_id}_seg_{i:05d}",
                "start_time": start_seconds,
                "end_time": end_seconds,
                "duration": end_seconds - start_seconds,
            }
        )
    return pd.DataFrame(rows)
