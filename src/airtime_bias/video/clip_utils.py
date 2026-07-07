from __future__ import annotations

from pathlib import Path


def get_video_duration_seconds(video_path: str | Path) -> float:
    """Return video duration in seconds using OpenCV metadata."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read video duration.") from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not fps or fps <= 0:
        raise ValueError(f"Could not determine FPS for video file: {video_path}")
    return float(frame_count / fps)


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
