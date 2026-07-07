from __future__ import annotations

from pathlib import Path


def get_video_metadata(video_path: str | Path) -> dict:
    """Return basic video metadata using OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read video metadata.") from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if not fps or fps <= 0:
        raise ValueError(f"Could not determine FPS for video file: {video_path}")

    duration_seconds = float(frame_count / fps)

    return {
        "video_path": str(video_path),
        "frame_count": float(frame_count),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "duration_seconds": duration_seconds,
    }


def get_video_duration_seconds(video_path: str | Path) -> float:
    """Return video duration in seconds."""
    return float(get_video_metadata(video_path)["duration_seconds"])


def get_video_fps(video_path: str | Path) -> float:
    """Return video FPS."""
    return float(get_video_metadata(video_path)["fps"])


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"