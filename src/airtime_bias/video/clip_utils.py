from pathlib import Path


def format_timestamp(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def local_video_exists(video_path: Path) -> bool:
    return video_path.exists() and video_path.is_file()
