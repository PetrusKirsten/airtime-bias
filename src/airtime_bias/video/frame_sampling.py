from pathlib import Path
import cv2
import numpy as np


def sample_frames_from_segment(
    video_path: Path,
    start_time: float,
    end_time: float,
    n_frames: int = 5,
) -> list[np.ndarray]:
    """Sample evenly spaced frames from a video segment."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times = np.linspace(start_time, end_time, num=n_frames + 2)[1:-1]
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames
