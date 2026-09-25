from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw" / "videos"
REFERENCES_DIR = DATA_DIR / "references" / "participants"
INTERIM_DIR = DATA_DIR / "interim"
METADATA_DIR = INTERIM_DIR / "metadata"
PROXY_VIDEO_DIR = INTERIM_DIR / "proxies"
PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_DIR = DATA_DIR / "validation"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def ensure_project_dirs() -> None:
    """Create the expected local data folders if they do not exist."""
    for path in [
        RAW_VIDEO_DIR,
        REFERENCES_DIR,
        METADATA_DIR,
        PROXY_VIDEO_DIR,
        INTERIM_DIR / "scenes",
        INTERIM_DIR / "sampled_frames",
        INTERIM_DIR / "face_features",
        INTERIM_DIR / "lower_thirds" / "frame_detections",
        INTERIM_DIR / "lower_thirds" / "events",
        INTERIM_DIR / "lower_thirds" / "images",
        INTERIM_DIR / "models",
        INTERIM_DIR / "embeddings",
        PROCESSED_DIR,
        PROCESSED_DIR / "commentary_candidates",
        PROCESSED_DIR / "commentary_candidates" / "fused",
        VALIDATION_DIR,
        VALIDATION_DIR / "commentary_reviews",
        VALIDATION_DIR / "identity_labels",
    ]:
        path.mkdir(parents=True, exist_ok=True)
