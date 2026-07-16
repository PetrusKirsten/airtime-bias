"""Lower-third detection, temporal grouping, OCR, and candidate integration."""

from .visual_detection import LowerThirdROI, LowerThirdVisualConfig, analyze_lower_third_frame
from .temporal_events import group_lower_third_events, scan_lower_thirds_in_video

__all__ = [
    "LowerThirdROI",
    "LowerThirdVisualConfig",
    "analyze_lower_third_frame",
    "group_lower_third_events",
    "scan_lower_thirds_in_video",
]
