"""Application state management"""
from typing import List, Optional, Tuple, Dict
from queue import Queue
from PIL import Image


class AppState:
    """Global application state"""
    def __init__(self):
        self.last_image: Optional[Image.Image] = None
        self.last_detections: List = []  # Filtered detections (for backward compatibility)
        self.unfiltered_detections: List = []  # All detections after confidence filtering (before size filtering)
        self.image_scale: float = 1.0
        self.screenshot_version: int = 0
        self.screenshot_queue: Queue = Queue()
        # Cache raw detections with confidence scores, keyed by screenshot version
        # Format: {version: List[Tuple[Tuple[int, int, int, int], float]]} where tuple is (x1, y1, x2, y2), score
        self.cached_raw_detections: Dict[int, List[Tuple[Tuple[int, int, int, int], float]]] = {}
        # Store last extraction result for UI display
        self.last_extracted_text: Optional[str] = None
        self.last_extraction_version: int = 0  # Screenshot version when last extraction was done

    def reset_detections(self):
        """Reset detections when new screenshot is captured"""
        self.last_detections = []
        self.unfiltered_detections = []
        self.image_scale = 1.0
        # Clear cached detections for old screenshot versions (keep only last 2 versions)
        if len(self.cached_raw_detections) > 2:
            oldest_version = min(self.cached_raw_detections.keys())
            del self.cached_raw_detections[oldest_version]


# Global state instance
state = AppState()

