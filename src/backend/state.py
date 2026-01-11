"""Application state management"""
from typing import List, Optional
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

    def reset_detections(self):
        """Reset detections when new screenshot is captured"""
        self.last_detections = []
        self.unfiltered_detections = []
        self.image_scale = 1.0


# Global state instance
state = AppState()

