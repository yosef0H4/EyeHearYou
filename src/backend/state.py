"""Application state management"""
from typing import List, Optional, Tuple, Dict
from queue import Queue
from PIL import Image
import hashlib
import json


class AppState:
    """Global application state"""
    def __init__(self):
        self.last_image: Optional[Image.Image] = None
        self.last_detections: List = []  # Filtered detections (for backward compatibility)
        self.unfiltered_detections: List = []  # All detections after confidence filtering (before size filtering)
        self.image_scale: float = 1.0
        self.screenshot_version: int = 0
        self.screenshot_queue: Queue = Queue()
        # Cache raw detections with confidence scores, keyed by (screenshot_version, preprocessing_hash)
        # Format: {(version, preproc_hash): List[Tuple[Tuple[int, int, int, int], float]]} 
        # where tuple is (x1, y1, x2, y2), score
        self.cached_raw_detections: Dict[Tuple[int, str], List[Tuple[Tuple[int, int, int, int], float]]] = {}
        # Store last extraction result for UI display
        self.last_extracted_text: Optional[str] = None
        self.last_extraction_version: int = 0  # Screenshot version when last extraction was done

        # Selection & Manual Mode State
        self.use_rapidocr: bool = True
        self.selection_base_state: bool = True  # True = Select All (White), False = Deselect All (Black)
        self.selection_ops: List[Dict] = []  # List of {"op": "add"|"sub", "rect": (x,y,w,h)} normalized 0-1
        self.manual_boxes: List[Tuple[int, int, int, int]] = []  # Manual boxes in pixel coordinates

    def reset_detections(self):
        """Reset detections when new screenshot is captured"""
        self.last_detections = []
        self.unfiltered_detections = []
        self.image_scale = 1.0
        # Clear manual boxes when new screenshot is captured (user usually wants fresh start)
        self.manual_boxes = []
        # Clear cached detections for old screenshot versions (keep only last 5 cache entries)
        if len(self.cached_raw_detections) > 5:
            # Remove oldest entries (by version)
            versions = set(version for version, _ in self.cached_raw_detections.keys())
            if len(versions) > 2:
                oldest_version = min(versions)
                keys_to_remove = [k for k in self.cached_raw_detections.keys() if k[0] == oldest_version]
                for k in keys_to_remove:
                    del self.cached_raw_detections[k]


# Global state instance
state = AppState()

