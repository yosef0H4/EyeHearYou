"""Text detection using RapidOCR"""
import numpy as np
import cv2
from PIL import Image

# Optional imports for text detection (RapidOCR)
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    RapidOCR = None

from .image_utils import check_gpu_available
from .filtering import filter_text_regions, sort_text_regions_by_reading_order
from .task_manager import task_manager

# Global RapidOCR instance (lazy initialization)
_rapidocr_instance = None


def _get_rapidocr_instance():
    """Get or create RapidOCR instance (singleton pattern)"""
    global _rapidocr_instance
    if _rapidocr_instance is None and RAPIDOCR_AVAILABLE:
        _rapidocr_instance = RapidOCR()
    return _rapidocr_instance


def detect_text_regions(image, min_width=30, min_height=30):
    """
    Detect text regions in the image using RapidOCR (detection only).
    Returns a list of bounding boxes as (x1, y1, x2, y2) tuples.
    
    Note: RapidOCR in detection-only mode does not provide confidence scores.
    Confidence scores are only available when recognition is enabled.
    
    Args:
        image: PIL Image to process
        min_width: Minimum width of text region in pixels. Default: 30
        min_height: Minimum height of text region in pixels. Default: 30
    
    Returns:
        List of (x1, y1, x2, y2) tuples representing bounding boxes
    """
    if task_manager.is_cancelled():
        return None
    
    if not RAPIDOCR_AVAILABLE:
        return None
    
    try:
        print("Using RapidOCR for text detection (CPU via ONNX Runtime)...")
        
        # Get RapidOCR instance
        ocr = _get_rapidocr_instance()
        if ocr is None:
            return None
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        
        # Run detection only (use_det=True, use_rec=False)
        # Result format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
        result, _ = ocr(img_array, use_det=True, use_rec=False)
        
        # Check for cancellation after detection
        if task_manager.is_cancelled():
            return None
        
        # Extract bounding boxes from RapidOCR output
        text_regions = []
        
        if result and len(result) > 0:
            for poly in result:
                if isinstance(poly, list) and len(poly) >= 4:
                    # Extract coordinates from polygon
                    # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    x_coords = [float(point[0]) for point in poly]
                    y_coords = [float(point[1]) for point in poly]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                    
                    # Only add if region is valid
                    if x2 > x1 and y2 > y1:
                        text_regions.append((x1, y1, x2, y2))
        
        # Filter out small regions (UI elements, icons, etc.)
        text_regions = filter_text_regions(text_regions, (img_height, img_width), 
                                          min_width=min_width, min_height=min_height)
        
        # Sort regions by reading order (top-to-bottom, left-to-right for English)
        text_regions = sort_text_regions_by_reading_order(text_regions, direction='hor_ltr')
        
        return text_regions
    except Exception as e:
        print(f"Error detecting text regions: {e}")
        print("Falling back to full image processing...")
        import traceback
        traceback.print_exc()
        return None


def detect_text_regions_unfiltered(image):
    """
    Detect text regions in the image using RapidOCR, but return ALL detections
    (before size filtering).
    This allows the backend to cache raw detections and apply filters later.
    
    Note: RapidOCR in detection-only mode does not provide confidence scores.
    All detections are returned with a default score of 1.0 for compatibility.
    
    Args:
        image: PIL Image to process
    
    Returns:
        List of tuples: ((x1, y1, x2, y2), confidence_score) for all valid detections.
        Note: confidence_score is always 1.0 since RapidOCR doesn't provide scores in detection-only mode.
    """
    if task_manager.is_cancelled():
        return None
    
    if not RAPIDOCR_AVAILABLE:
        return None
    
    try:
        print("Using RapidOCR for text detection (CPU via ONNX Runtime)...")
        
        # Get RapidOCR instance
        ocr = _get_rapidocr_instance()
        if ocr is None:
            return None
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        
        # Run detection only
        result, _ = ocr(img_array, use_det=True, use_rec=False)
        
        # Check for cancellation after detection
        if task_manager.is_cancelled():
            return None
        
        # Extract bounding boxes from RapidOCR output
        text_regions = []
        
        if result and len(result) > 0:
            for poly in result:
                if isinstance(poly, list) and len(poly) >= 4:
                    # Extract coordinates from polygon
                    x_coords = [float(point[0]) for point in poly]
                    y_coords = [float(point[1]) for point in poly]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                    
                    if x2 > x1 and y2 > y1:
                        text_regions.append((x1, y1, x2, y2))
        
        # Return all detections with default confidence scores (1.0) for compatibility
        # Note: RapidOCR doesn't provide confidence scores in detection-only mode
        confidence_scores = [1.0] * len(text_regions)
        detections_with_scores = list(zip(text_regions, confidence_scores))
        
        if detections_with_scores:
            print(f"Detected {len(detections_with_scores)} raw text regions (unfiltered, will be filtered by size)")
        
        # Return as list of ((x1, y1, x2, y2), score) tuples
        return detections_with_scores
    except Exception as e:
        print(f"Error detecting text regions: {e}")
        import traceback
        traceback.print_exc()
        return None
