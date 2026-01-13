"""Text detection using RapidOCR"""
import numpy as np
import cv2
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .image_utils import check_gpu_available
from .filtering import filter_text_regions, sort_text_regions_by_reading_order
from .task_manager import task_manager

# Global RapidOCR instance (lazy initialization)
_rapidocr_instance = None
_rapidocr_use_gpu = None


def _get_rapidocr_instance(use_gpu=None):
    """Get or create RapidOCR instance (singleton pattern)
    
    Args:
        use_gpu: If True, use GPU (CUDAExecutionProvider). If None, check config.
    """
    global _rapidocr_instance, _rapidocr_use_gpu
    
    # Determine GPU usage
    if use_gpu is None:
        # Check config for GPU setting
        from src.backend.core.config import load_config
        config = load_config()
        td_config = config.get("text_detection", {})
        use_gpu = td_config.get("use_gpu", False)
    
    # Reinitialize if GPU setting changed
    if _rapidocr_instance is None or _rapidocr_use_gpu != use_gpu:
        _rapidocr_use_gpu = use_gpu
        
        if use_gpu:
            # Check if GPU is available
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    print("[RapidOCR] Initializing with GPU support (CUDAExecutionProvider)...")
                    # RapidOCR-onnxruntime should automatically use CUDA if onnxruntime-gpu is installed
                    # We can try passing providers via kwargs
                    try:
                        _rapidocr_instance = RapidOCR(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    except TypeError:
                        # If providers kwarg not supported, try default (should auto-detect GPU)
                        _rapidocr_instance = RapidOCR()
                        print("[RapidOCR] Using default initialization (should auto-detect GPU)")
                else:
                    print("[RapidOCR] GPU requested but CUDAExecutionProvider not available, using CPU")
                    _rapidocr_instance = RapidOCR()
            except ImportError:
                print("[RapidOCR] onnxruntime-gpu not available, using CPU")
                _rapidocr_instance = RapidOCR()
        else:
            print("[RapidOCR] Initializing with CPU (use_gpu=False)...")
            _rapidocr_instance = RapidOCR()
    
    return _rapidocr_instance


def preload_rapidocr(test=True):
    """
    Preload the RapidOCR instance at startup.
    
    Args:
        test: If True, run a simple test to verify RapidOCR works
        
    Returns:
        True if RapidOCR loaded (and tested) successfully, False otherwise
    """
    print("[RapidOCR] Preloading RapidOCR model...")
    try:
        ocr = _get_rapidocr_instance()
        
        if test:
            print("[RapidOCR] Running startup test...")
            try:
                # Create a simple test image (white with black text)
                import numpy as np
                test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
                # Run detection on test image
                result, _ = ocr(test_image, use_det=True, use_rec=False)
                print("[RapidOCR] ✓ RapidOCR loaded and tested successfully!")
                return True
            except Exception as e:
                print(f"[RapidOCR] ⚠ RapidOCR loaded but test failed: {e}")
                return True  # Still return True, RapidOCR is loaded
        else:
            print("[RapidOCR] ✓ RapidOCR loaded successfully!")
            return True
    except Exception as e:
        print(f"[RapidOCR] ⚠ Failed to load RapidOCR: {e}")
        import traceback
        traceback.print_exc()
        return False


def detect_text_regions(image, min_width_ratio=0.0, min_height_ratio=0.0, median_height_fraction=0.4):
    """
    Detect text regions in the image using RapidOCR (detection only).
    Returns a list of bounding boxes as (x1, y1, x2, y2) tuples.
    
    Note: RapidOCR in detection-only mode does not provide confidence scores.
    Confidence scores are only available when recognition is enabled.
    
    Args:
        image: PIL Image to process
        min_width_ratio: Minimum width as fraction of image width (e.g., 0.01 for 1%)
        min_height_ratio: Minimum height as fraction of image height
        median_height_fraction: If a box is smaller than (median_height * this_fraction), discard it
    
    Returns:
        List of (x1, y1, x2, y2) tuples representing bounding boxes
    """
    if task_manager.is_cancelled():
        return None
    
    # Check config for GPU setting
    from src.backend.core.config import load_config
    config = load_config()
    td_config = config.get("text_detection", {})
    use_gpu = td_config.get("use_gpu", False)
    
    device_str = "GPU" if use_gpu else "CPU"
    print(f"Using RapidOCR for text detection ({device_str} via ONNX Runtime)...")
    
    # Get RapidOCR instance
    ocr = _get_rapidocr_instance(use_gpu=use_gpu)
    
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
    
    # Filter out small regions using adaptive logic
    text_regions = filter_text_regions(text_regions, (img_height, img_width), 
                                      min_width_ratio=min_width_ratio,
                                      min_height_ratio=min_height_ratio,
                                      median_height_fraction=median_height_fraction)
    
    # Sort regions by reading order (top-to-bottom, left-to-right for English)
    # Note: This is a fallback sort, the main sorting happens in worker.py with config
    text_regions = sort_text_regions_by_reading_order(text_regions, direction='horizontal_ltr')
    
    return text_regions


def detect_text_regions_unfiltered(image, config=None, use_cache=True):
    """
    Detect text regions in the image using RapidOCR, but return ALL detections
    (before size filtering).
    This allows the backend to cache raw detections and apply filters later.
    
    Note: RapidOCR in detection-only mode does not provide confidence scores.
    All detections are returned with a default score of 1.0 for compatibility.
    
    Args:
        image: PIL Image to process
        config: Optional config dict (needed for cache key generation)
        use_cache: Whether to check cache before running RapidOCR
    
    Returns:
        List of tuples: ((x1, y1, x2, y2), confidence_score) for all valid detections.
        Note: confidence_score is always 1.0 since RapidOCR doesn't provide scores in detection-only mode.
    """
    if task_manager.is_cancelled():
        return None
    
    # Try to use cache if available
    if use_cache and config:
        from src.backend.state import state
        from src.backend.core.preprocessing import get_preprocessing_hash
        
        preproc_hash = get_preprocessing_hash(config)
        cache_key = (state.screenshot_version, preproc_hash)
        
        if cache_key in state.cached_raw_detections:
            cached = state.cached_raw_detections[cache_key]
            print(f"Using cached raw detections ({len(cached)} regions)")
            return cached
    
    # Check config for GPU setting
    from src.backend.core.config import load_config
    config = load_config()
    td_config = config.get("text_detection", {})
    use_gpu = td_config.get("use_gpu", False)
    
    device_str = "GPU" if use_gpu else "CPU"
    print(f"Using RapidOCR for text detection ({device_str} via ONNX Runtime)...")
    
    # Get RapidOCR instance
    ocr = _get_rapidocr_instance(use_gpu=use_gpu)
    
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
    
    # Cache the results if config is provided
    if config:
        from src.backend.state import state
        from src.backend.core.preprocessing import get_preprocessing_hash
        
        preproc_hash = get_preprocessing_hash(config)
        cache_key = (state.screenshot_version, preproc_hash)
        state.cached_raw_detections[cache_key] = detections_with_scores
    
    # Return as list of ((x1, y1, x2, y2), score) tuples
    return detections_with_scores
