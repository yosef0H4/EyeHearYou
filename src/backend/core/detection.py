"""Text detection using PaddleOCR"""
import numpy as np
import cv2
from PIL import Image

# Optional imports for text detection (PaddleOCR)
try:
    from paddleocr import TextDetection
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    TextDetection = None
    cv2 = None

from .image_utils import check_gpu_available
from .filtering import filter_text_regions, sort_text_regions_by_reading_order
from .task_manager import task_manager


def detect_text_regions(image, min_confidence=0.6, min_width=30, min_height=30):
    """
    Detect text regions in the image using PaddleOCR TextDetection module (detection only).
    Returns a list of bounding boxes as (x1, y1, x2, y2) tuples.
    
    Args:
        image: PIL Image to process
        min_confidence: Minimum confidence score (0.0-1.0) to keep a detection. 
                       Higher values filter out more false positives. Default: 0.5
    """
    if task_manager.is_cancelled():
        return None
    
    if not PADDLEOCR_AVAILABLE or TextDetection is None or cv2 is None:
        return None
    
    try:
        # Check if GPU is available (for informational purposes)
        use_gpu = check_gpu_available()
        if use_gpu:
            print("GPU detected, using GPU acceleration for text detection...")
            device = 'gpu'
        else:
            print("Using CPU for text detection...")
            device = 'cpu'
        
        # Use TextDetection module directly for detection-only
        # According to official docs: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html
        # Initialize TextDetection model (use mobile model for efficiency)
        det_model = TextDetection(model_name="PP-OCRv5_mobile_det", device=device)
        
        # Convert PIL to numpy array (RGB)
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        # Convert RGB to BGR for OpenCV/PaddleOCR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection - predict() returns a list of results
        output = det_model.predict(img_bgr, batch_size=1)
        
        # Check for cancellation after detection
        if task_manager.is_cancelled():
            return None
        
        # Extract bounding boxes from TextDetection output
        # Output format: [{'res': {'dt_polys': array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]), 'dt_scores': [...]}}]
        text_regions = []
        confidence_scores = []
        
        if output and len(output) > 0:
            for result in output:
                # Get the result dictionary
                if hasattr(result, 'json'):
                    res_dict = result.json
                elif isinstance(result, dict):
                    res_dict = result
                else:
                    continue
                
                # Extract dt_polys (detection polygons) and dt_scores (confidence scores) from the result
                if 'res' in res_dict and 'dt_polys' in res_dict['res']:
                    dt_polys = res_dict['res']['dt_polys']
                    dt_scores = res_dict['res'].get('dt_scores', [])
                    
                    # Process polygons (can be numpy array or list)
                    # Convert numpy array to list for consistent processing
                    if isinstance(dt_polys, np.ndarray):
                        dt_polys = dt_polys.tolist()
                    if isinstance(dt_scores, np.ndarray):
                        dt_scores = dt_scores.tolist()
                    elif not isinstance(dt_scores, list):
                        dt_scores = []
                    
                    if isinstance(dt_polys, list):
                        for idx, poly in enumerate(dt_polys):
                            # Handle different polygon formats
                            if isinstance(poly, np.ndarray):
                                poly = poly.tolist()
                            
                            if isinstance(poly, list) and len(poly) >= 4:
                                # Check if it's a list of [x, y] pairs
                                first_elem = poly[0]
                                if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) >= 2:
                                    # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                                    x_coords = [float(point[0]) for point in poly]
                                    y_coords = [float(point[1]) for point in poly]
                                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                                    
                                    # Only add if region is valid
                                    if x2 > x1 and y2 > y1:
                                        text_regions.append((x1, y1, x2, y2))
                                        # Get corresponding confidence score (default to 1.0 if not available)
                                        score = float(dt_scores[idx]) if idx < len(dt_scores) else 1.0
                                        confidence_scores.append(score)
        
        # Filter by confidence score
        filtered_regions = []
        for region, score in zip(text_regions, confidence_scores):
            if score >= min_confidence:
                filtered_regions.append(region)
        
        if len(text_regions) > len(filtered_regions):
            print(f"Filtered out {len(text_regions) - len(filtered_regions)} low-confidence detections (confidence < {min_confidence})")
        
        # Then filter out small regions (UI elements, icons, etc.)
        text_regions = filter_text_regions(filtered_regions, (img_height, img_width), 
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


def detect_text_regions_unfiltered(image, min_confidence=0.6):
    """
    Detect text regions in the image using PaddleOCR, but return ALL detections
    with their confidence scores (before any filtering).
    This allows the backend to cache raw detections and apply filters later.
    
    Args:
        image: PIL Image to process
        min_confidence: Minimum confidence score (0.0-1.0) to keep a detection.
                      Note: This is applied during detection, but we return all detections
                      with scores so confidence filtering can be reapplied later.
    
    Returns:
        List of tuples: ((x1, y1, x2, y2), confidence_score) for all valid detections
    """
    if task_manager.is_cancelled():
        return None
    
    if not PADDLEOCR_AVAILABLE or TextDetection is None or cv2 is None:
        return None
    
    try:
        # Check if GPU is available
        use_gpu = check_gpu_available()
        if use_gpu:
            print("GPU detected, using GPU acceleration for text detection...")
            device = 'gpu'
        else:
            print("Using CPU for text detection...")
            device = 'cpu'
        
        # Use TextDetection module directly for detection-only
        det_model = TextDetection(model_name="PP-OCRv5_mobile_det", device=device)
        
        # Convert PIL to numpy array (RGB)
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        # Convert RGB to BGR for OpenCV/PaddleOCR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection
        output = det_model.predict(img_bgr, batch_size=1)
        
        # Check for cancellation after detection
        if task_manager.is_cancelled():
            return None
        
        # Extract bounding boxes from TextDetection output
        text_regions = []
        confidence_scores = []
        
        if output and len(output) > 0:
            for result in output:
                if hasattr(result, 'json'):
                    res_dict = result.json
                elif isinstance(result, dict):
                    res_dict = result
                else:
                    continue
                
                if 'res' in res_dict and 'dt_polys' in res_dict['res']:
                    dt_polys = res_dict['res']['dt_polys']
                    dt_scores = res_dict['res'].get('dt_scores', [])
                    
                    if isinstance(dt_polys, np.ndarray):
                        dt_polys = dt_polys.tolist()
                    if isinstance(dt_scores, np.ndarray):
                        dt_scores = dt_scores.tolist()
                    elif not isinstance(dt_scores, list):
                        dt_scores = []
                    
                    if isinstance(dt_polys, list):
                        for idx, poly in enumerate(dt_polys):
                            if isinstance(poly, np.ndarray):
                                poly = poly.tolist()
                            
                            if isinstance(poly, list) and len(poly) >= 4:
                                first_elem = poly[0]
                                if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) >= 2:
                                    x_coords = [float(point[0]) for point in poly]
                                    y_coords = [float(point[1]) for point in poly]
                                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                                    
                                    if x2 > x1 and y2 > y1:
                                        text_regions.append((x1, y1, x2, y2))
                                        score = float(dt_scores[idx]) if idx < len(dt_scores) else 1.0
                                        confidence_scores.append(score)
        
        # Return all detections with their scores (no filtering applied yet)
        # This allows the backend to cache and reapply filters later
        detections_with_scores = list(zip(text_regions, confidence_scores))
        
        if detections_with_scores:
            print(f"Detected {len(detections_with_scores)} raw text regions (unfiltered, will be filtered by confidence/size)")
        
        # Return as list of ((x1, y1, x2, y2), score) tuples
        return detections_with_scores
    except Exception as e:
        print(f"Error detecting text regions: {e}")
        import traceback
        traceback.print_exc()
        return None


