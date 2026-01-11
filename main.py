"""
Screenshot OCR Application
Captures screenshots on Ctrl+Shift+Alt+Z and extracts text using OpenAI Vision API
Uses text detection to only send text regions to the API, reducing costs
"""
import base64
import json
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import keyboard
import numpy as np
import pygetwindow as gw
from openai import OpenAI
from PIL import Image, ImageGrab

# Optional imports for text detection (PaddleOCR)
try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    paddle = None

try:
    from paddleocr import PaddleOCR, TextDetection
    import cv2
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None
    TextDetection = None
    cv2 = None

# Configuration file path
CONFIG_FILE = Path("config.json")


def load_config():
    """Load configuration from config.json or create default template"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Validate required fields
                if not all(key in config for key in ["api_url", "api_key", "model"]):
                    print("Error: config.json is missing required fields (api_url, api_key, model)")
                    sys.exit(1)
                return config
        except json.JSONDecodeError:
            print("Error: config.json is not valid JSON")
            sys.exit(1)
    else:
        # Create default config template
        default_config = {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model": "gpt-4-vision-preview",
            "max_image_dimension": 1080,
            "text_detection": {
                "min_confidence": 0.6,
                "min_width": 30,
                "min_height": 30,
                "merge_vertical_tolerance": 30,
                "merge_horizontal_tolerance": 50,
                "merge_width_ratio_threshold": 0.3
            }
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"Created {CONFIG_FILE} with default LM Studio configuration.")
        print("Please edit it with your API URL, key, and model.")
        sys.exit(0)


def capture_screenshot():
    """Capture the active window"""
    try:
        # Get the active window
        active_window = gw.getActiveWindow()
        if active_window is None:
            print("Error: No active window found")
            return None
        
        # Get window position and size
        left = active_window.left
        top = active_window.top
        width = active_window.width
        height = active_window.height
        
        # Capture only the active window region
        bbox = (left, top, left + width, top + height)
        screenshot = ImageGrab.grab(bbox=bbox)
        
        return screenshot
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None


def resize_image_if_needed(image, max_dimension=2048):
    """
    Resize image if it exceeds maximum dimension, maintaining aspect ratio.
    This prevents context window overflow when sending to vision APIs.
    
    Args:
        image: PIL Image to resize
        max_dimension: Maximum width or height in pixels (default: 2048)
    
    Returns:
        Resized PIL Image (or original if no resize needed)
    """
    width, height = image.size
    max_size = max(width, height)
    
    if max_size <= max_dimension:
        return image
    
    # Calculate scaling factor to fit within max_dimension
    scale = max_dimension / max_size
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Use high-quality resampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized image from {width}x{height} to {new_width}x{new_height} to fit context window")
    return resized


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def check_gpu_available():
    """
    Check if GPU is available for PaddleOCR.
    Returns True if GPU is available, False otherwise.
    """
    if not PADDLE_AVAILABLE or paddle is None:
        return False
    
    try:
        # Check if PaddlePaddle was compiled with CUDA support
        if hasattr(paddle.device, 'is_compiled_with_cuda'):
            if not paddle.device.is_compiled_with_cuda():
                return False
            # Also check if CUDA devices are actually available
            try:
                # Method 1: Check get_device() which returns "gpu:0" or "cpu"
                if hasattr(paddle.device, 'get_device'):
                    device = paddle.device.get_device()
                    if device and ('gpu' in str(device).lower() or 'cuda' in str(device).lower()):
                        return True
                # Method 2: Check CUDA device count
                if hasattr(paddle.device, 'cuda') and hasattr(paddle.device.cuda, 'device_count'):
                    if paddle.device.cuda.device_count() > 0:
                        return True
            except Exception as e:
                # If we can't check device count, assume GPU is available if compiled with CUDA
                # This is a safe fallback since is_compiled_with_cuda() already passed
                return True
        # Fallback: try to check if CUDA is available (older PaddlePaddle versions < 2.5)
        # Note: paddle.fluid was removed in PaddlePaddle 3.0+, so this will fail silently
        try:
            import paddle.fluid as fluid
            if hasattr(fluid, 'is_compiled_with_cuda'):
                return fluid.is_compiled_with_cuda()
        except (ImportError, AttributeError):
            # paddle.fluid doesn't exist in PaddlePaddle 3.0+, ignore
            pass
        return False
    except Exception as e:
        # Log the exception for debugging but return False
        print(f"Warning: Error checking GPU availability: {e}")
        return False


def sort_text_regions_by_reading_order(text_regions, direction='hor_ltr', band_ratio=0.5):
    """
    Sort text regions by reading order (top-to-bottom, left-to-right for English).
    Similar to comic-translate's sorting logic.
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        direction: Reading direction ('hor_ltr' for English, 'hor_rtl' for RTL languages)
        band_ratio: Ratio for grouping items on the same line (default: 0.5)
    
    Returns:
        Sorted list of bounding boxes in reading order
    """
    if not text_regions:
        return []
    
    # Calculate median height for adaptive band grouping
    heights = [(y2 - y1) for x1, y1, x2, y2 in text_regions]
    median_h = np.median(heights) if heights else 30.0
    adaptive_band = band_ratio * median_h
    
    # Group regions into lines (regions with similar y-coordinates)
    lines = []
    used = [False] * len(text_regions)
    
    for i, (x1, y1, x2, y2) in enumerate(text_regions):
        if used[i]:
            continue
        
        # Start a new line with this region
        line = [(x1, y1, x2, y2)]
        used[i] = True
        center_y = (y1 + y2) / 2
        
        # Find other regions on the same line
        for j, (x1_j, y1_j, x2_j, y2_j) in enumerate(text_regions):
            if used[j]:
                continue
            center_y_j = (y1_j + y2_j) / 2
            if abs(center_y - center_y_j) <= adaptive_band:
                line.append((x1_j, y1_j, x2_j, y2_j))
                used[j] = True
        
        lines.append(line)
    
    # Sort items within each line by x-coordinate (left-to-right for LTR)
    for line_idx, line in enumerate(lines):
        if direction == 'hor_ltr':
            lines[line_idx] = sorted(line, key=lambda bbox: bbox[0])  # Sort by x1
        else:  # hor_rtl
            lines[line_idx] = sorted(line, key=lambda bbox: -bbox[0], reverse=True)  # Sort by x1 descending
    
    # Sort lines by y-coordinate (top-to-bottom)
    lines.sort(key=lambda line: min(bbox[1] for bbox in line))  # Sort by min y1
    
    # Flatten back to list of regions
    sorted_regions = []
    for line in lines:
        sorted_regions.extend(line)
    
    return sorted_regions


def filter_text_regions(text_regions, image_shape, min_width=30, min_height=30):
    """
    Filter out small text regions that are likely UI elements, icons, or noise.
    Similar to comic-translate's filter_and_fix_bboxes but with larger thresholds.
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        image_shape: Tuple (height, width) of the image
        min_width: Minimum width in pixels to keep (default: 30)
        min_height: Minimum height in pixels to keep (default: 30)
    
    Returns:
        Filtered list of bounding boxes
    """
    if not text_regions:
        return []
    
    img_h, img_w = image_shape[:2]
    filtered = []
    
    for x1, y1, x2, y2 in text_regions:
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))
        
        # Calculate dimensions
        w = x2 - x1
        h = y2 - y1
        
        # Filter out invalid or too small regions
        if w <= 0 or h <= 0:
            continue
        
        # Filter by minimum size (width and height)
        if w <= min_width or h <= min_height:
            continue
        
        filtered.append((x1, y1, x2, y2))
    
    return filtered


def detect_text_regions(image, min_confidence=0.6, min_width=30, min_height=30):
    """
    Detect text regions in the image using PaddleOCR TextDetection module (detection only).
    Returns a list of bounding boxes as (x1, y1, x2, y2) tuples.
    
    Args:
        image: PIL Image to process
        min_confidence: Minimum confidence score (0.0-1.0) to keep a detection. 
                       Higher values filter out more false positives. Default: 0.5
    """
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
    after confidence filtering (before size filtering).
    This allows the frontend to do size filtering live.
    
    Args:
        image: PIL Image to process
        min_confidence: Minimum confidence score (0.0-1.0) to keep a detection.
    
    Returns:
        List of bounding boxes as (x1, y1, x2, y2) tuples (after confidence filtering only)
    """
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
        
        # Filter by confidence score only (no size filtering)
        filtered_regions = []
        for region, score in zip(text_regions, confidence_scores):
            if score >= min_confidence:
                filtered_regions.append(region)
        
        if len(text_regions) > len(filtered_regions):
            print(f"Filtered out {len(text_regions) - len(filtered_regions)} low-confidence detections (confidence < {min_confidence})")
        
        # Return regions after confidence filtering only (no size filtering, no sorting)
        # The frontend will handle size filtering and sorting for live updates
        return filtered_regions
    except Exception as e:
        print(f"Error detecting text regions: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error detecting text regions: {e}")
        print("Falling back to full image processing...")
        import traceback
        traceback.print_exc()
        return None


def merge_close_text_boxes(text_regions, vertical_tolerance=30, horizontal_tolerance=50, width_ratio_threshold=0.3):
    """
    Merge text boxes that are close together vertically (like split lines of dialogue).
    Based on comic-translate's _are_text_blocks_mergeable logic.
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        vertical_tolerance: Maximum vertical gap between boxes to merge (pixels)
        horizontal_tolerance: Maximum horizontal offset to consider boxes aligned (pixels)
        width_ratio_threshold: Minimum ratio of smaller width to larger width (0.0-1.0) to consider boxes similar
    
    Returns:
        Tuple of (merged_boxes, is_merged_list) where:
        - merged_boxes: List of merged bounding boxes as (x1, y1, x2, y2) tuples
        - is_merged_list: List of booleans indicating if each box is a merged box (True) or original (False)
    """
    if not text_regions or len(text_regions) <= 1:
        return text_regions, [False] * len(text_regions) if text_regions else []
    
    # Convert to list of lists for easier manipulation
    boxes = [list(bbox) for bbox in text_regions]
    merged = []
    is_merged = []
    used = [False] * len(boxes)
    
    for i, box1 in enumerate(boxes):
        if used[i]:
            continue
        
        # Start a group with this box
        group = [box1]
        used[i] = True
        
        # Try to find boxes that can be merged with this one
        changed = True
        while changed:
            changed = False
            for j, box2 in enumerate(boxes):
                if used[j] or j == i:
                    continue
                
                # Check if box2 can be merged with any box in the group
                can_merge = False
                for group_box in group:
                    x1_1, y1_1, x2_1, y2_1 = group_box
                    x1_2, y1_2, x2_2, y2_2 = box2
                    
                    # Calculate dimensions
                    w1 = x2_1 - x1_1
                    h1 = y2_1 - y1_1
                    w2 = x2_2 - x1_2
                    h2 = y2_2 - y1_2
                    
                    # Check vertical adjacency (one box's bottom is close to another's top)
                    vertical_gap1 = abs(y2_1 - y1_2)  # box1 bottom to box2 top
                    vertical_gap2 = abs(y2_2 - y1_1)  # box2 bottom to box1 top
                    is_vertically_adjacent = (vertical_gap1 < vertical_tolerance or 
                                            vertical_gap2 < vertical_tolerance)
                    
                    if not is_vertically_adjacent:
                        continue
                    
                    # Check horizontal alignment (similar x positions)
                    # Use a more lenient check - allow some horizontal offset
                    x_alignment = abs(x1_1 - x1_2)
                    if x_alignment > horizontal_tolerance:
                        continue
                    
                    # Check width similarity using ratio instead of absolute difference
                    # This is more lenient for dialogue boxes where lines can be different lengths
                    if w1 > 0 and w2 > 0:
                        width_ratio = min(w1, w2) / max(w1, w2)
                        if width_ratio < width_ratio_threshold:
                            continue
                    
                    # All checks passed, can merge
                    can_merge = True
                    break
                
                if can_merge:
                    group.append(box2)
                    used[j] = True
                    changed = True
        
        # Merge all boxes in the group into one bounding box
        if len(group) > 1:
            # Calculate the union bounding box
            min_x = min(box[0] for box in group)
            min_y = min(box[1] for box in group)
            max_x = max(box[2] for box in group)
            max_y = max(box[3] for box in group)
            merged.append((min_x, min_y, max_x, max_y))
            is_merged.append(True)  # This is a merged box
        else:
            # Single box, keep as is
            merged.append(tuple(group[0]))
            is_merged.append(False)  # This is an original box
    
    return merged, is_merged


def crop_text_regions(image, text_regions, padding=5):
    """
    Crop text regions from the image with optional padding.
    Returns a list of cropped PIL Images.
    """
    cropped_images = []
    img_width, img_height = image.size
    
    for x1, y1, x2, y2 in text_regions:
        # Add padding and ensure coordinates are within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Only crop if region is valid
        if x2 > x1 and y2 > y1:
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
    
    return cropped_images


def extract_text_with_vision_api(image, config):
    """Extract text from a single image using OpenAI Vision API"""
    # Initialize OpenAI client with custom base URL
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["api_url"]
    )
    
    # Start with configured max dimension, but reduce if we hit context window errors
    max_dimension = config.get("max_image_dimension", 1080)
    original_image = image
    retry_count = 0
    max_retries = 3
    
    while retry_count <= max_retries:
        try:
            # Resize image if too large to prevent context window overflow
            current_image = resize_image_if_needed(original_image, max_dimension=max_dimension)
            
            # Convert image to base64
            base64_image = image_to_base64(current_image)
            
            # Call Vision API
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the extracted text, no additional commentary."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            extracted_text = response.choices[0].message.content
            return extracted_text
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a context window error
            if "context length" in error_str.lower() or "context overflow" in error_str.lower():
                retry_count += 1
                if retry_count <= max_retries:
                    # Reduce max dimension by 25% and retry
                    max_dimension = int(max_dimension * 0.75)
                    print(f"Context window overflow detected. Retrying with smaller image size ({max_dimension}px max dimension)...")
                    continue
                else:
                    print(f"Error: Failed after {max_retries} retries with reduced image sizes. Original error: {e}")
                    return None
            else:
                # For other errors, don't retry
                print(f"Error calling Vision API: {e}")
                return None
    
    return None


def save_debug_images(full_image, text_regions, cropped_images, is_merged=None, output_dir="./tmp"):
    """
    Save debug image to tmp folder for inspection.
    Saves full image with bounding boxes drawn on detected text regions.
    Merged boxes are drawn in blue, original boxes in green.
    
    Args:
        full_image: PIL Image
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        cropped_images: List of cropped PIL Images
        is_merged: Optional list of booleans indicating if each box is merged (True) or original (False)
        output_dir: Directory to save debug images
    """
    try:
        # Create tmp directory if it doesn't exist
        tmp_path = Path(output_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full image with bounding boxes drawn
        if text_regions and cv2 is not None:
            # Convert PIL to numpy array
            img_array = np.array(full_image)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array.copy()
            
            # Default: all boxes are original (green) if is_merged not provided
            if is_merged is None:
                is_merged = [False] * len(text_regions)
            
            # Draw bounding boxes on the image
            for i, (x1, y1, x2, y2) in enumerate(text_regions):
                # Choose color: blue for merged boxes, green for original boxes
                if i < len(is_merged) and is_merged[i]:
                    color = (255, 0, 0)  # Blue in BGR
                    label_color = (255, 0, 0)
                else:
                    color = (0, 255, 0)  # Green in BGR
                    label_color = (0, 255, 0)
                
                # Draw rectangle
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                # Add region number label
                cv2.putText(img_bgr, str(i+1), (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            
            # Convert back to RGB and save
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            debug_img = Image.fromarray(img_rgb)
            debug_path = tmp_path / f"debug_full_{timestamp}.png"
            debug_img.save(debug_path)
            print(f"Saved debug image with bounding boxes: {debug_path}")
        
    except Exception as e:
        print(f"Warning: Failed to save debug images: {e}")


def extract_text_from_regions(full_image, config):
    """
    Extract text by first detecting text regions, then sending only those regions to the API.
    This is more cost-effective than sending the full image.
    """
    # Get text detection settings from config
    text_detection_config = config.get("text_detection", {})
    min_confidence = text_detection_config.get("min_confidence", 0.6)
    min_width = text_detection_config.get("min_width", 30)
    min_height = text_detection_config.get("min_height", 30)
    
    # Try to detect text regions first
    text_regions = detect_text_regions(full_image, 
                                      min_confidence=min_confidence,
                                      min_width=min_width,
                                      min_height=min_height)
    
    if text_regions is None or len(text_regions) == 0:
        # Fallback: if detection fails or no regions found, use full image
        print("No text regions detected or detection unavailable, using full image...")
        return extract_text_with_vision_api(full_image, config)
    
    print(f"Detected {len(text_regions)} text region(s), processing individually...")
    
    # Merge close text boxes (like split dialogue lines)
    merge_vertical_tolerance = text_detection_config.get("merge_vertical_tolerance", 30)
    merge_horizontal_tolerance = text_detection_config.get("merge_horizontal_tolerance", 50)
    merge_width_ratio_threshold = text_detection_config.get("merge_width_ratio_threshold", 0.3)
    text_regions, is_merged = merge_close_text_boxes(
        text_regions,
        vertical_tolerance=merge_vertical_tolerance,
        horizontal_tolerance=merge_horizontal_tolerance,
        width_ratio_threshold=merge_width_ratio_threshold
    )
    print(f"After merging close boxes: {len(text_regions)} text region(s)")
    
    # Crop text regions
    cropped_images = crop_text_regions(full_image, text_regions)
    
    if not cropped_images:
        print("No valid text regions to process, using full image...")
        return extract_text_with_vision_api(full_image, config)
    
    # Save debug images (merged boxes in blue, original boxes in green)
    save_debug_images(full_image, text_regions, cropped_images, is_merged=is_merged)
    
    # Process each cropped region
    all_texts = []
    for i, cropped_img in enumerate(cropped_images):
        print(f"Processing region {i+1}/{len(cropped_images)}...")
        text = extract_text_with_vision_api(cropped_img, config)
        if text and text.strip():
            all_texts.append(text.strip())
    
    # Combine all extracted texts
    if all_texts:
        return "\n".join(all_texts)
    else:
        return None


def process_screenshot():
    """Main function to capture screenshot and extract text"""
    print("\n" + "="*60)
    print("Capturing active window...")
    
    # Load config
    config = load_config()
    
    # Capture screenshot
    screenshot = capture_screenshot()
    if screenshot is None:
        return
    
    print("Detecting text regions...")
    
    # Extract text using Vision API (with region detection)
    extracted_text = extract_text_from_regions(screenshot, config)
    
    if extracted_text:
        print("\n" + "-"*60)
        print("EXTRACTED TEXT:")
        print("-"*60)
        print(extracted_text)
        print("-"*60)
    else:
        print("Failed to extract text from screenshot.")


def main():
    """Main entry point"""
    print("Screenshot OCR Application")
    print("="*60)
    
    # Check if config exists, if not create template
    if not CONFIG_FILE.exists():
        load_config()
        return
    
    # Load and validate config
    config = load_config()
    
    print(f"API URL: {config['api_url']}")
    print(f"Model: {config['model']}")
    print("\nPress Ctrl+Shift+Alt+Z to capture screenshot and extract text")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    # Register hotkey
    try:
        keyboard.add_hotkey("ctrl+shift+alt+z", process_screenshot)
        
        # Keep the program running
        keyboard.wait()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
