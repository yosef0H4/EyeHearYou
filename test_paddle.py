#!/usr/bin/env python3
"""
Simple CLI script to test PaddleOCR TextDetection module.
Usage: python test_paddle.py <image_path>
Returns text region bounding boxes as (x1, y1, x2, y2) tuples.
"""
import sys
import numpy as np
from PIL import Image
import cv2

try:
    from paddleocr import TextDetection
except ImportError as e:
    print(f"ERROR: Failed to import TextDetection: {e}", file=sys.stderr)
    sys.exit(1)

def detect_text_regions(image_path):
    """Detect text regions in an image and return bounding boxes."""
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV/PaddleOCR
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Initialize TextDetection model
        det_model = TextDetection(model_name="PP-OCRv5_mobile_det", device='gpu')
        
        # Run detection
        output = det_model.predict(img_bgr, batch_size=1)
        
        # Extract bounding boxes
        text_regions = []
        
        if output and len(output) > 0:
            for result in output:
                # Get the result dictionary
                if hasattr(result, 'json'):
                    res_dict = result.json
                elif isinstance(result, dict):
                    res_dict = result
                else:
                    continue
                
                # Extract dt_polys (detection polygons) from the result
                if 'res' in res_dict and 'dt_polys' in res_dict['res']:
                    dt_polys = res_dict['res']['dt_polys']
                    
                    # Process polygons (can be numpy array or list)
                    if isinstance(dt_polys, np.ndarray):
                        dt_polys = dt_polys.tolist()
                    
                    if isinstance(dt_polys, list):
                        for poly in dt_polys:
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
        
        return text_regions
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_paddle.py <image_path>", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    text_regions = detect_text_regions(image_path)
    
    if text_regions is None:
        print("ERROR: Detection failed", file=sys.stderr)
        sys.exit(1)
    elif len(text_regions) == 0:
        print("WARNING: No text regions detected", file=sys.stderr)
        sys.exit(1)
    else:
        # Output text regions
        for x1, y1, x2, y2 in text_regions:
            print(f"{x1},{y1},{x2},{y2}")
        sys.exit(0)

if __name__ == "__main__":
    main()
