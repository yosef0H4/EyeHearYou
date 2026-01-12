"""Debug image saving functionality"""
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

# Optional imports for debug image drawing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


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
        if text_regions and CV2_AVAILABLE and cv2 is not None:
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





