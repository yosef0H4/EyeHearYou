"""Text region filtering and sorting"""
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


def sort_text_regions_by_reading_order(text_regions, direction='horizontal_ltr', group_tolerance=0.5):
    """
    Sort text regions by reading order with configurable direction.
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        direction: Reading direction:
            'horizontal_ltr' (Left-Right, Top-Bottom) - Standard English
            'horizontal_rtl' (Right-Left, Top-Bottom) - Manga
            'vertical_ltr'   (Top-Bottom, Left-Right) - Vertical text (Columns LTR)
            'vertical_rtl'   (Top-Bottom, Right-Left) - Vertical text (Columns RTL)
            'ltr' (legacy) - Same as 'horizontal_ltr'
            'rtl' (legacy) - Same as 'horizontal_rtl'
        group_tolerance: Ratio of median height/width to group items (default: 0.8)
    
    Returns:
        Sorted list of bounding boxes in reading order
    """
    if not text_regions:
        return []
    
    # Ensure direction is valid (handle None or empty string)
    if not direction:
        direction = 'horizontal_ltr'
    
    # Backward compatibility: map old 'ltr'/'rtl' to new format
    if direction == 'ltr':
        direction = 'horizontal_ltr'
    elif direction == 'rtl':
        direction = 'horizontal_rtl'
    
    # Calculate dimensions for adaptive grouping
    widths = [(x2 - x1) for x1, y1, x2, y2 in text_regions]
    heights = [(y2 - y1) for x1, y1, x2, y2 in text_regions]
    median_h = np.median(heights) if heights else 30.0
    median_w = np.median(widths) if widths else 30.0
    
    # --- Horizontal Sorting (Rows first, then X) ---
    if direction.startswith('horizontal'):
        # Group regions into lines based on Y-coordinate
        adaptive_band = group_tolerance * median_h
        
        # Sort by Y top-to-bottom first
        y_sorted = sorted(text_regions, key=lambda r: r[1])
        
        lines = []
        current_line = []
        current_y = -1000
        
        for box in y_sorted:
            y1 = box[1]
            # If this box is close enough to the current line's Y, add it
            if current_line and abs(y1 - current_y) <= adaptive_band:
                current_line.append(box)
                # Update current_y to average of line (moving average)
                current_y = np.mean([(b[1] + b[3]) / 2 for b in current_line])
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [box]
                current_y = (box[1] + box[3]) / 2
        
        if current_line:
            lines.append(current_line)
            
        # Sort within lines
        sorted_regions = []
        for line in lines:
            if 'ltr' in direction:
                # Left to Right
                line.sort(key=lambda r: r[0])
            else:
                # Right to Left (Manga)
                line.sort(key=lambda r: r[0], reverse=True)
            sorted_regions.extend(line)
            
        return sorted_regions

    # --- Vertical Sorting (Columns first, then Y) ---
    elif direction.startswith('vertical'):
        # Group regions into columns based on X-coordinate
        adaptive_band = group_tolerance * median_w
        
        # Sort by X first (LTR or RTL depending on sub-type)
        if 'ltr' in direction:
            x_sorted = sorted(text_regions, key=lambda r: r[0])
        else:
            x_sorted = sorted(text_regions, key=lambda r: r[0], reverse=True)
            
        columns = []
        current_col = []
        current_x = -1000
        
        for box in x_sorted:
            x1 = box[0]
            if current_col and abs(x1 - current_x) <= adaptive_band:
                current_col.append(box)
                # Update current_x to average of column
                current_x = np.mean([(b[0] + b[2]) / 2 for b in current_col])
            else:
                if current_col:
                    columns.append(current_col)
                current_col = [box]
                current_x = (box[0] + box[2]) / 2
        
        if current_col:
            columns.append(current_col)
            
        # Sort within columns (Top to Bottom)
        sorted_regions = []
        for col in columns:
            col.sort(key=lambda r: r[1])
            sorted_regions.extend(col)
            
        return sorted_regions

    # Fallback: return original order
    return text_regions


def filter_text_regions(text_regions, image_shape, min_width_ratio=0.0, min_height_ratio=0.0, median_height_fraction=0.4):
    """
    Filter regions based on relative size and statistical outliers.
    Uses adaptive logic that works across different screen sizes and font sizes.
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        image_shape: Tuple (height, width) of the image
        min_width_ratio: Minimum width as fraction of image width (e.g., 0.01 for 1%)
        min_height_ratio: Minimum height as fraction of image height
        median_height_fraction: If a box is smaller than (median_height * this_fraction), discard it.
                                Default 0.4 means "discard if smaller than 40% of average text size"
    
    Returns:
        Filtered list of bounding boxes
    """
    if not text_regions:
        return []

    img_h, img_w = image_shape[:2]
    
    # 1. Calculate dimensions for all boxes
    valid_regions = []
    heights = []
    
    for box in text_regions:
        x1, y1, x2, y2 = box
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))
        
        w = x2 - x1
        h = y2 - y1
        
        # Basic sanity check
        if w <= 0 or h <= 0:
            continue
        
        heights.append(h)
        valid_regions.append((x1, y1, x2, y2))

    if not valid_regions:
        return []

    # 2. Calculate Statistical Median Height
    # This tells us how big the "normal" text is on this specific screen
    median_h = np.median(heights) if heights else 0
    
    filtered = []
    for x1, y1, x2, y2 in valid_regions:
        w = x2 - x1
        h = y2 - y1
        
        # A. Filter by Image Ratio (e.g., "Must be at least 1% of screen height")
        # Good for removing tiny specks regardless of content
        if min_height_ratio > 0 and h < (img_h * min_height_ratio):
            continue
        if min_width_ratio > 0 and w < (img_w * min_width_ratio):
            continue
            
        # B. Filter by Content Statistics (The "Smart" Filter)
        # If this box is tiny compared to the median text size, it's noise.
        # Exception: Punctuation (., " -) might be small, but usually RapidOCR groups them with text.
        # We allow small boxes if they are very wide (like a horizontal line/separator)
        if median_h > 0 and h < (median_h * median_height_fraction) and w < (median_h * 2):
            continue
            
        filtered.append((x1, y1, x2, y2))
        
    return filtered


def generate_selection_mask(image_shape, selection_ops, base_state=True):
    """
    Generate a binary mask (255=Keep, 0=Discard) based on selection operations.
    Ops are normalized coordinates (0-1).
    
    Args:
        image_shape: Tuple (height, width) of the image
        selection_ops: List of {"op": "add"|"sub", "rect": (x,y,w,h)} normalized 0-1
        base_state: True = Select All (White), False = Deselect All (Black)
    
    Returns:
        Binary mask as numpy array (255=Keep, 0=Discard), or None if cv2 unavailable
    """
    if cv2 is None:
        return None

    h, w = image_shape[:2]
    # Start with base state
    initial_color = 255 if base_state else 0
    mask = np.full((h, w), initial_color, dtype=np.uint8)
    
    for op in selection_ops:
        mode = op['op']
        nx, ny, nw, nh = op['rect']
        
        # Convert normalized to pixel
        x1 = int(nx * w)
        y1 = int(ny * h)
        x2 = int((nx + nw) * w)
        y2 = int((ny + nh) * h)
        
        # Clamp
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            color = 255 if mode == 'add' else 0
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
        
    return mask


def filter_regions_by_mask(regions, mask, threshold=0.1):
    """
    Filter text regions. Keep if > threshold of the region is white in the mask.
    
    Args:
        regions: List of (x1, y1, x2, y2) bounding boxes
        mask: Binary mask (255=Keep, 0=Discard)
        threshold: Minimum ratio of white pixels to keep region (default 0.1 = 10%)
    
    Returns:
        Filtered list of bounding boxes
    """
    if not regions or mask is None or cv2 is None:
        return regions
        
    filtered = []
    for box in regions:
        x1, y1, x2, y2 = box
        # Clamp
        x1, x2 = max(0, x1), min(mask.shape[1], x2)
        y1, y2 = max(0, y1), min(mask.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Crop mask area
        mask_roi = mask[y1:y2, x1:x2]
        if mask_roi.size == 0:
            continue
            
        # Calculate white pixel ratio
        white_pixels = cv2.countNonZero(mask_roi)
        total_pixels = mask_roi.size
        
        if (white_pixels / total_pixels) > threshold:
            filtered.append(box)
            
    return filtered


def filter_contained_boxes(boxes, threshold=0.9):
    """
    Remove boxes that are significantly contained within other larger boxes.
    Useful when a user manually selects a large area that covers auto-detected regions.
    
    Args:
        boxes: List of (x1, y1, x2, y2) tuples
        threshold: Fraction of the smaller box area that must be covered by the larger box
                   to consider it "contained" (0.0-1.0). Default 0.9.
    
    Returns:
        List of boxes with contained ones removed.
    """
    if not boxes:
        return []
        
    # Calculate areas
    box_areas = []
    for b in boxes:
        w = b[2] - b[0]
        h = b[3] - b[1]
        box_areas.append(w * h)
    
    keep = [True] * len(boxes)
    
    for i in range(len(boxes)):
        if not keep[i]:
            continue
            
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue
            
            # Check if box[i] is inside box[j] or vice versa
            # We want to keep the LARGER box
            
            # Get intersection
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h
            
            if inter_area == 0:
                continue
                
            # Check containment
            # Is i contained in j?
            if inter_area >= (box_areas[i] * threshold):
                # i is inside j. 
                # If j is larger (or equal), discard i.
                if box_areas[j] >= box_areas[i]:
                    keep[i] = False
                    break 
            
            # Is j contained in i?
            elif inter_area >= (box_areas[j] * threshold):
                # j is inside i.
                # If i is larger, discard j.
                if box_areas[i] >= box_areas[j]:
                    keep[j] = False
    
    return [boxes[i] for i in range(len(boxes)) if keep[i]]


def get_regions_from_mask(mask):
    """
    Convert white areas in mask to bounding boxes (Manual Mode).
    
    Args:
        mask: Binary mask (255=Keep, 0=Discard)
    
    Returns:
        List of (x1, y1, x2, y2) bounding boxes
    """
    if mask is None or cv2 is None:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 4 and h > 4:  # Min size filter
            regions.append((x, y, x + w, y + h))
    return regions


