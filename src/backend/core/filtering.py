"""Text region filtering and sorting"""
import numpy as np


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
        group_tolerance: Ratio of median height/width to group items (default: 0.5)
    
    Returns:
        Sorted list of bounding boxes in reading order
    """
    if not text_regions:
        return []
    
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



