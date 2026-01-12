"""Text region filtering and sorting"""
import numpy as np


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


