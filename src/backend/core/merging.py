"""Text box merging functionality"""
from typing import List, Tuple, Dict, Any


def get_overlap(min1, max1, min2, max2):
    """Calculate overlap amount between two ranges"""
    return max(0, min(max1, max2) - max(min1, min2))


def should_merge(box1, box2, v_tol=30, h_tol=50, w_ratio=0.3):
    """
    Determine if two boxes should be merged based on horizontal OR vertical proximity.
    
    This function implements two types of merging:
    1. Horizontal Merging (Words → Lines): Merges boxes that are side-by-side
    2. Vertical Merging (Lines → Paragraphs): Merges boxes that are stacked
    
    Args:
        box1, box2: (x1, y1, x2, y2) tuples
        v_tol: Vertical tolerance (gap between lines)
        h_tol: Horizontal tolerance (gap between words)
        w_ratio: Width ratio threshold (used for vertical merging validation)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Dimensions
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    min_h = min(h1, h2)
    min_w = min(w1, w2)
    
    # --- 1. Horizontal Merge (Words on same line) ---
    # Condition: High vertical overlap (they are on the same "row")
    y_overlap = get_overlap(y1_1, y2_1, y1_2, y2_2)
    is_same_row = y_overlap > (min_h * 0.4)  # Overlap at least 40% of height
    
    if is_same_row:
        # Check horizontal gap (distance between right of A and left of B)
        # We allow negative gap (overlap) or positive gap (space)
        gap = max(x1_1, x1_2) - min(x2_1, x2_2)
        
        # Merge if gap is small enough (roughly 1-2 character widths)
        # Note: h_tol here acts as "max space width"
        if gap < h_tol:
            return True

    # --- 2. Vertical Merge (Lines in a paragraph) ---
    # Condition: High horizontal overlap OR visually aligned (center/left)
    x_overlap = get_overlap(x1_1, x2_1, x1_2, x2_2)
    
    # Check if vertically adjacent (one bottom close to other top)
    v_gap = max(y1_1, y1_2) - min(y2_1, y2_2)
    
    if v_gap < v_tol:
        # Strict alignment check:
        # Either significant horizontal overlap (stacked text)
        # w_ratio controls how much overlap is needed relative to the smaller box width
        if x_overlap > (min_w * w_ratio):
            return True
            
        # OR Left-aligned (roughly)
        if abs(x1_1 - x1_2) < 20: 
            return True
            
        # OR Center-aligned (roughly)
        center1 = (x1_1 + x2_1) / 2
        center2 = (x1_2 + x2_2) / 2
        if abs(center1 - center2) < 20:
            return True
            
    return False


def merge_close_text_boxes(text_regions, vertical_tolerance=30, horizontal_tolerance=50, width_ratio_threshold=0.3):
    """
    Merge text boxes that are close together (both horizontally and vertically).
    
    This function handles two types of merging:
    1. Horizontal Merging: Merges words that are side-by-side into lines
    2. Vertical Merging: Merges lines that are stacked into paragraphs
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        vertical_tolerance: Max gap between lines (pixels)
        horizontal_tolerance: Max gap between words (pixels)
        width_ratio_threshold: (Legacy/Optional) Kept for API compatibility
    
    Returns:
        Tuple of (merged_boxes, is_merged_list, original_groups) where:
        - merged_boxes: List of merged bounding boxes as (x1, y1, x2, y2) tuples
        - is_merged_list: List of booleans indicating if each box is a merged box (True) or original (False)
        - original_groups: List of lists, where each inner list contains the original boxes that were merged
                          into the corresponding merged box (empty list for non-merged boxes)
    """
    if not text_regions or len(text_regions) <= 1:
        return text_regions, [False] * len(text_regions) if text_regions else [], [[list(r)] for r in text_regions] if text_regions else []
    
    # Convert to list of lists for easier manipulation
    # Store as simple list of coordinates
    boxes = [list(bbox) for bbox in text_regions]
    
    # Adjacency matrix or graph-based merging is cleaner, 
    # but iterative clustering is robust for this use case.
    merged = []
    is_merged = []
    original_groups = []
    
    used = [False] * len(boxes)
    
    for i, box1 in enumerate(boxes):
        if used[i]:
            continue
        
        # Start a group with this box
        group = [tuple(box1)]
        used[i] = True
        
        # Iteratively try to add boxes to this group
        # (This handles A->B->C chaining)
        changed = True
        while changed:
            changed = False
            for j, box2 in enumerate(boxes):
                if used[j]:
                    continue
                
                # Check if box2 can be merged with ANY box currently in the group
                can_merge = False
                for group_box in group:
                    if should_merge(group_box, box2, vertical_tolerance, horizontal_tolerance, width_ratio_threshold):
                        can_merge = True
                        break
                
                if can_merge:
                    group.append(tuple(box2))
                    used[j] = True
                    changed = True
        
        # Finalize the group into a single bounding box
        if len(group) > 0:
            min_x = min(b[0] for b in group)
            min_y = min(b[1] for b in group)
            max_x = max(b[2] for b in group)
            max_y = max(b[3] for b in group)
            
            merged.append((min_x, min_y, max_x, max_y))
            # True if multiple boxes were combined
            is_merged.append(len(group) > 1)
            original_groups.append(group)
            
    return merged, is_merged, original_groups
