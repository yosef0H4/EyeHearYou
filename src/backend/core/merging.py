"""Text box merging functionality"""
from typing import List, Tuple, Dict, Any


def get_overlap(min1, max1, min2, max2):
    """Calculate overlap amount between two ranges"""
    return max(0, min(max1, max2) - max(min1, min2))


def should_merge(box1, box2, v_ratio=0.5, h_ratio=1.5, w_ratio=0.3):
    """
    Adaptive merge based on the size of the boxes themselves.
    Uses ratios relative to text height instead of fixed pixels.
    
    This function implements two types of merging:
    1. Horizontal Merging (Words → Lines): Merges boxes that are side-by-side
    2. Vertical Merging (Lines → Paragraphs): Merges boxes that are stacked
    
    Args:
        box1, box2: (x1, y1, x2, y2) tuples
        v_ratio: Max vertical gap as a ratio of average box height (e.g. 0.5 = half a line height)
        h_ratio: Max horizontal gap as a ratio of average box height (e.g. 1.5 = 1.5x line height)
        w_ratio: Width ratio threshold (used for vertical merging validation)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Dimensions
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    min_h = min(h1, h2)
    
    # Determine "reference size" for these two boxes
    # We use the average height to be adaptive
    ref_h = (h1 + h2) / 2
    
    # Use adaptive ratios
    max_v_gap = ref_h * v_ratio
    max_h_gap = ref_h * h_ratio
    
    # --- 1. Horizontal Merge (Words on same line) ---
    # Condition: High vertical overlap (they are on the same "row")
    y_overlap = get_overlap(y1_1, y2_1, y1_2, y2_2)
    is_same_row = y_overlap > (min_h * 0.5)  # Overlap at least 50% of the smaller box's height
    
    if is_same_row:
        # Check horizontal gap (distance between right of A and left of B)
        gap = max(x1_1, x1_2) - min(x2_1, x2_2)
        
        # ADAPTIVE LOGIC: Gap limit depends on text size
        # If text is 100px tall, we allow 150px gap (h_ratio=1.5).
        # If text is 10px tall, we allow 15px gap.
        if gap < max_h_gap:
            return True

    # --- 2. Vertical Merge (Lines in a paragraph) ---
    # Check vertical gap
    v_gap = max(y1_1, y1_2) - min(y2_1, y2_2)
    
    # ADAPTIVE LOGIC: Vertical gap limit depends on text size
    if v_gap < max_v_gap:
        # For vertical merge, they must also be aligned horizontally
        x_overlap = get_overlap(x1_1, x2_1, x1_2, x2_2)
        
        # Must overlap horizontally by some amount relative to width
        # OR be left-aligned (x1 coords close)
        # Left alignment tolerance: ~50% of font height (slight indentation allowed)
        alignment_tol = ref_h * 0.5
        if x_overlap > 0 or abs(x1_1 - x1_2) < alignment_tol:
            return True
            
    return False


def merge_close_text_boxes(text_regions, vertical_ratio=0.5, horizontal_ratio=1.5, width_ratio_threshold=0.3):
    """
    Merge text boxes that are close together using adaptive ratios.
    
    This function handles two types of merging:
    1. Horizontal Merging: Merges words that are side-by-side into lines
    2. Vertical Merging: Merges lines that are stacked into paragraphs
    
    Args:
        text_regions: List of (x1, y1, x2, y2) bounding boxes
        vertical_ratio: Multiplier of text height for vertical merging (lines) - default 0.5
        horizontal_ratio: Multiplier of text height for horizontal merging (words) - default 1.5
        width_ratio_threshold: Width overlap ratio for vertical merging validation
    
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
    boxes = [list(bbox) for bbox in text_regions]
    
    # Iterative clustering
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
                    if should_merge(group_box, box2, v_ratio=vertical_ratio, h_ratio=horizontal_ratio, 
                                  w_ratio=width_ratio_threshold):
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
