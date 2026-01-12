"""Text box merging functionality"""
from typing import List, Tuple


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
        Tuple of (merged_boxes, is_merged_list, original_groups) where:
        - merged_boxes: List of merged bounding boxes as (x1, y1, x2, y2) tuples
        - is_merged_list: List of booleans indicating if each box is a merged box (True) or original (False)
        - original_groups: List of lists, where each inner list contains the original boxes that were merged
                          into the corresponding merged box (empty list for non-merged boxes)
    """
    if not text_regions or len(text_regions) <= 1:
        return text_regions, [False] * len(text_regions) if text_regions else [], [[]] * len(text_regions) if text_regions else []
    
    # Convert to list of lists for easier manipulation
    boxes = [list(bbox) for bbox in text_regions]
    merged = []
    is_merged = []
    original_groups = []
    used = [False] * len(boxes)
    
    for i, box1 in enumerate(boxes):
        if used[i]:
            continue
        
        # Start a group with this box (store as tuple for immutability)
        group = [tuple(box1)]
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
                    group.append(tuple(box2))
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
            original_groups.append(list(group))  # Store original boxes
        else:
            # Single box, keep as is
            merged.append(group[0])
            is_merged.append(False)  # This is an original box
            original_groups.append([])  # No original boxes (it's the original itself)
    
    return merged, is_merged, original_groups


