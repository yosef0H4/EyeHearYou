"""Text detection routes"""
import traceback
from fastapi import Request
from typing import List, Tuple

from ..core.detection import detect_text_regions_unfiltered
from ..core.filtering import filter_text_regions, sort_text_regions_by_reading_order
from ..core.merging import merge_close_text_boxes
from ..core.task_manager import task_manager
from ..state import state


def _convert_merge_result_to_frontend_format(
    merged_boxes: List[Tuple[int, int, int, int]],
    is_merged_list: List[bool],
    original_boxes_before_merge: List[Tuple[int, int, int, int]]
) -> List[dict]:
    """
    Convert Python merge result to frontend MergedBox format.
    For merged boxes, we need to track which original boxes were merged together.
    """
    result = []
    original_idx = 0
    
    for merged_box, is_merged in zip(merged_boxes, is_merged_list):
        if is_merged:
            # Find which original boxes were merged into this box
            # We need to reconstruct this from the merge algorithm
            # For now, we'll use a simpler approach: track original boxes during merge
            # This is a limitation - we'll need to modify merge_close_text_boxes to return groups
            # For MVP, we'll return empty originalBoxes and let frontend handle it
            result.append({
                "rect": list(merged_box),
                "count": 1,  # Will be updated if we track groups
                "originalBoxes": []  # Will be populated if we track groups
            })
        else:
            result.append({
                "rect": list(merged_box),
                "count": 1,
                "originalBoxes": [list(merged_box)]
            })
        original_idx += 1
    
    return result


async def detect_preview(request: Request):
    """
    Run PaddleOCR detection and compute filtering/merging in Python (backend-authoritative).
    Returns structured response with raw, filtered, and merged boxes for frontend rendering.
    
    Request body should include:
    - text_detection: dict with min_confidence, min_width, min_height, merge_vertical_tolerance,
                      merge_horizontal_tolerance, merge_width_ratio_threshold
    - run_detection: bool (optional, default True) - if False, use cached detections
    """
    if state.last_image is None:
        return {"status": "error", "message": "No image captured. Please capture a screenshot first."}
    
    try:
        data = await request.json()
        settings = data.get("text_detection", {})
        run_detection = data.get("run_detection", True)
        
        # Get or cache raw detections with scores
        cached_key = state.screenshot_version
        raw_detections_with_scores = None
        
        if not run_detection and cached_key in state.cached_raw_detections:
            # Use cached detections
            raw_detections_with_scores = state.cached_raw_detections[cached_key]
            print(f"Using cached detections for screenshot version {cached_key}")
        else:
            # Run new detection
            task_manager.start_task("Detection Preview")
            task_manager.emit_status("Running PaddleOCR detection...", progress=5)
            min_confidence_for_detection = float(settings.get("min_confidence", 0.6))
            raw_detections_with_scores = detect_text_regions_unfiltered(
                state.last_image,
                min_confidence=0.0  # Get all detections, we'll filter by confidence later
            )
            
            if task_manager.is_cancelled():
                task_manager.finish_task()
                return {"status": "error", "message": "Detection cancelled"}
            
            task_manager.emit_status("Processing detections...", progress=50)
            
            if raw_detections_with_scores is None:
                return {
                    "status": "error",
                    "message": "Failed to detect text regions. Is PaddleOCR installed?"
                }
            
            if raw_detections_with_scores:
                # Cache the raw detections with scores
                state.cached_raw_detections[cached_key] = raw_detections_with_scores
                print(f"Cached {len(raw_detections_with_scores)} raw detections for screenshot version {cached_key}")
            # If empty list, continue - we'll return empty results
        
        # Handle None case (shouldn't happen if we cached properly, but be safe)
        if raw_detections_with_scores is None:
            return {
                "status": "error",
                "message": "No cached detections available. Please run detection first."
            }
        
        if not raw_detections_with_scores:
            return {
                "status": "success",
                "raw": [],
                "filtered": [],
                "merged": []
            }
        
        # Extract settings
        min_confidence = float(settings.get("min_confidence", 0.6))
        min_width = int(settings.get("min_width", 30))
        min_height = int(settings.get("min_height", 30))
        merge_vertical_tolerance = int(settings.get("merge_vertical_tolerance", 30))
        merge_horizontal_tolerance = int(settings.get("merge_horizontal_tolerance", 50))
        merge_width_ratio_threshold = float(settings.get("merge_width_ratio_threshold", 0.3))
        
        img_height, img_width = state.last_image.size[1], state.last_image.size[0]
        
        # Step 1: Apply confidence filter
        confidence_filtered = [
            bbox for bbox, score in raw_detections_with_scores
            if score >= min_confidence
        ]
        
        # Step 2: Apply size filter
        size_filtered = filter_text_regions(
            confidence_filtered,
            (img_height, img_width),
            min_width=min_width,
            min_height=min_height
        )
        
        # Step 3: Sort by reading order (optional, but recommended for VN dialogue)
        sorted_boxes = sort_text_regions_by_reading_order(size_filtered, direction='hor_ltr')
        
        # Step 4: Merge close boxes
        merged_boxes, is_merged_list, original_groups = merge_close_text_boxes(
            sorted_boxes,
            vertical_tolerance=merge_vertical_tolerance,
            horizontal_tolerance=merge_horizontal_tolerance,
            width_ratio_threshold=merge_width_ratio_threshold
        )
        
        # Convert to frontend format
        merged_frontend = []
        for i, (merged_box, is_merged, orig_group) in enumerate(zip(merged_boxes, is_merged_list, original_groups)):
            if is_merged and orig_group:
                # Merged box with original boxes tracked
                merged_frontend.append({
                    "rect": list(merged_box),
                    "count": len(orig_group),
                    "originalBoxes": [list(bbox) for bbox in orig_group]
                })
            else:
                # Single box (not merged)
                merged_frontend.append({
                    "rect": list(merged_box),
                    "count": 1,
                    "originalBoxes": [list(merged_box)]
                })
        
        # Format raw boxes with scores for frontend
        try:
            raw_frontend = [
                {"bbox": list(bbox), "score": float(score)}
                for bbox, score in raw_detections_with_scores
            ]
        except (ValueError, TypeError) as e:
            print(f"Error formatting raw detections: {e}")
            print(f"Sample detection: {raw_detections_with_scores[0] if raw_detections_with_scores else 'empty'}")
            raise
        
        # Update state for backward compatibility
        state.unfiltered_detections = confidence_filtered
        state.last_detections = size_filtered
        
        # Finish task if we started one
        if task_manager.is_running():
            task_manager.emit_status("Detection complete", is_loading=False, progress=100)
            task_manager.finish_task()
        
        return {
            "status": "success",
            "raw": raw_frontend,
            "filtered": [list(bbox) for bbox in size_filtered],
            "merged": merged_frontend
        }
    except Exception as e:
        traceback.print_exc()
        error_msg = f"Error in detect_preview: {type(e).__name__}: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

