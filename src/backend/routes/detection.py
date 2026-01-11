"""Text detection routes"""
import sys
import traceback
from pathlib import Path
from fastapi import Request

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import main as ocr_app
from ..state import state


async def detect_preview(request: Request):
    """
    Run PaddleOCR detection on the current image with provided settings.
    Returns ALL detections after confidence filtering (before size filtering).
    Size filtering (min_width/min_height) is done in JavaScript for live updates.
    Merge logic is also done in JavaScript for live updates.
    """
    if state.last_image is None:
        return {"status": "error", "message": "No image captured. Please capture a screenshot first."}
    
    try:
        data = await request.json()
        settings = data.get("text_detection", {})
        
        # Run detection but return unfiltered detections (after confidence, before size filtering)
        # This allows the frontend to do size filtering live
        unfiltered_regions = ocr_app.detect_text_regions_unfiltered(
            state.last_image,
            min_confidence=float(settings.get("min_confidence", 0.6))
        )
        
        # Store unfiltered detections for live filtering
        if unfiltered_regions:
            state.unfiltered_detections = unfiltered_regions
            # Also store filtered version for backward compatibility
            # Apply initial size filtering with current settings
            min_width = int(settings.get("min_width", 30))
            min_height = int(settings.get("min_height", 30))
            img_height, img_width = state.last_image.size[1], state.last_image.size[0]
            filtered = ocr_app.filter_text_regions(unfiltered_regions, (img_height, img_width), 
                                                   min_width=min_width, min_height=min_height)
            state.last_detections = filtered
        else:
            state.unfiltered_detections = []
            state.last_detections = []
        
        return {
            "status": "success",
            "regions": state.unfiltered_detections  # Return unfiltered for live size filtering
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

