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
    Does NOT run the merge logic (JS handles that for live preview).
    Returns raw detection boxes as [x1, y1, x2, y2] tuples.
    """
    if state.last_image is None:
        return {"status": "error", "message": "No image captured. Please capture a screenshot first."}
    
    try:
        data = await request.json()
        settings = data.get("text_detection", {})
        
        # Run detection using main.py logic
        regions = ocr_app.detect_text_regions(
            state.last_image,
            min_confidence=float(settings.get("min_confidence", 0.6)),
            min_width=int(settings.get("min_width", 30)),
            min_height=int(settings.get("min_height", 30))
        )
        
        # Store for future use
        if regions:
            state.last_detections = regions
        else:
            state.last_detections = []
        
        return {
            "status": "success",
            "regions": state.last_detections  # List of (x1, y1, x2, y2) tuples
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

