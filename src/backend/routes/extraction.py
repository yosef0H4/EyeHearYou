"""Text extraction routes"""
import traceback
from fastapi import Request

from ..core.config import load_config
from ..core.extraction import extract_text_from_regions
from ..state import state


async def run_extraction(request: Request):
    """Run the full OCR extraction process using saved config"""
    if state.last_image is None:
        return {"status": "error", "message": "No image captured. Please capture a screenshot first."}
    
    try:
        # Reload config to ensure we use latest settings
        config = load_config()
        
        # Run the full extraction pipeline
        text = extract_text_from_regions(state.last_image, config)
        
        if text:
            return {"status": "success", "text": text}
        else:
            return {"status": "error", "message": "Failed to extract text"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

