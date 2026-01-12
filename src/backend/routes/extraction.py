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
            # Store result in state for UI
            state.last_extracted_text = text
            state.last_extraction_version = state.screenshot_version
            return {"status": "success", "text": text}
        else:
            return {"status": "error", "message": "Failed to extract text"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


async def get_last_extraction():
    """Get the last extracted text from hotkey or manual extraction"""
    if state.last_extracted_text and state.last_extraction_version == state.screenshot_version:
        return {
            "status": "success",
            "text": state.last_extracted_text,
            "version": state.last_extraction_version
        }
    else:
        return {
            "status": "success",
            "text": None,
            "version": state.last_extraction_version,
            "message": "No extraction available for current screenshot"
        }

