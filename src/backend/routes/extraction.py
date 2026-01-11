"""Text extraction routes"""
import sys
import traceback
from pathlib import Path
from fastapi import Request

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import main as ocr_app
from ..state import state


async def run_extraction(request: Request):
    """Run the full OCR extraction process using saved config"""
    if state.last_image is None:
        return {"status": "error", "message": "No image captured. Please capture a screenshot first."}
    
    try:
        # Reload config to ensure we use latest settings
        config = ocr_app.load_config()
        
        # Run the full extraction pipeline
        text = ocr_app.extract_text_from_regions(state.last_image, config)
        
        if text:
            return {"status": "success", "text": text}
        else:
            return {"status": "error", "message": "Failed to extract text"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

