"""Screenshot routes"""
import asyncio
import json
import sys
from pathlib import Path

from fastapi.responses import StreamingResponse

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import main as ocr_app
from ..state import state
from ..hotkey import capture_and_update_state


async def capture():
    """Capture screenshot and return base64 for preview"""
    try:
        if capture_and_update_state():
            # Convert to base64
            image_b64 = ocr_app.image_to_base64(state.last_image)
            
            return {
                "status": "success", 
                "image": image_b64,
                "width": state.last_image.width,
                "height": state.last_image.height,
                "version": state.screenshot_version
            }
        return {"status": "error", "message": "Failed to capture screenshot"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def get_screenshot():
    """Get the current screenshot as base64"""
    if state.last_image is None:
        return {"status": "error", "message": "No screenshot available"}
    
    try:
        image_b64 = ocr_app.image_to_base64(state.last_image)
        return {
            "status": "success",
            "image": image_b64,
            "width": state.last_image.width,
            "height": state.last_image.height,
            "version": state.screenshot_version
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def screenshot_events():
    """Server-Sent Events stream for screenshot updates"""
    async def event_generator():
        last_version = state.screenshot_version
        while True:
            try:
                # Check for new screenshot (non-blocking)
                try:
                    event = state.screenshot_queue.get_nowait()
                    if event["version"] > last_version:
                        last_version = event["version"]
                        yield f"data: {json.dumps(event)}\n\n"
                except:
                    pass  # Queue is empty, continue
                
                # Also check state directly (in case queue was missed)
                if state.screenshot_version > last_version:
                    last_version = state.screenshot_version
                    if state.last_image:
                        yield f"data: {json.dumps({'version': last_version, 'width': state.last_image.width, 'height': state.last_image.height})}\n\n"
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in SSE stream: {e}")
                break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

