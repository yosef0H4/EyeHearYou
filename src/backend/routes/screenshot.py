"""Screenshot routes"""
import asyncio
import json
from fastapi.responses import StreamingResponse

from ..core.image_utils import image_to_base64
from ..core.task_manager import task_manager
from ..state import state
from ..hotkey import capture_and_update_state


async def capture():
    """Capture screenshot and return base64 for preview"""
    try:
        if capture_and_update_state():
            # Convert to base64
            image_b64 = image_to_base64(state.last_image)
            
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
        image_b64 = image_to_base64(state.last_image)
        return {
            "status": "success",
            "image": image_b64,
            "width": state.last_image.width,
            "height": state.last_image.height,
            "version": state.screenshot_version
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def cancel_task():
    """Cancel currently running task"""
    task_manager.cancel_current_task()
    return {"status": "success", "message": "Cancellation requested"}


async def screenshot_events():
    """Server-Sent Events stream for screenshot AND status updates"""
    async def event_generator():
        last_version = state.screenshot_version
        
        while True:
            try:
                # 1. Check for State/Screenshot events
                try:
                    while not state.screenshot_queue.empty():
                        event = state.screenshot_queue.get_nowait()
                        # Ensure type is set for frontend discrimination
                        if "type" not in event:
                            event["type"] = "screenshot"
                        yield f"data: {json.dumps(event)}\n\n"
                except:
                    pass

                # 2. Check for Task Manager Status events
                try:
                    while not task_manager.message_queue.empty():
                        msg = task_manager.message_queue.get_nowait()
                        yield f"data: {json.dumps(msg)}\n\n"
                except:
                    pass
                
                # 3. Also check state directly (in case queue was missed)
                if state.screenshot_version > last_version:
                    last_version = state.screenshot_version
                    if state.last_image:
                        yield f"data: {json.dumps({'type': 'screenshot', 'version': last_version, 'width': state.last_image.width, 'height': state.last_image.height})}\n\n"
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in SSE stream: {e}")
                break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

