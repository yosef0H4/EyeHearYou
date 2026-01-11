"""
FastAPI server for OCR configuration UI
Provides web interface for tweaking OCR settings with live preview
"""
import asyncio
import base64
import json
import threading
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional
from queue import Queue

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Import logic from existing main.py
import main as ocr_app

# Import keyboard for hotkey support
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

app = FastAPI()

# Global state to hold the last screenshot and detections
class AppState:
    last_image: Optional[Image.Image] = None
    last_detections: List = []
    image_scale: float = 1.0  # Track scaling if image was resized
    screenshot_version: int = 0  # Increment when new screenshot is captured
    screenshot_queue: Queue = Queue()  # Queue for SSE events

state = AppState()

# Pydantic models for API
class ConfigUpdate(BaseModel):
    api_url: str
    api_key: str
    model: str
    max_image_dimension: int
    text_detection: Dict[str, Any]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the single page UI"""
    try:
        with open("ui.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: ui.html not found</h1>", status_code=404)

@app.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        return ocr_app.load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def save_config(config: ConfigUpdate):
    """Save configuration to file"""
    try:
        config_dict = config.dict()
        with open(ocr_app.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        return {"status": "success", "message": "Config saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def capture_and_update_state():
    """Capture screenshot and update global state (called by hotkey)"""
    try:
        screenshot = ocr_app.capture_screenshot()
        if screenshot:
            state.last_image = screenshot
            state.last_detections = []  # Reset detections on new capture
            state.image_scale = 1.0  # Reset scale
            state.screenshot_version += 1  # Increment version
            
            # Notify SSE listeners
            state.screenshot_queue.put({
                "version": state.screenshot_version,
                "width": screenshot.width,
                "height": screenshot.height
            })
            
            print(f"[Hotkey] Screenshot captured: {screenshot.width}x{screenshot.height} (version {state.screenshot_version})")
            return True
        return False
    except Exception as e:
        print(f"[Hotkey] Error capturing screenshot: {e}")
        return False

@app.post("/capture")
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

@app.get("/screenshot")
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

@app.get("/screenshot/events")
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

@app.post("/detect_preview")
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
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/run_extraction")
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
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def setup_hotkey():
    """Setup keyboard hotkey in a background thread"""
    if not KEYBOARD_AVAILABLE:
        print("[Hotkey] Keyboard module not available. Hotkey disabled.")
        return
    
    try:
        keyboard.add_hotkey("ctrl+shift+alt+z", capture_and_update_state)
        print("[Hotkey] Registered Ctrl+Shift+Alt+Z for screenshot capture")
        print("[Hotkey] Press Ctrl+Shift+Alt+Z anywhere to capture screenshot")
        # Keep the keyboard listener running
        keyboard.wait()
    except Exception as e:
        print(f"[Hotkey] Failed to register hotkey: {e}")
        print("[Hotkey] You may need to run as administrator on Windows")

if __name__ == "__main__":
    import uvicorn
    
    # Setup hotkey in background thread
    if KEYBOARD_AVAILABLE:
        hotkey_thread = threading.Thread(target=setup_hotkey, daemon=True)
        hotkey_thread.start()
    
    print("="*60)
    print("Starting OCR Configuration UI")
    print("="*60)
    print("Open your browser to: http://localhost:8000")
    if KEYBOARD_AVAILABLE:
        print("Press Ctrl+Shift+Alt+Z anywhere to capture screenshot")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
            except:
                pass

