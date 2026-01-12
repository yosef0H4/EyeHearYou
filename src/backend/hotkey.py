"""Keyboard hotkey management"""
import threading

from .core.capture import capture_screenshot
from .core.config import load_config
from .core.extraction import extract_text_from_regions
from .core.task_manager import task_manager
from .core.tts import speak_text, stop_tts_engine
from .state import state


# Import keyboard for hotkey support
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None


def capture_and_update_state():
    """Capture screenshot, run detection+OCR, and read aloud (called by hotkey)"""
    # 1. Cancel any existing tasks immediately
    task_manager.start_task("Hotkey Capture")
    
    try:
        task_manager.emit_status("Capturing Screenshot...", progress=5)
        screenshot = capture_screenshot()
        
        if screenshot:
            if task_manager.is_cancelled():
                return False

            state.last_image = screenshot
            state.reset_detections()
            state.screenshot_version += 1
            
            # Clear old audio immediately upon new capture
            stop_tts_engine()
            
            # Notify SSE listeners about the new image
            state.screenshot_queue.put({
                "type": "screenshot",
                "version": state.screenshot_version,
                "width": screenshot.width,
                "height": screenshot.height
            })
            
            print(f"[Hotkey] Screenshot captured: {screenshot.width}x{screenshot.height} (version {state.screenshot_version})")
            
            # Run full pipeline: detection + extraction
            try:
                print("[Hotkey] Running detection and OCR...")
                config = load_config()
                
                # Define callback for streaming
                def on_text_stream(text_chunk):
                    print(f"[Hotkey] Streamed: {text_chunk[:30]}...")
                    speak_text(text_chunk, clear_queue=False)

                # Run extraction with callback
                extracted_text = extract_text_from_regions(
                    screenshot, 
                    config, 
                    on_text_found=on_text_stream
                )
                
                if task_manager.is_cancelled():
                    print("[Hotkey] Task cancelled.")
                    task_manager.emit_status("Cancelled", is_loading=False)
                    return False

                if extracted_text:
                    # Store result in state for UI
                    state.last_extracted_text = extracted_text
                    state.last_extraction_version = state.screenshot_version
                    # No need to call speak_text(extracted_text) here, it was streamed!
                    msg = f"✅ Text extracted ({len(extracted_text)} chars) and sent to TTS"
                    print(f"[Hotkey] {msg}")
                    task_manager.emit_status(f"Reading {len(extracted_text)} chars...", is_loading=False, progress=100)
                    
                    # Also print to console for visibility
                    print("\n" + "="*60)
                    print("EXTRACTED TEXT:")
                    print("="*60)
                    print(extracted_text)
                    print("="*60 + "\n")
                else:
                    print("[Hotkey] ⚠️  No text extracted from screenshot")
                    task_manager.emit_status("No text found", is_loading=False, progress=100)
                    state.last_extracted_text = None
            except Exception as e:
                print(f"[Hotkey] ⚠️  Error during detection/OCR: {e}")
                import traceback
                traceback.print_exc()
                task_manager.emit_status(f"Error: {str(e)[:30]}...", is_loading=False)
                # Still return True since screenshot was captured successfully
            
            task_manager.finish_task()
            return True
        else:
            task_manager.emit_status("Screenshot failed", is_loading=False)
            task_manager.finish_task()
            return False
    except Exception as e:
        print(f"[Hotkey] Error capturing screenshot: {e}")
        task_manager.emit_status("Error capturing", is_loading=False)
        task_manager.finish_task()
        return False


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


def start_hotkey_thread():
    """Start hotkey listener in background thread"""
    if KEYBOARD_AVAILABLE:
        hotkey_thread = threading.Thread(target=setup_hotkey, daemon=True)
        hotkey_thread.start()
        return True
    return False


def cleanup_hotkey():
    """Cleanup hotkey on shutdown"""
    if KEYBOARD_AVAILABLE:
        try:
            keyboard.unhook_all()
        except:
            pass

