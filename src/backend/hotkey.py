"""Keyboard hotkey management"""
import threading

from .core.capture import capture_screenshot
from .state import state


# Import keyboard for hotkey support
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None


def capture_and_update_state():
    """Capture screenshot and update global state (called by hotkey)"""
    try:
        screenshot = capture_screenshot()
        if screenshot:
            state.last_image = screenshot
            state.reset_detections()
            state.screenshot_version += 1
            
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

