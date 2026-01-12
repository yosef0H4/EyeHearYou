"""Screenshot capture functionality"""
import pygetwindow as gw
from PIL import ImageGrab


def capture_screenshot():
    """Capture the active window"""
    try:
        # Get the active window
        active_window = gw.getActiveWindow()
        if active_window is None:
            print("Error: No active window found")
            return None
        
        # Get window position and size
        left = active_window.left
        top = active_window.top
        width = active_window.width
        height = active_window.height
        
        # Capture only the active window region
        bbox = (left, top, left + width, top + height)
        screenshot = ImageGrab.grab(bbox=bbox)
        
        return screenshot
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None



