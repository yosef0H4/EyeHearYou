"""Screenshot capture functionality"""
import pygetwindow as gw
import mss
from PIL import Image


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
        
        # Capture using MSS (faster than ImageGrab)
        with mss.mss() as sct:
            # MSS handles multi-monitor coordinates correctly
            monitor = {"top": top, "left": left, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            
            # Convert to PIL Image
            screenshot = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        return screenshot
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None






