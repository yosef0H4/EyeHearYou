"""Main entry point for OCR Configuration UI"""
import uvicorn

from .app import app
from .hotkey import start_hotkey_thread, cleanup_hotkey, KEYBOARD_AVAILABLE


def main():
    """Main function to start the server"""
    # Setup hotkey in background thread
    start_hotkey_thread()
    
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
        cleanup_hotkey()


if __name__ == "__main__":
    main()

