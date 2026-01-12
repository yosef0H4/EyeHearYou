"""
Standalone CLI entry point for OCR application
Captures screenshots on Ctrl+Shift+Alt+Z and extracts text using local H2OVL model
Uses text detection to only process text regions, improving efficiency

This is the standalone CLI version (without the GUI).
For the GUI, use run_gui.py instead.
"""
import keyboard
import sys
from pathlib import Path

# Import from core modules
from .core import (
    load_config,
    CONFIG_FILE,
    capture_screenshot,
    extract_text_from_regions,
)


def process_screenshot():
    """Main function to capture screenshot and extract text"""
    print("\n" + "="*60)
    print("Capturing active window...")
    
    # Load config
    config = load_config()
    
    # Capture screenshot
    screenshot = capture_screenshot()
    if screenshot is None:
        return
    
    print("Detecting text regions...")
    
    # Extract text using local H2OVL model (with region detection)
    extracted_text = extract_text_from_regions(screenshot, config)
    
    if extracted_text:
        print("\n" + "-"*60)
        print("EXTRACTED TEXT:")
        print("-"*60)
        print(extracted_text)
        print("-"*60)
    else:
        print("Failed to extract text from screenshot.")


def main():
    """Main entry point"""
    print("Screenshot OCR Application")
    print("="*60)
    
    # Check if config exists, if not create template
    if not CONFIG_FILE.exists():
        load_config()
        return
    
    # Load config
    config = load_config()
    
    print("Model: H2OVL-Mississippi-0.8B (Local)")
    print("\nPress Ctrl+Shift+Alt+Z to capture screenshot and extract text")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    # Register hotkey
    try:
        keyboard.add_hotkey("ctrl+shift+alt+z", process_screenshot)
        
        # Keep the program running
        keyboard.wait()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


