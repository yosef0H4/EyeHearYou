"""
Standalone CLI entry point for OCR accessibility tool
Captures screenshots on Ctrl+Shift+Alt+Z, extracts text using local H2OVL model, and reads it aloud
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
    detect_text_regions,
)
from .core.model_loader import preload_model
from .core.tts import preload_tts, speak_text, stop_tts_engine
from .core.detection import preload_rapidocr


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
    
    # Clear old audio immediately upon new capture
    stop_tts_engine()
    
    # Define callback for streaming
    def on_text_stream(text_chunk):
        print(f"[Stream] {text_chunk[:30]}...")
        speak_text(text_chunk, clear_queue=False)
    
    # Extract text using local H2OVL model (with region detection and streaming)
    extracted_text = extract_text_from_regions(screenshot, config, on_text_found=on_text_stream)
    
    if extracted_text:
        print("\n" + "-"*60)
        print("EXTRACTED TEXT:")
        print("-"*60)
        print(extracted_text)
        print("-"*60)
        # No need to call speak_text(extracted_text) here, it was streamed!
    else:
        print("Failed to extract text from screenshot.")


def process_screenshot_detect():
    """Capture screenshot and run detection only (Ctrl+Shift+Alt+X)"""
    print("\n" + "="*60)
    print("Capturing active window (Detection Only)...")
    
    config = load_config()
    screenshot = capture_screenshot()
    if screenshot is None:
        return
    
    print("Detecting text regions...")
    td_config = config["text_detection"]
    regions = detect_text_regions(screenshot, 
                                min_width_ratio=td_config.get("min_width_ratio", 0.0),
                                min_height_ratio=td_config.get("min_height_ratio", 0.0),
                                median_height_fraction=td_config.get("median_height_fraction", 0.4))
    
    if regions:
        print(f"✓ Found {len(regions)} text regions.")
        print("For full extraction, press Ctrl+Shift+Alt+Z")
    else:
        print("No text regions detected.")


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
    print("\nPreloading models (this may take a moment on first run)...")
    
    # Preload and test RapidOCR
    preload_rapidocr(test=True)
    
    # Preload and test the OCR model
    preload_model(test=True)
    
    # Preload and test the TTS model
    preload_tts(test=True)
    
    print("\nHotkeys:")
    print("  Ctrl+Shift+Alt+Z : Capture + Extract Text")
    print("  Ctrl+Shift+Alt+X : Capture + Detect Only")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    # Register hotkey
    try:
        keyboard.add_hotkey("ctrl+shift+alt+z", process_screenshot)
        keyboard.add_hotkey("ctrl+shift+alt+x", process_screenshot_detect)
        
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


