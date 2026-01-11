"""
Screenshot OCR Application
Captures screenshots on Ctrl+Shift+Alt+Z and extracts text using OpenAI Vision API
"""
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import keyboard
import pygetwindow as gw
from openai import OpenAI
from PIL import ImageGrab

# Configuration file path
CONFIG_FILE = Path("config.json")


def load_config():
    """Load configuration from config.json or create default template"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Validate required fields
                if not all(key in config for key in ["api_url", "api_key", "model"]):
                    print("Error: config.json is missing required fields (api_url, api_key, model)")
                    sys.exit(1)
                return config
        except json.JSONDecodeError:
            print("Error: config.json is not valid JSON")
            sys.exit(1)
    else:
        # Create default config template
        default_config = {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model": "gpt-4-vision-preview"
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"Created {CONFIG_FILE} with default LM Studio configuration.")
        print("Please edit it with your API URL, key, and model.")
        sys.exit(0)


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


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_text_with_vision_api(image, config):
    """Extract text from image using OpenAI Vision API"""
    try:
        # Initialize OpenAI client with custom base URL
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["api_url"]
        )
        
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        # Call Vision API
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the extracted text, no additional commentary."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        extracted_text = response.choices[0].message.content
        return extracted_text
        
    except Exception as e:
        print(f"Error calling Vision API: {e}")
        return None


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
    
    print("Extracting text using Vision API...")
    
    # Extract text using Vision API
    extracted_text = extract_text_with_vision_api(screenshot, config)
    
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
    
    # Load and validate config
    config = load_config()
    
    print(f"API URL: {config['api_url']}")
    print(f"Model: {config['model']}")
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
