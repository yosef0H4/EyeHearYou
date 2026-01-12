"""Configuration management"""
import json
import sys
from pathlib import Path

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
            "model": "gpt-4-vision-preview",
            "max_image_dimension": 1080,
            "reading_direction": "ltr",  # Legacy: "ltr" for Left-to-Right, "rtl" for Right-to-Left (Manga)
            "preprocessing": {
                "binary_threshold": 0,    # 0-255, 0=disabled
                "invert": False,
                "dilation": 0,            # 0-5
                "contrast": 1.0,          # 0.5 - 3.0
                "brightness": 0           # -100 to 100
            },
            "text_detection": {
                "min_width": 30,
                "min_height": 30,
                "merge_vertical_tolerance": 30,
                "merge_horizontal_tolerance": 50,
                "merge_width_ratio_threshold": 0.3
            },
            "text_sorting": {
                "direction": "horizontal_ltr",  # Options: horizontal_ltr, horizontal_rtl, vertical_ltr, vertical_rtl
                "group_tolerance": 0.8          # Multiplier for line/column grouping (0.1-2.0)
            }
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"Created {CONFIG_FILE} with default LM Studio configuration.")
        print("Please edit it with your API URL, key, and model.")
        sys.exit(0)



