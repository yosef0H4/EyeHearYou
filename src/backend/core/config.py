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
                # Merge in missing TTS config if it doesn't exist (for backward compatibility)
                if "tts" not in config:
                    config["tts"] = {
                        "voice": "af_heart",
                        "speed": 1.0
                    }
                    # Save updated config
                    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4)
                else:
                    # Ensure all TTS fields exist (remove "enabled" if present, it's no longer used)
                    tts_defaults = {"voice": "af_heart", "speed": 1.0}
                    updated = False
                    # Remove "enabled" if it exists (backward compatibility)
                    if "enabled" in config["tts"]:
                        del config["tts"]["enabled"]
                        updated = True
                    for key, default_value in tts_defaults.items():
                        if key not in config["tts"]:
                            config["tts"][key] = default_value
                            updated = True
                    if updated:
                        # Save updated config
                        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                            json.dump(config, f, indent=4)
                # Ensure adaptive parameters exist (remove legacy pixel-based ones)
                if "text_detection" in config:
                    td = config["text_detection"]
                    updated = False
                    
                    # Remove legacy pixel-based parameters
                    if "min_width" in td:
                        del td["min_width"]
                        updated = True
                    if "min_height" in td:
                        del td["min_height"]
                        updated = True
                    if "merge_vertical_tolerance" in td:
                        del td["merge_vertical_tolerance"]
                        updated = True
                    if "merge_horizontal_tolerance" in td:
                        del td["merge_horizontal_tolerance"]
                        updated = True
                    
                    # Add adaptive parameters if missing
                    if "min_height_ratio" not in td:
                        td["min_height_ratio"] = 0.01
                        updated = True
                    if "min_width_ratio" not in td:
                        td["min_width_ratio"] = 0.0
                        updated = True
                    if "median_height_fraction" not in td:
                        td["median_height_fraction"] = 0.4
                        updated = True
                    if "merge_vertical_ratio" not in td:
                        td["merge_vertical_ratio"] = 0.5
                        updated = True
                    if "merge_horizontal_ratio" not in td:
                        td["merge_horizontal_ratio"] = 1.5
                        updated = True
                    
                    if updated:
                        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                            json.dump(config, f, indent=4)
                
                return config
        except json.JSONDecodeError:
            print("Error: config.json is not valid JSON")
            sys.exit(1)
    else:
        # Create default config template (Local Model Version)
        default_config = {
            "max_image_dimension": 1080,
            "reading_direction": "ltr",  # Legacy: "ltr" for Left-to-Right, "rtl" for Right-to-Left (Manga)
            "preprocessing": {
                "binary_threshold": 0,    # 0-255, 0=disabled
                "invert": False,
                "dilation": 0,            # 0-5
                "contrast": 1.0,          # 0.5 - 3.0
                "brightness": 0           # -100 to 100
            },
            "tts": {
                "voice": "af_heart",      # Kokoro voice ID
                "speed": 1.0              # 0.5 - 2.0
            },
            "text_detection": {
                # Adaptive parameters (works across all screen sizes and font sizes)
                "min_height_ratio": 0.01,       # Box must be at least 1% of screen height
                "min_width_ratio": 0.0,          # Minimum width as fraction of screen width (0.0 = disabled)
                "median_height_fraction": 0.4,   # Discard if < 40% of median text size (removes noise)
                "merge_vertical_ratio": 0.5,     # Merge lines if gap < 0.5x text height
                "merge_horizontal_ratio": 1.5,  # Merge words if gap < 1.5x text height
                "merge_width_ratio_threshold": 0.3
            },
            "text_sorting": {
                "direction": "horizontal_ltr",  # Options: horizontal_ltr, horizontal_rtl, vertical_ltr, vertical_rtl
                "group_tolerance": 0.8          # Multiplier for line/column grouping (0.1-2.0)
            }
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"Created {CONFIG_FILE} with default settings.")
        return default_config



