"""Configuration management"""
import json
import sys
from pathlib import Path

# Configuration file paths
CONFIG_FILE = Path("config.json")
PROFILES_FILE = Path("profiles.json")

# Factory Default Profile (embedded in code, never written to disk)
FACTORY_DEFAULT = {
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
        "speed": 1.0,             # 0.5 - 2.0
        "volume": 1.0,            # 0.0 - 2.0 (0% - 200%)
        "lang_code": "a"          # 'a' for American, 'b' for British
    },
    "text_detection": {
        # Adaptive parameters (works across all screen sizes and font sizes)
        # Optimized defaults based on testing with various games
        "min_height_ratio": 0.031,      # Box must be at least 3.1% of screen height
        "min_width_ratio": 0.0,          # Minimum width as fraction of screen width (0.0 = disabled)
        "median_height_fraction": 1.0,   # Discard if < 100% of median text size (less aggressive noise filtering)
        "merge_vertical_ratio": 0.07,     # Merge lines if gap < 0.07x text height (tight vertical merging)
        "merge_horizontal_ratio": 0.37,  # Merge words if gap < 0.37x text height (tight horizontal merging)
        "merge_width_ratio_threshold": 0.75  # Minimum horizontal overlap for vertical merging
    },
    "text_sorting": {
        "direction": "horizontal_ltr",  # Options: horizontal_ltr, horizontal_rtl, vertical_ltr, vertical_rtl
        "group_tolerance": 0.5          # Multiplier for line/column grouping (0.1-2.0)
    },
    "manual_boxes": []
}


def load_profiles():
    """Load all profiles and the active profile pointer"""
    if not PROFILES_FILE.exists():
        return {"active": "Default", "profiles": {}}
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Validate structure
            if not isinstance(data, dict):
                return {"active": "Default", "profiles": {}}
            if "active" not in data:
                data["active"] = "Default"
            if "profiles" not in data:
                data["profiles"] = {}
            # Ensure active profile exists or reset to Default
            if data["active"] != "Default" and data["active"] not in data["profiles"]:
                data["active"] = "Default"
            return data
    except (json.JSONDecodeError, IOError, Exception) as e:
        print(f"Warning: Error loading profiles.json: {e}. Using Default profile.")
        return {"active": "Default", "profiles": {}}


def save_profiles(data):
    """Save the profiles dictionary to disk"""
    try:
        # Validate structure before saving
        if not isinstance(data, dict):
            raise ValueError("Profiles data must be a dictionary")
        if "active" not in data:
            data["active"] = "Default"
        if "profiles" not in data:
            data["profiles"] = {}
        # Never save "Default" as a profile
        if "Default" in data["profiles"]:
            del data["profiles"]["Default"]
        # If active is Default but not in profiles, that's fine
        if data["active"] == "Default":
            pass  # Default is embedded, not stored
        elif data["active"] not in data["profiles"]:
            # Active profile doesn't exist, reset to Default
            data["active"] = "Default"
        
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving profiles.json: {e}")


def load_config():
    """Load configuration for the active profile"""
    data = load_profiles()
    active_name = data.get("active", "Default")
    
    # Get config from active profile
    if active_name == "Default":
        config = FACTORY_DEFAULT.copy()
    else:
        # Load from user profiles
        config = data["profiles"].get(active_name, FACTORY_DEFAULT.copy()).copy()
    
    # Apply backward compatibility and validation
    config = _validate_and_migrate_config(config)
    
    return config


def _validate_and_migrate_config(config):
    """Validate and migrate config to ensure all required fields exist"""
    # Ensure TTS config exists
    if "tts" not in config:
        config["tts"] = FACTORY_DEFAULT["tts"].copy()
    else:
        # Remove "enabled" if present (backward compatibility)
        if "enabled" in config["tts"]:
            del config["tts"]["enabled"]
        # Ensure all TTS fields exist
        for key, default_value in FACTORY_DEFAULT["tts"].items():
            if key not in config["tts"]:
                config["tts"][key] = default_value
    
    # Ensure preprocessing config exists
    if "preprocessing" not in config:
        config["preprocessing"] = FACTORY_DEFAULT["preprocessing"].copy()
    else:
        for key, default_value in FACTORY_DEFAULT["preprocessing"].items():
            if key not in config["preprocessing"]:
                config["preprocessing"][key] = default_value
    
    # Ensure text_detection config exists and migrate legacy parameters
    if "text_detection" not in config:
        config["text_detection"] = FACTORY_DEFAULT["text_detection"].copy()
    else:
        td = config["text_detection"]
        # Remove legacy pixel-based parameters
        for legacy_key in ["min_width", "min_height", "merge_vertical_tolerance", "merge_horizontal_tolerance"]:
            if legacy_key in td:
                del td[legacy_key]
        # Add adaptive parameters if missing
        for key, default_value in FACTORY_DEFAULT["text_detection"].items():
            if key not in td:
                td[key] = default_value
    
    # Ensure text_sorting config exists
    if "text_sorting" not in config:
        config["text_sorting"] = FACTORY_DEFAULT["text_sorting"].copy()
    else:
        for key, default_value in FACTORY_DEFAULT["text_sorting"].items():
            if key not in config["text_sorting"]:
                config["text_sorting"][key] = default_value
    
    # Ensure other top-level fields exist
    if "max_image_dimension" not in config:
        config["max_image_dimension"] = FACTORY_DEFAULT["max_image_dimension"]
    if "reading_direction" not in config:
        config["reading_direction"] = FACTORY_DEFAULT["reading_direction"]
    if "manual_boxes" not in config:
        config["manual_boxes"] = FACTORY_DEFAULT["manual_boxes"].copy()
    
    return config



