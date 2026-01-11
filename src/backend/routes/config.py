"""Configuration routes"""
import json
import sys
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import main as ocr_app


class ConfigUpdate(BaseModel):
    """Configuration update model"""
    api_url: str
    api_key: str
    model: str
    max_image_dimension: int
    text_detection: Dict[str, Any]


async def get_config():
    """Get current configuration"""
    try:
        return ocr_app.load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def save_config(config: ConfigUpdate):
    """Save configuration to file"""
    try:
        config_dict = config.dict()
        with open(ocr_app.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        return {"status": "success", "message": "Config saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

