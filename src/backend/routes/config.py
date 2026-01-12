"""Configuration routes"""
import json
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from ..core.config import load_config, CONFIG_FILE


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
        return load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def save_config(config: ConfigUpdate):
    """Save configuration to file"""
    try:
        config_dict = config.dict()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        return {"status": "success", "message": "Config saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

