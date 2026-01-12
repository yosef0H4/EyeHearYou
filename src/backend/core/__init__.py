"""Core OCR functionality modules"""
# Apply torchvision patch FIRST, before any imports that might trigger torchvision
from .torchvision_patch import apply_torchvision_patch
apply_torchvision_patch()

from .config import load_config, CONFIG_FILE
from .capture import capture_screenshot
from .image_utils import resize_image_if_needed, image_to_base64, check_gpu_available
from .filtering import filter_text_regions, sort_text_regions_by_reading_order
from .detection import detect_text_regions, detect_text_regions_unfiltered
from .merging import merge_close_text_boxes
from .extraction import extract_text_with_local_model, extract_text_with_vision_api, extract_text_from_regions, crop_text_regions
from .preprocessing import process_image
from .debug import save_debug_images
from .tts import speak_text

__all__ = [
    'load_config',
    'CONFIG_FILE',
    'capture_screenshot',
    'resize_image_if_needed',
    'image_to_base64',
    'check_gpu_available',
    'filter_text_regions',
    'sort_text_regions_by_reading_order',
    'detect_text_regions',
    'detect_text_regions_unfiltered',
    'merge_close_text_boxes',
    'extract_text_with_local_model',
    'extract_text_with_vision_api',  # Alias for backwards compatibility
    'extract_text_from_regions',
    'crop_text_regions',
    'process_image',
    'save_debug_images',
    'speak_text',
]



