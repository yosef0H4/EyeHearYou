"""Image utility functions"""
import base64
from io import BytesIO
from PIL import Image


def resize_image_if_needed(image, max_dimension=2048):
    """
    Resize image if it exceeds maximum dimension, maintaining aspect ratio.
    This prevents context window overflow when sending to vision APIs.
    
    Args:
        image: PIL Image to resize
        max_dimension: Maximum width or height in pixels (default: 2048)
    
    Returns:
        Resized PIL Image (or original if no resize needed)
    """
    width, height = image.size
    max_size = max(width, height)
    
    if max_size <= max_dimension:
        return image
    
    # Calculate scaling factor to fit within max_dimension
    scale = max_dimension / max_size
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Use high-quality resampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized image from {width}x{height} to {new_width}x{new_height} to fit context window")
    return resized


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def check_gpu_available():
    """
    Check if GPU is available.
    Note: RapidOCR uses ONNX Runtime which runs on CPU by default.
    This function is kept for compatibility but always returns False
    since RapidOCR doesn't use GPU acceleration.
    
    Returns:
        False (RapidOCR runs on CPU via ONNX Runtime)
    """
    # RapidOCR uses ONNX Runtime which runs on CPU
    # GPU support would require onnxruntime-gpu, but we're using CPU version
    return False



