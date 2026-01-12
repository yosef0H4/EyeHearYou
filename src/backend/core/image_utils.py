"""Image utility functions"""
import base64
from io import BytesIO
from PIL import Image

# Optional imports for GPU checking
try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    paddle = None


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
    Check if GPU is available for PaddleOCR.
    Returns True if GPU is available, False otherwise.
    """
    if not PADDLE_AVAILABLE or paddle is None:
        return False
    
    try:
        # Check if PaddlePaddle was compiled with CUDA support
        if hasattr(paddle.device, 'is_compiled_with_cuda'):
            if not paddle.device.is_compiled_with_cuda():
                return False
            # Also check if CUDA devices are actually available
            try:
                # Method 1: Check get_device() which returns "gpu:0" or "cpu"
                if hasattr(paddle.device, 'get_device'):
                    device = paddle.device.get_device()
                    if device and ('gpu' in str(device).lower() or 'cuda' in str(device).lower()):
                        return True
                # Method 2: Check CUDA device count
                if hasattr(paddle.device, 'cuda') and hasattr(paddle.device.cuda, 'device_count'):
                    if paddle.device.cuda.device_count() > 0:
                        return True
            except Exception as e:
                # If we can't check device count, assume GPU is available if compiled with CUDA
                # This is a safe fallback since is_compiled_with_cuda() already passed
                return True
        # Fallback: try to check if CUDA is available (older PaddlePaddle versions < 2.5)
        # Note: paddle.fluid was removed in PaddlePaddle 3.0+, so this will fail silently
        try:
            import paddle.fluid as fluid
            if hasattr(fluid, 'is_compiled_with_cuda'):
                return fluid.is_compiled_with_cuda()
        except (ImportError, AttributeError):
            # paddle.fluid doesn't exist in PaddlePaddle 3.0+, ignore
            pass
        return False
    except Exception as e:
        # Log the exception for debugging but return False
        print(f"Warning: Error checking GPU availability: {e}")
        return False



