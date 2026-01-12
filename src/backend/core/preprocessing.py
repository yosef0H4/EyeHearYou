"""Image preprocessing for OCR improvement"""
import numpy as np
import cv2
from PIL import Image


def process_image(pil_image, config):
    """
    Apply image processing pipeline based on config.
    Returns processed PIL Image.
    """
    settings = config.get("preprocessing", {})
    
    # Defaults
    binary_threshold = settings.get("binary_threshold", 0) # 0 = disabled
    invert = settings.get("invert", False)
    dilation = settings.get("dilation", 0) # 0 = disabled
    contrast = settings.get("contrast", 1.0) # 1.0 = normal
    brightness = settings.get("brightness", 0) # 0 = normal
    
    # If no settings are active, return original
    if binary_threshold == 0 and not invert and dilation == 0 and contrast == 1.0 and brightness == 0:
        return pil_image

    # Convert PIL to OpenCV (numpy)
    # PIL RGB -> OpenCV BGR
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Brightness and Contrast
    if contrast != 1.0 or brightness != 0:
        # clip results to 0-255
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # 2. Binarization (Thresholding)
    if binary_threshold > 0:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, img = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
        # Convert back to BGR for consistency (AI/Preview expects RGB/BGR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 3. Inversion
    if invert:
        img = cv2.bitwise_not(img)

    # 4. Dilation (Thicken text)
    if dilation > 0:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=dilation)

    # Convert back to PIL
    # OpenCV BGR -> PIL RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img)


