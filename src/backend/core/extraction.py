"""Text extraction using Local H2OVL Model"""
from .image_utils import resize_image_if_needed
from .detection import detect_text_regions
from .merging import merge_close_text_boxes
from .debug import save_debug_images
from .task_manager import task_manager
from .model_loader import get_model


def crop_text_regions(image, text_regions, padding=5):
    """
    Crop text regions from the image with optional padding.
    Returns a list of cropped PIL Images.
    """
    cropped_images = []
    img_width, img_height = image.size
    
    for x1, y1, x2, y2 in text_regions:
        # Add padding and ensure coordinates are within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Only crop if region is valid
        if x2 > x1 and y2 > y1:
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
    
    return cropped_images


def extract_text_with_local_model(image, config):
    """Extract text from a single image using local H2OVL model"""
    # Check for cancellation before starting
    if task_manager.is_cancelled():
        return None
    
    try:
        # Resize to prevent excessive VRAM usage if image is massive
        # H2OVL handles dynamic resolutions well, but a safety cap is good
        max_dimension = config.get("max_image_dimension", 1080)
        current_image = resize_image_if_needed(image, max_dimension=max_dimension)
        
        # Get Singleton Model
        model = get_model()
        
        # Run Inference
        extracted_text = model.predict(current_image)
        return extracted_text
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None


# Alias for backwards compatibility with existing code
extract_text_with_vision_api = extract_text_with_local_model


def extract_text_from_regions(full_image, config):
    """
    Extract text by first detecting text regions, then processing only those regions with the local model.
    This is more efficient than processing the full image.
    """
    # 1. Detection Phase
    task_manager.emit_status("Detecting text regions...", progress=10)
    if task_manager.is_cancelled():
        return None

    # Get text detection settings from config
    text_detection_config = config.get("text_detection", {})
    min_width = text_detection_config.get("min_width", 30)
    min_height = text_detection_config.get("min_height", 30)
    
    # Try to detect text regions first
    # Note: min_confidence is not used since RapidOCR doesn't provide confidence scores in detection-only mode
    text_regions = detect_text_regions(full_image, 
                                      min_width=min_width,
                                      min_height=min_height)
    
    if task_manager.is_cancelled():
        return None

    if text_regions is None or len(text_regions) == 0:
        # Fallback: if detection fails or no regions found, use full image
        task_manager.emit_status("No regions found, processing full image...", progress=20)
        print("No text regions detected or detection unavailable, using full image...")
        return extract_text_with_local_model(full_image, config)
    
    print(f"Detected {len(text_regions)} text region(s), processing individually...")
    task_manager.emit_status(f"Detected {len(text_regions)} text region(s), processing individually...", progress=15)
    
    # 2. Merging Phase
    task_manager.emit_status(f"Merging {len(text_regions)} regions...", progress=20)
    
    # Merge close text boxes (like split dialogue lines)
    merge_vertical_tolerance = text_detection_config.get("merge_vertical_tolerance", 30)
    merge_horizontal_tolerance = text_detection_config.get("merge_horizontal_tolerance", 50)
    merge_width_ratio_threshold = text_detection_config.get("merge_width_ratio_threshold", 0.3)
    text_regions, is_merged, _ = merge_close_text_boxes(
        text_regions,
        vertical_tolerance=merge_vertical_tolerance,
        horizontal_tolerance=merge_horizontal_tolerance,
        width_ratio_threshold=merge_width_ratio_threshold
    )
    print(f"After merging close boxes: {len(text_regions)} text region(s)")
    
    if task_manager.is_cancelled():
        return None

    # 3. Cropping Phase
    cropped_images = crop_text_regions(full_image, text_regions)
    if not cropped_images:
        print("No valid text regions to process, using full image...")
        return extract_text_with_local_model(full_image, config)
    
    # Save debug images (merged boxes in blue, original boxes in green)
    save_debug_images(full_image, text_regions, cropped_images, is_merged=is_merged)
    
    # 4. Extraction Phase (Sequential Inference)
    all_texts = []
    total_regions = len(cropped_images)
    
    # Ensure model is loaded before loop to avoid lag on first item
    task_manager.emit_status("Loading AI Model...", progress=25)
    model = get_model()

    for i, cropped_img in enumerate(cropped_images):
        if task_manager.is_cancelled():
            print("Extraction cancelled during region processing.")
            return None
            
        progress = 30 + ((i / total_regions) * 70)  # Map 0-100% of regions to 30-100% total progress
        task_manager.emit_status(f"Reading region {i+1}/{total_regions}...", progress=progress)
        print(f"Processing region {i+1}/{total_regions}...")
        
        # Direct model call
        text = model.predict(cropped_img)
        
        if text and text.strip():
            all_texts.append(text.strip())
    
    task_manager.emit_status("Finalizing...", progress=100)
    
    # Combine all extracted texts
    if all_texts:
        return "\n".join(all_texts)
    else:
        return None


