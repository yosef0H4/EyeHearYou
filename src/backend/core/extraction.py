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


def extract_text_from_regions(full_image, config, on_text_found=None):
    """
    Extract text by first detecting text regions, then processing only those regions with the local model.
    This is more efficient than processing the full image.
    
    Args:
        full_image: The PIL Image
        config: Config dict
        on_text_found: Optional callback(text) that runs immediately when a region is processed.
                       Used for streaming text to TTS.
    """
    # 1. Detection Phase
    task_manager.emit_status("Detecting text regions...", progress=10)
    if task_manager.is_cancelled():
        return None

    # Get text detection settings from config
    text_detection_config = config.get("text_detection", {})
    # Use adaptive parameters only
    min_width_ratio = text_detection_config.get("min_width_ratio", 0.0)
    min_height_ratio = text_detection_config.get("min_height_ratio", 0.0)
    median_height_fraction = text_detection_config.get("median_height_fraction", 1.0)
    
    # Try to detect text regions first
    # Note: min_confidence is not used since RapidOCR doesn't provide confidence scores in detection-only mode
    text_regions = detect_text_regions(full_image, 
                                      min_width_ratio=min_width_ratio,
                                      min_height_ratio=min_height_ratio,
                                      median_height_fraction=median_height_fraction)
    
    if task_manager.is_cancelled():
        return None

    if text_regions is None or len(text_regions) == 0:
        # Fallback: if detection fails or no regions found, use full image
        task_manager.emit_status("No regions found, processing full image...", progress=20)
        print("No text regions detected or detection unavailable, using full image...")
        text = extract_text_with_local_model(full_image, config)
        if text and on_text_found:
            on_text_found(text)
        return text
    
    print(f"Detected {len(text_regions)} text region(s), processing individually...")
    task_manager.emit_status(f"Detected {len(text_regions)} text region(s), processing individually...", progress=15)
    
    # 2. Merging Phase
    task_manager.emit_status(f"Merging {len(text_regions)} regions...", progress=20)
    
    # Merge close text boxes (like split dialogue lines)
    # Use adaptive ratios only
    merge_vertical_ratio = text_detection_config.get("merge_vertical_ratio", 0.07)
    merge_horizontal_ratio = text_detection_config.get("merge_horizontal_ratio", 0.37)
    merge_width_ratio_threshold = text_detection_config.get("merge_width_ratio_threshold", 0.75)
    
    text_regions, is_merged, _ = merge_close_text_boxes(
        text_regions,
        vertical_ratio=merge_vertical_ratio,
        horizontal_ratio=merge_horizontal_ratio,
        width_ratio_threshold=merge_width_ratio_threshold
    )
    print(f"After merging close boxes: {len(text_regions)} text region(s)")
    
    if task_manager.is_cancelled():
        return None

    # 3. Cropping Phase
    cropped_images = crop_text_regions(full_image, text_regions)
    if not cropped_images:
        print("No valid text regions to process, using full image...")
        text = extract_text_with_local_model(full_image, config)
        if text and on_text_found:
            on_text_found(text)
        return text
    
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
            clean_text = text.strip()
            all_texts.append(clean_text)
            
            # --- CRITICAL CHANGE: Stream result immediately ---
            if on_text_found:
                on_text_found(clean_text)
            # --------------------------------------------------
    
    task_manager.emit_status("Finalizing...", progress=100)
    
    # Combine all extracted texts
    if all_texts:
        return "\n".join(all_texts)
    else:
        return None


