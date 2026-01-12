"""Text extraction using Vision API"""
from openai import OpenAI
from .image_utils import resize_image_if_needed, image_to_base64
from .detection import detect_text_regions
from .merging import merge_close_text_boxes
from .debug import save_debug_images


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


def extract_text_with_vision_api(image, config):
    """Extract text from a single image using OpenAI Vision API"""
    # Initialize OpenAI client with custom base URL
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["api_url"]
    )
    
    # Start with configured max dimension, but reduce if we hit context window errors
    max_dimension = config.get("max_image_dimension", 1080)
    original_image = image
    retry_count = 0
    max_retries = 3
    
    while retry_count <= max_retries:
        try:
            # Resize image if too large to prevent context window overflow
            current_image = resize_image_if_needed(original_image, max_dimension=max_dimension)
            
            # Convert image to base64
            base64_image = image_to_base64(current_image)
            
            # Call Vision API
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the extracted text, no additional commentary."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            extracted_text = response.choices[0].message.content
            return extracted_text
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a context window error
            if "context length" in error_str.lower() or "context overflow" in error_str.lower():
                retry_count += 1
                if retry_count <= max_retries:
                    # Reduce max dimension by 25% and retry
                    max_dimension = int(max_dimension * 0.75)
                    print(f"Context window overflow detected. Retrying with smaller image size ({max_dimension}px max dimension)...")
                    continue
                else:
                    print(f"Error: Failed after {max_retries} retries with reduced image sizes. Original error: {e}")
                    return None
            else:
                # For other errors, don't retry
                print(f"Error calling Vision API: {e}")
                return None
    
    return None


def extract_text_from_regions(full_image, config):
    """
    Extract text by first detecting text regions, then sending only those regions to the API.
    This is more cost-effective than sending the full image.
    """
    # Get text detection settings from config
    text_detection_config = config.get("text_detection", {})
    min_confidence = text_detection_config.get("min_confidence", 0.6)
    min_width = text_detection_config.get("min_width", 30)
    min_height = text_detection_config.get("min_height", 30)
    
    # Try to detect text regions first
    text_regions = detect_text_regions(full_image, 
                                      min_confidence=min_confidence,
                                      min_width=min_width,
                                      min_height=min_height)
    
    if text_regions is None or len(text_regions) == 0:
        # Fallback: if detection fails or no regions found, use full image
        print("No text regions detected or detection unavailable, using full image...")
        return extract_text_with_vision_api(full_image, config)
    
    print(f"Detected {len(text_regions)} text region(s), processing individually...")
    
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
    
    # Crop text regions
    cropped_images = crop_text_regions(full_image, text_regions)
    
    if not cropped_images:
        print("No valid text regions to process, using full image...")
        return extract_text_with_vision_api(full_image, config)
    
    # Save debug images (merged boxes in blue, original boxes in green)
    save_debug_images(full_image, text_regions, cropped_images, is_merged=is_merged)
    
    # Process each cropped region
    all_texts = []
    for i, cropped_img in enumerate(cropped_images):
        print(f"Processing region {i+1}/{len(cropped_images)}...")
        text = extract_text_with_vision_api(cropped_img, config)
        if text and text.strip():
            all_texts.append(text.strip())
    
    # Combine all extracted texts
    if all_texts:
        return "\n".join(all_texts)
    else:
        return None


