"""Worker thread for OCR operations"""
from PyQt6.QtCore import QThread, pyqtSignal

from src.backend.core.config import load_config
from src.backend.core.detection import detect_text_regions_unfiltered
from src.backend.core.merging import merge_close_text_boxes
from src.backend.core.filtering import (
    filter_text_regions, 
    sort_text_regions_by_reading_order,
    generate_selection_mask,
    filter_regions_by_mask,
    get_regions_from_mask,
    filter_contained_boxes
)
from src.backend.core.extraction import crop_text_regions
from src.backend.core.preprocessing import process_image
from src.backend.state import state
from src.backend.core.tts import speak_text, stop_tts_engine


class OCRWorker(QThread):
    """Worker thread for OCR operations to prevent UI freezing"""
    progress_signal = pyqtSignal(str, int)  # Message, Percent
    finished_signal = pyqtSignal(object, object, object, str)  # Raw Boxes, Filtered Boxes, Merged Boxes (with info), Text
    error_signal = pyqtSignal(str)
    image_processed_signal = pyqtSignal(object)  # Emits PIL Image after preprocessing

    def __init__(self, mode="full", config=None):
        super().__init__()
        self.mode = mode  # "detection_only" or "full"
        self.is_cancelled = False
        self.config = config or load_config()

    def run(self):
        try:
            image = state.last_image
            if not image:
                self.error_signal.emit("No image captured. Please capture a screenshot first.")
                return

            # 1. Preprocessing & Detection Phase
            td_config = self.config.get("text_detection", {})
            
            # Apply Preprocessing
            self.progress_signal.emit("Preprocessing image...", 5)
            processed_image = process_image(image, self.config)
            
            # Emit processed image for UI preview
            self.image_processed_signal.emit(processed_image)
            
            # Use processed image for detection and extraction
            image = processed_image
            
            # --- SELECTION & DETECTION LOGIC ---
            img_height, img_width = image.size[1], image.size[0]
            
            # 1. Generate Mask
            selection_mask = generate_selection_mask(
                (img_height, img_width), 
                state.selection_ops, 
                state.selection_base_state
            )

            # Convert manual boxes from normalized to pixels for this image size
            # Normalized coordinates (0-1) ensure boxes work across different resolutions/aspect ratios
            manual_pixel_boxes = []
            for nx, ny, nw, nh in state.manual_boxes:
                # Convert normalized to pixel coordinates
                x1 = int(nx * img_width)
                y1 = int(ny * img_height)
                x2 = int((nx + nw) * img_width)
                y2 = int((ny + nh) * img_height)
                
                # Clamp to image bounds (handle edge cases from different aspect ratios)
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                # Only add if valid box
                if x2 > x1 and y2 > y1:
                    manual_pixel_boxes.append((x1, y1, x2, y2))

            all_boxes = []
            regions = []

            if state.use_rapidocr:
                # AUTOMATIC MODE (RapidOCR + Mask Filtering)
                # Run detection (with smart caching - only runs RapidOCR if preprocessing changed)
                self.progress_signal.emit("Detecting text regions...", 10)
                if self.is_cancelled:
                    return
                
                raw_detections_with_scores = detect_text_regions_unfiltered(image, config=self.config, use_cache=True)
                
                if self.is_cancelled:
                    return
                
                if raw_detections_with_scores is None:
                    self.error_signal.emit("Failed to detect text regions. Is RapidOCR installed?")
                    return
                
                # Filter logic
                if raw_detections_with_scores:
                    # Extract raw boxes
                    all_boxes = [bbox for bbox, score in raw_detections_with_scores]
                    
                    # A. Filter by Selection Mask
                    self.progress_signal.emit("Filtering by selection...", 15)
                    masked_boxes = filter_regions_by_mask(all_boxes, selection_mask)
                    
                    # B. Apply Size Filter
                    self.progress_signal.emit("Applying size filter...", 20)
                    min_width_ratio = td_config.get("min_width_ratio", 0.0)
                    min_height_ratio = td_config.get("min_height_ratio", 0.0)
                    median_height_fraction = td_config.get("median_height_fraction", 1.0)
                    
                    regions = filter_text_regions(
                        masked_boxes,
                        (img_height, img_width),
                        min_width_ratio=min_width_ratio,
                        min_height_ratio=min_height_ratio,
                        median_height_fraction=median_height_fraction
                    )
                
                # Always include manual boxes (even when RapidOCR is on)
                # Manual boxes are user's explicit choice, so add them to results
                if manual_pixel_boxes:
                    # Add manual boxes to all_boxes for visualization
                    all_boxes.extend(manual_pixel_boxes)
                    # Add to regions (they bypass size filtering since user explicitly selected them)
                    regions.extend(manual_pixel_boxes)
            else:
                # MANUAL MODE (No RapidOCR)
                # Use manual boxes OR selection mask regions (whichever exists)
                self.progress_signal.emit("Processing manual selection...", 10)
                all_boxes = []
                regions = []
                
                # Priority 1: Manual boxes (user-drawn boxes)
                if manual_pixel_boxes:
                    all_boxes.extend(manual_pixel_boxes)
                    regions.extend(manual_pixel_boxes)
                
                # Priority 2: Selection mask regions (if user selected areas but no manual boxes)
                # Only use mask if user has made explicit selection operations
                # If no ops and base_state=True, mask would be all white (whole image) - skip that
                elif selection_mask is not None and state.selection_ops:
                    # User has made selection operations (add/sub), extract regions from mask
                    manual_regions = get_regions_from_mask(selection_mask)
                    all_boxes = manual_regions
                    regions = manual_regions
                elif selection_mask is not None and not state.selection_base_state:
                    # User deselected all (base_state=False) but no ops yet
                    # This means nothing is selected, so return empty
                    all_boxes = []
                    regions = []

            # Handle case where no regions found
            if not regions and state.use_rapidocr:
                # If RapidOCR found nothing, or selection filtered everything out
                # AND we are in full mode, maybe fallback? 
                # But if user deselected everything, we shouldn't fallback to full image.
                # If selection mask exists (not default), assume user Intent.
                # Default selection (Select All) -> Fallback OK.
                # Custom selection -> No Fallback.
                is_default_selection = (state.selection_base_state is True and not state.selection_ops)
                
                if is_default_selection and not all_boxes:
                    self.progress_signal.emit("No regions found, processing full image...", 20)
                    # Fallback
                    if self.mode == "full":
                        from src.backend.core.model_loader import get_model
                        model = get_model()
                        # Run inference on full image
                        text = model.predict(image)
                        self.finished_signal.emit(all_boxes, [], [], text or "No text extracted.")
                    else:
                        self.finished_signal.emit(all_boxes, [], [], None)
                    return
            
            # If manual mode and no regions, just stop
            if not regions and not state.use_rapidocr:
                if self.mode == "full":
                    self.finished_signal.emit([], [], [], "No area selected.")
                else:
                    self.finished_signal.emit([], [], [], None)
                return

            if self.is_cancelled:
                return

            # --- NEW STEP: Filter Contained Boxes ---
            # If we have mixed Manual and Auto boxes, or overlapping Manual boxes,
            # we want to keep the larger container box and discard the inner ones.
            if len(regions) > 1:
                # Use a high threshold (0.8) so we only remove boxes that are mostly inside others
                regions = filter_contained_boxes(regions, threshold=0.8)

            if self.is_cancelled:
                return

            # Apply size filter to manual boxes if needed (already done for RapidOCR)
            # Note: For manual boxes, we might want to skip size filtering since user explicitly selected them
            # But for selection mask regions, we should filter
            if not state.use_rapidocr and regions:
                # Only filter if these are from selection mask, not manual boxes
                # Manual boxes are user's explicit choice, so don't filter them
                if not manual_pixel_boxes:  # These are from selection mask
                    min_width_ratio = td_config.get("min_width_ratio", 0.0)
                    min_height_ratio = td_config.get("min_height_ratio", 0.0)
                    median_height_fraction = td_config.get("median_height_fraction", 1.0)
                    regions = filter_text_regions(
                        regions,
                        (img_height, img_width),
                        min_width_ratio=min_width_ratio,
                        min_height_ratio=min_height_ratio,
                        median_height_fraction=median_height_fraction
                    )
            
            if self.is_cancelled:
                return
            
            # Get sorting config (with backward compatibility)
            sort_config = self.config.get("text_sorting", {})
            if not sort_config:
                # Fallback to legacy reading_direction
                legacy_dir = self.config.get("reading_direction", "ltr")
                direction = "horizontal_ltr" if legacy_dir == "ltr" else "horizontal_rtl"
                group_tol = 0.8
            else:
                direction = sort_config.get("direction", "horizontal_ltr")
                group_tol = sort_config.get("group_tolerance", 0.5)
            
            # Step 3: Sort by reading order (Initial sort)
            regions = sort_text_regions_by_reading_order(
                regions, 
                direction=direction,
                group_tolerance=group_tol
            )
            
            if not regions:
                self.progress_signal.emit("No regions found, processing full image...", 20)
                # Fallback to full image extraction
                if self.mode == "full":
                    from src.backend.core.model_loader import get_model
                    model = get_model()
                    # Run inference on full image
                    text = model.predict(image)
                    self.finished_signal.emit(all_boxes, [], [], text or "No text extracted.")
                else:
                    self.finished_signal.emit(all_boxes, [], [], None)
                return

            self.progress_signal.emit(f"Filtered to {len(regions)} text region(s)...", 30)

            # 2. Merging Phase
            self.progress_signal.emit(f"Merging {len(regions)} regions...", 20)
            if self.is_cancelled:
                return

            # Use adaptive ratios only
            merge_vertical_ratio = td_config.get("merge_vertical_ratio", 0.07)
            merge_horizontal_ratio = td_config.get("merge_horizontal_ratio", 0.37)
            merge_width_ratio_threshold = td_config.get("merge_width_ratio_threshold", 0.75)
            
            merged_regions, is_merged, original_groups = merge_close_text_boxes(
                regions,
                vertical_ratio=merge_vertical_ratio,
                horizontal_ratio=merge_horizontal_ratio,
                width_ratio_threshold=merge_width_ratio_threshold
            )

            if self.is_cancelled:
                return

            self.progress_signal.emit(f"After merging close boxes: {len(merged_regions)} text region(s)", 25)

            # Format merged boxes with count and original boxes (before re-sorting)
            merged_boxes_info_unsorted = []
            for merged_box, is_merged_flag, orig_group in zip(merged_regions, is_merged, original_groups):
                if is_merged_flag and orig_group:
                    merged_boxes_info_unsorted.append({
                        "rect": merged_box,
                        "count": len(orig_group),
                        "originalBoxes": orig_group
                    })
                else:
                    merged_boxes_info_unsorted.append({
                        "rect": merged_box,
                        "count": 1,
                        "originalBoxes": [merged_box]
                    })
            
            # RE-SORT MERGED REGIONS
            # Merging might have combined boxes in a way that slightly shifts 
            # the effective "center", so we re-sort to ensure perfect reading order.
            merged_regions = sort_text_regions_by_reading_order(
                merged_regions,
                direction=direction,
                group_tolerance=group_tol
            )

            # Re-order merged_boxes_info to match the sorted merged_regions
            # Create a mapping from box tuple to box info
            box_to_info = {tuple(box_info["rect"]): box_info for box_info in merged_boxes_info_unsorted}
            merged_boxes_info = []
            for merged_box in merged_regions:
                box_key = tuple(merged_box)
                if box_key in box_to_info:
                    merged_boxes_info.append(box_to_info[box_key])
                else:
                    # Fallback if mapping fails
                    merged_boxes_info.append({
                        "rect": merged_box,
                        "count": 1,
                        "originalBoxes": [merged_box]
                    })

            if self.mode == "detection_only":
                self.progress_signal.emit("Detection Complete", 100)
                self.finished_signal.emit(all_boxes, regions, merged_boxes_info, None)
                return

            # 3. Extraction Phase (Full Mode)
            self.progress_signal.emit("Cropping regions...", 30)
            cropped_images = crop_text_regions(image, merged_regions)

            if not cropped_images:
                # Fallback to full image
                from src.backend.core.model_loader import get_model
                model = get_model()
                # Run inference on full image
                text = model.predict(image)
                if text:
                    speak_text(text, clear_queue=True)  # Speak immediately
                self.finished_signal.emit(all_boxes, regions, merged_boxes_info, text or "No text extracted.")
                return

            # Clear TTS queue before starting a new batch of regions
            stop_tts_engine()

            # 4. Process regions in BATCH (Faster)
            self.progress_signal.emit(f"Reading {len(cropped_images)} regions...", 40)
            
            from src.backend.core.model_loader import get_model
            model = get_model()
            
            # Run batch inference
            # This sends all images to the GPU at once (or sequentially without disk I/O overhead)
            texts = model.predict_batch(cropped_images)
            
            extracted_texts = []
            total = len(texts)
            
            # Process results
            for i, text in enumerate(texts):
                if self.is_cancelled:
                    break
                    
                if text and text.strip():
                    clean_text = text.strip()
                    extracted_texts.append(clean_text)
                    
                    # STREAMING: Speak immediately
                    speak_text(clean_text, clear_queue=False)
                
                # Update progress
                percent = 40 + int(((i + 1) / total) * 60)
                self.progress_signal.emit(f"Processed {i+1}/{total}...", percent)

            final_text = "\n".join(extracted_texts) if extracted_texts else "No text extracted."
            # Store extracted text in state for Play button
            state.last_extracted_text = final_text if final_text != "No text extracted." else ""
            self.progress_signal.emit("Finalizing...", 100)
            # We don't trigger speak_text(final_text) here because we already streamed it!
            self.finished_signal.emit(all_boxes, regions, merged_boxes_info, final_text)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Error: {str(e)}")

    def cancel(self):
        self.is_cancelled = True
        # Also stop TTS if we cancel OCR
        stop_tts_engine()

