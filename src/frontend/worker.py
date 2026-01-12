"""Worker thread for OCR operations"""
from PyQt6.QtCore import QThread, pyqtSignal

from src.backend.core.config import load_config
from src.backend.core.detection import detect_text_regions_unfiltered
from src.backend.core.merging import merge_close_text_boxes
from src.backend.core.filtering import filter_text_regions, sort_text_regions_by_reading_order
from src.backend.core.extraction import crop_text_regions
from src.backend.core.preprocessing import process_image
from src.backend.state import state


class OCRWorker(QThread):
    """Worker thread for OCR operations to prevent UI freezing"""
    progress_signal = pyqtSignal(str, int)  # Message, Percent
    finished_signal = pyqtSignal(object, object, str)  # Filtered Boxes, Merged Boxes (with info), Text
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
            
            if not raw_detections_with_scores:
                # No detections at all
                if self.mode == "full":
                    from src.backend.core.extraction import extract_text_with_vision_api
                    text = extract_text_with_vision_api(image, self.config)
                    self.finished_signal.emit([], [], text or "No text extracted.")
                else:
                    self.finished_signal.emit([], [], None)
                return
            
            # Apply filters
            # Note: RapidOCR doesn't provide confidence scores in detection-only mode,
            # so we skip confidence filtering and only filter by size
            min_width = int(td_config.get("min_width", 30))
            min_height = int(td_config.get("min_height", 30))
            
            img_height, img_width = image.size[1], image.size[0]
            
            # Extract bounding boxes (ignore confidence scores since they're all 1.0)
            all_boxes = [bbox for bbox, score in raw_detections_with_scores]
            
            # Apply size filter
            self.progress_signal.emit("Applying size filter...", 20)
            size_filtered = filter_text_regions(
                all_boxes,
                (img_height, img_width),
                min_width=min_width,
                min_height=min_height
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
                group_tol = sort_config.get("group_tolerance", 0.8)
            
            # Step 3: Sort by reading order (Initial sort)
            regions = sort_text_regions_by_reading_order(
                size_filtered, 
                direction=direction,
                group_tolerance=group_tol
            )
            
            if not regions:
                self.progress_signal.emit("No regions found, processing full image...", 20)
                # Fallback to full image extraction
                if self.mode == "full":
                    from src.backend.core.extraction import extract_text_with_vision_api
                    text = extract_text_with_vision_api(image, self.config)
                    self.finished_signal.emit([], [], text or "No text extracted.")
                else:
                    self.finished_signal.emit([], [], None)
                return

            self.progress_signal.emit(f"Filtered to {len(regions)} text region(s)...", 30)

            # 2. Merging Phase
            self.progress_signal.emit(f"Merging {len(regions)} regions...", 20)
            if self.is_cancelled:
                return

            merged_regions, is_merged, original_groups = merge_close_text_boxes(
                regions,
                vertical_tolerance=td_config.get("merge_vertical_tolerance", 30),
                horizontal_tolerance=td_config.get("merge_horizontal_tolerance", 50),
                width_ratio_threshold=td_config.get("merge_width_ratio_threshold", 0.3)
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
                self.finished_signal.emit(regions, merged_boxes_info, None)
                return

            # 3. Extraction Phase (Full Mode)
            self.progress_signal.emit("Cropping regions...", 30)
            cropped_images = crop_text_regions(image, merged_regions)

            if not cropped_images:
                from src.backend.core.extraction import extract_text_with_vision_api
                text = extract_text_with_vision_api(image, self.config)
                self.finished_signal.emit(regions, merged_boxes_info, text or "No text extracted.")
                return

            # 4. Process each region
            extracted_texts = []
            total = len(cropped_images)

            for i, crop in enumerate(cropped_images):
                if self.is_cancelled:
                    self.progress_signal.emit("Cancelled", 0)
                    return

                # Update progress: 30-100% for extraction
                percent = 30 + int((i / total) * 70)
                self.progress_signal.emit(f"Processing region {i+1}/{total}...", percent)

                from src.backend.core.extraction import extract_text_with_vision_api
                text = extract_text_with_vision_api(crop, self.config)
                if text and text.strip():
                    extracted_texts.append(text.strip())

            final_text = "\n".join(extracted_texts) if extracted_texts else "No text extracted."
            self.progress_signal.emit("Finalizing...", 100)
            self.finished_signal.emit(regions, merged_boxes_info, final_text)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Error: {str(e)}")

    def cancel(self):
        self.is_cancelled = True

