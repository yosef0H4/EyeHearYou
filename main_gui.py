"""PyQt6 GUI for Visual Novel OCR"""
import sys
import json
import threading
from pathlib import Path
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGroupBox, QPushButton, QTextEdit, QProgressBar,
    QGraphicsView, QGraphicsScene, QSplitter, QLineEdit, QSpinBox,
    QGraphicsTextItem, QGraphicsPixmapItem, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont

# Import existing core logic
from src.backend.core.config import load_config, CONFIG_FILE
from src.backend.core.capture import capture_screenshot
from src.backend.core.detection import detect_text_regions, detect_text_regions_unfiltered
from src.backend.core.merging import merge_close_text_boxes
from src.backend.core.filtering import filter_text_regions, sort_text_regions_by_reading_order
from src.backend.core.extraction import extract_text_from_regions, crop_text_regions
from src.backend.core.image_utils import image_to_base64
from src.backend.state import state

# Try to import keyboard for hotkeys
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

# Try to import pyperclip for clipboard
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    pyperclip = None


class PaddleVizWidget(QWidget):
    """Visualizer widget for Paddle settings showing confidence and size"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        self.setMaximumSize(120, 120)
        self.confidence = 0.6
        self.min_width = 30
        self.min_height = 30
        
    def update_values(self, confidence: float, min_width: int, min_height: int):
        """Update visualization values"""
        self.confidence = confidence
        self.min_width = min_width
        self.min_height = min_height
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Calculate box size (scaled to fit 20-100px range)
        box_w = max(20, min(100, self.min_width))
        box_h = max(20, min(100, self.min_height))
        
        # Center the box
        x = (self.width() - box_w) / 2
        y = (self.height() - box_h) / 2
        
        # Draw box with opacity based on confidence
        opacity = int(255 * self.confidence)
        pen = QPen(QColor(255, 0, 0), 2)
        brush = QBrush(QColor(255, 0, 0, opacity // 3))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(int(x), int(y), int(box_w), int(box_h))
        
        # Draw "A" text in center
        painter.setPen(QColor(255, 0, 0))
        font = QFont()
        font.setBold(True)
        font.setPointSize(24)
        painter.setFont(font)
        painter.drawText(QRect(int(x), int(y), int(box_w), int(box_h)), 
                         Qt.AlignmentFlag.AlignCenter, "A")


class MergeVizWidget(QWidget):
    """Visualizer widget for Merge settings showing tolerance zones"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        self.setMaximumSize(120, 120)
        self.v_tol = 30
        self.h_tol = 50
        
    def update_values(self, v_tol: int, h_tol: int):
        """Update visualization values"""
        self.v_tol = v_tol
        self.h_tol = h_tol
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Base box size (50x25)
        base_w = 50
        base_h = 25
        
        # Scale factor for tolerance visualization
        scale = 0.4
        
        # Calculate tolerance zone size
        zone_w = base_w + (self.h_tol * 2 * scale)
        zone_h = base_h + (self.v_tol * 2 * scale)
        
        # Center position
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Draw tolerance zone (yellow, dashed)
        zone_x = center_x - zone_w / 2
        zone_y = center_y - zone_h / 2
        pen = QPen(QColor(255, 255, 0), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        brush = QBrush(QColor(255, 255, 0, 25))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(int(zone_x), int(zone_y), int(zone_w), int(zone_h))
        
        # Draw center box (yellow, solid)
        box_x = center_x - base_w / 2
        box_y = center_y - base_h / 2
        pen = QPen(QColor(255, 255, 0), 1)
        pen.setStyle(Qt.PenStyle.SolidLine)
        brush = QBrush(QColor(255, 255, 0, 75))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(int(box_x), int(box_y), base_w, base_h)


class OCRWorker(QThread):
    """Worker thread for OCR operations to prevent UI freezing"""
    progress_signal = pyqtSignal(str, int)  # Message, Percent
    finished_signal = pyqtSignal(object, object, str)  # Filtered Boxes, Merged Boxes (with info), Text
    error_signal = pyqtSignal(str)

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

            # 1. Detection Phase (with caching like server)
            td_config = self.config.get("text_detection", {})
            cached_key = state.screenshot_version
            raw_detections_with_scores = None
            
            # Check if we have cached detections
            if cached_key in state.cached_raw_detections:
                raw_detections_with_scores = state.cached_raw_detections[cached_key]
                self.progress_signal.emit("Using cached detections...", 5)
            else:
                # Run new detection
                self.progress_signal.emit("Detecting text regions...", 10)
                if self.is_cancelled:
                    return
                
                raw_detections_with_scores = detect_text_regions_unfiltered(
                    image,
                    min_confidence=0.0  # Get all detections, we'll filter by confidence later
                )
                
                if self.is_cancelled:
                    return
                
                if raw_detections_with_scores is None:
                    self.error_signal.emit("Failed to detect text regions. Is PaddleOCR installed?")
                    return
                
                # Cache the raw detections
                if raw_detections_with_scores:
                    state.cached_raw_detections[cached_key] = raw_detections_with_scores
                    self.progress_signal.emit(f"Cached {len(raw_detections_with_scores)} raw detections", 15)
            
            if not raw_detections_with_scores:
                # No detections at all
                if self.mode == "full":
                    from src.backend.core.extraction import extract_text_with_vision_api
                    text = extract_text_with_vision_api(image, self.config)
                    self.finished_signal.emit([], [], text or "No text extracted.")
                else:
                    self.finished_signal.emit([], [], None)
                return
            
            # Apply filters (like server does)
            min_confidence = float(td_config.get("min_confidence", 0.6))
            min_width = int(td_config.get("min_width", 30))
            min_height = int(td_config.get("min_height", 30))
            
            img_height, img_width = image.size[1], image.size[0]
            
            # Step 1: Apply confidence filter
            self.progress_signal.emit("Applying confidence filter...", 20)
            confidence_filtered = [
                bbox for bbox, score in raw_detections_with_scores
                if score >= min_confidence
            ]
            
            if self.is_cancelled:
                return
            
            # Step 2: Apply size filter
            self.progress_signal.emit("Applying size filter...", 25)
            size_filtered = filter_text_regions(
                confidence_filtered,
                (img_height, img_width),
                min_width=min_width,
                min_height=min_height
            )
            
            if self.is_cancelled:
                return
            
            # Step 3: Sort by reading order
            regions = sort_text_regions_by_reading_order(size_filtered, direction='hor_ltr')
            
            if not regions:
                self.progress_signal.emit("No regions found, processing full image...", 20)
                # Fallback to full image extraction
                if self.mode == "full":
                    from src.backend.core.extraction import extract_text_with_vision_api
                    text = extract_text_with_vision_api(image, self.config)
                    self.finished_signal.emit([], text or "No text extracted.")
                else:
                    self.finished_signal.emit([], None)
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

            # Format merged boxes with count and original boxes (like server does)
            merged_boxes_info = []
            for i, (merged_box, is_merged_flag, orig_group) in enumerate(zip(merged_regions, is_merged, original_groups)):
                if is_merged_flag and orig_group:
                    merged_boxes_info.append({
                        "rect": merged_box,
                        "count": len(orig_group),
                        "originalBoxes": orig_group
                    })
                else:
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


class OCRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Novel OCR - Settings Tuner")
        self.resize(1400, 900)
        self.config = load_config()
        self.worker: Optional[OCRWorker] = None
        self.filtered_boxes: List[Tuple[int, int, int, int]] = []
        self.merged_boxes: List[dict] = []  # List of dicts with 'rect', 'count', 'originalBoxes'
        self.pixmap_item = None  # Store reference to pixmap item

        # Setup hotkey listener
        self.setup_hotkey_listener()

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- LEFT PANEL: Controls (Scrollable) ---
        # Create scroll area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(350)
        scroll_area.setMaximumWidth(450)
        
        # Create the controls widget
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # API Settings Group
        self.create_api_group(controls_layout)

        # Detection Settings Group
        self.create_detection_group(controls_layout)

        # Merge Settings Group
        self.create_merge_group(controls_layout)

        # Buttons
        btn_layout = QVBoxLayout()
        self.btn_capture = QPushButton("📸 Capture Screenshot")
        self.btn_capture.clicked.connect(self.run_capture_and_detect)
        self.btn_cancel = QPushButton("⛔ Cancel")
        self.btn_cancel.clicked.connect(self.cancel_process)
        self.btn_cancel.setEnabled(False)
        self.btn_extract = QPushButton("🚀 Extract Text")
        self.btn_extract.clicked.connect(self.run_extraction)

        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_extract)
        controls_layout.addLayout(btn_layout)

        # Status and Progress
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)
        controls_layout.addWidget(status_group)

        # Output Text
        output_group = QGroupBox("Extracted Text")
        output_layout = QVBoxLayout()
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMinimumHeight(150)
        output_layout.addWidget(self.text_output)
        output_group.setLayout(output_layout)
        controls_layout.addWidget(output_group)

        controls_layout.addStretch()
        
        # Set the controls widget as the scroll area's widget
        scroll_area.setWidget(controls_widget)

        # --- RIGHT PANEL: Image Preview ---
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel("Image Preview")
        preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        preview_layout.addWidget(preview_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # Connect view resize to update fit
        self.view.viewport().installEventFilter(self)
        preview_layout.addWidget(self.view)

        # Add to splitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(preview_widget)
        splitter.setSizes([400, 1000])

        # Apply dark theme
        self.apply_dark_theme()

    def create_api_group(self, layout):
        group = QGroupBox("API Settings")
        g_layout = QVBoxLayout()

        # API URL
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("API URL:"))
        self.api_url_input = QLineEdit()
        self.api_url_input.setText(self.config.get("api_url", "http://localhost:1234/v1"))
        self.api_url_input.editingFinished.connect(self.update_api_config)
        url_layout.addWidget(self.api_url_input)
        g_layout.addLayout(url_layout)

        # API Key
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(self.config.get("api_key", "lm-studio"))
        self.api_key_input.editingFinished.connect(self.update_api_config)
        key_layout.addWidget(self.api_key_input)
        g_layout.addLayout(key_layout)

        # Model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_input = QLineEdit()
        self.model_input.setText(self.config.get("model", "gpt-4-vision-preview"))
        self.model_input.editingFinished.connect(self.update_api_config)
        model_layout.addWidget(self.model_input)
        g_layout.addLayout(model_layout)

        group.setLayout(g_layout)
        layout.addWidget(group)

    def create_detection_group(self, layout):
        group = QGroupBox("Detection Settings (PaddleOCR)")
        g_layout = QVBoxLayout()

        td_config = self.config.get("text_detection", {})

        # Mini Visualizer
        viz_container = QWidget()
        viz_container.setMinimumHeight(120)
        viz_container.setMaximumHeight(120)
        viz_container.setStyleSheet("background: #000; border: 1px dashed #555; border-radius: 4px;")
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        self.paddle_viz = PaddleVizWidget()
        viz_layout.addWidget(self.paddle_viz, alignment=Qt.AlignmentFlag.AlignCenter)
        g_layout.addWidget(viz_container)

        # Min Confidence
        conf_layout = QHBoxLayout()
        conf_label_text = QLabel("Min Confidence:")
        self.conf_label = QLabel(f"{td_config.get('min_confidence', 0.6):.2f}")
        self.conf_label.setMinimumWidth(50)
        self.conf_label.setStyleSheet("color: #aaa; font-weight: 600;")
        conf_layout.addWidget(conf_label_text)
        conf_slider = QSlider(Qt.Orientation.Horizontal)
        conf_slider.setMinimum(10)
        conf_slider.setMaximum(100)
        conf_slider.setValue(int(td_config.get("min_confidence", 0.6) * 100))
        conf_slider.valueChanged.connect(
            lambda v: self.on_confidence_change(v / 100.0)
        )
        conf_slider.sliderReleased.connect(self.save_and_refresh)
        conf_layout.addWidget(conf_slider)
        conf_layout.addWidget(self.conf_label)
        g_layout.addLayout(conf_layout)
        info_conf = QLabel("Controls detection sensitivity. Lower = more detections (may include noise)")
        info_conf.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        info_conf.setWordWrap(True)
        g_layout.addWidget(info_conf)

        # Min Width
        width_layout = QHBoxLayout()
        width_label_text = QLabel("Min Width (px):")
        self.width_label = QLabel(str(td_config.get("min_width", 30)))
        self.width_label.setMinimumWidth(50)
        self.width_label.setStyleSheet("color: #aaa; font-weight: 600;")
        width_slider = QSlider(Qt.Orientation.Horizontal)
        width_slider.setMinimum(5)
        width_slider.setMaximum(300)
        width_slider.setValue(td_config.get("min_width", 30))
        width_spin = QSpinBox()
        width_spin.setMinimum(5)
        width_spin.setMaximum(1000)  # Allow higher values than slider
        width_spin.setValue(td_config.get("min_width", 30))
        width_spin.setMaximumWidth(80)
        
        def on_width_slider_change(v):
            self.on_slider_change("min_width", v, self.width_label, "{:.0f}")
            width_spin.blockSignals(True)
            width_spin.setValue(v)
            width_spin.blockSignals(False)
            self.update_paddle_viz()
        
        def on_width_spin_change(v):
            clamped = max(5, min(300, v))
            width_slider.blockSignals(True)
            width_slider.setValue(clamped)
            width_slider.blockSignals(False)
            self.on_slider_change("min_width", v, self.width_label, "{:.0f}")
            self.update_paddle_viz()
        
        width_slider.valueChanged.connect(on_width_slider_change)
        width_slider.sliderReleased.connect(self.save_and_refresh)
        width_spin.valueChanged.connect(on_width_spin_change)
        width_spin.editingFinished.connect(self.save_and_refresh)
        
        width_layout.addWidget(width_label_text)
        width_layout.addWidget(width_slider)
        width_layout.addWidget(width_spin)
        width_layout.addWidget(self.width_label)
        g_layout.addLayout(width_layout)

        # Min Height
        height_layout = QHBoxLayout()
        height_label_text = QLabel("Min Height (px):")
        self.height_label = QLabel(str(td_config.get("min_height", 30)))
        self.height_label.setMinimumWidth(50)
        self.height_label.setStyleSheet("color: #aaa; font-weight: 600;")
        height_slider = QSlider(Qt.Orientation.Horizontal)
        height_slider.setMinimum(5)
        height_slider.setMaximum(300)
        height_slider.setValue(td_config.get("min_height", 30))
        height_spin = QSpinBox()
        height_spin.setMinimum(5)
        height_spin.setMaximum(1000)
        height_spin.setValue(td_config.get("min_height", 30))
        height_spin.setMaximumWidth(80)
        
        def on_height_slider_change(v):
            self.on_slider_change("min_height", v, self.height_label, "{:.0f}")
            height_spin.blockSignals(True)
            height_spin.setValue(v)
            height_spin.blockSignals(False)
            self.update_paddle_viz()
        
        def on_height_spin_change(v):
            clamped = max(5, min(300, v))
            height_slider.blockSignals(True)
            height_slider.setValue(clamped)
            height_slider.blockSignals(False)
            self.on_slider_change("min_height", v, self.height_label, "{:.0f}")
            self.update_paddle_viz()
        
        height_slider.valueChanged.connect(on_height_slider_change)
        height_slider.sliderReleased.connect(self.save_and_refresh)
        height_spin.valueChanged.connect(on_height_spin_change)
        height_spin.editingFinished.connect(self.save_and_refresh)
        
        height_layout.addWidget(height_label_text)
        height_layout.addWidget(height_slider)
        height_layout.addWidget(height_spin)
        height_layout.addWidget(self.height_label)
        g_layout.addLayout(height_layout)

        # Store references for later updates
        self.conf_slider = conf_slider
        self.width_slider = width_slider
        self.height_slider = height_slider
        self.width_spin = width_spin
        self.height_spin = height_spin

        # Initialize visualizer
        self.update_paddle_viz()

        group.setLayout(g_layout)
        layout.addWidget(group)

    def create_merge_group(self, layout):
        group = QGroupBox("Merge Settings")
        g_layout = QVBoxLayout()

        td_config = self.config.get("text_detection", {})

        # Mini Visualizer
        viz_container = QWidget()
        viz_container.setMinimumHeight(120)
        viz_container.setMaximumHeight(120)
        viz_container.setStyleSheet("background: #000; border: 1px dashed #555; border-radius: 4px;")
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        self.merge_viz = MergeVizWidget()
        viz_layout.addWidget(self.merge_viz, alignment=Qt.AlignmentFlag.AlignCenter)
        g_layout.addWidget(viz_container)

        # Vertical Tolerance
        vtol_layout = QHBoxLayout()
        vtol_label_text = QLabel("Vertical Tolerance:")
        self.vtol_label = QLabel(str(td_config.get("merge_vertical_tolerance", 30)))
        self.vtol_label.setMinimumWidth(50)
        self.vtol_label.setStyleSheet("color: #aaa; font-weight: 600;")
        vtol_slider = QSlider(Qt.Orientation.Horizontal)
        vtol_slider.setMinimum(0)
        vtol_slider.setMaximum(300)
        vtol_slider.setValue(td_config.get("merge_vertical_tolerance", 30))
        vtol_spin = QSpinBox()
        vtol_spin.setMinimum(0)
        vtol_spin.setMaximum(1000)
        vtol_spin.setValue(td_config.get("merge_vertical_tolerance", 30))
        vtol_spin.setMaximumWidth(80)
        
        def on_vtol_slider_change(v):
            self.on_slider_change("merge_vertical_tolerance", v, self.vtol_label, "{:.0f}")
            vtol_spin.blockSignals(True)
            vtol_spin.setValue(v)
            vtol_spin.blockSignals(False)
            self.update_merge_viz()
        
        def on_vtol_spin_change(v):
            clamped = max(0, min(300, v))
            vtol_slider.blockSignals(True)
            vtol_slider.setValue(clamped)
            vtol_slider.blockSignals(False)
            self.on_slider_change("merge_vertical_tolerance", v, self.vtol_label, "{:.0f}")
            self.update_merge_viz()
        
        vtol_slider.valueChanged.connect(on_vtol_slider_change)
        vtol_slider.sliderReleased.connect(self.save_and_refresh)
        vtol_spin.valueChanged.connect(on_vtol_spin_change)
        vtol_spin.editingFinished.connect(self.save_and_refresh)
        
        vtol_layout.addWidget(vtol_label_text)
        vtol_layout.addWidget(vtol_slider)
        vtol_layout.addWidget(vtol_spin)
        vtol_layout.addWidget(self.vtol_label)
        g_layout.addLayout(vtol_layout)
        info_vtol = QLabel("Max vertical gap between boxes to merge (px)")
        info_vtol.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        info_vtol.setWordWrap(True)
        g_layout.addWidget(info_vtol)

        # Horizontal Tolerance
        htol_layout = QHBoxLayout()
        htol_label_text = QLabel("Horizontal Tolerance:")
        self.htol_label = QLabel(str(td_config.get("merge_horizontal_tolerance", 50)))
        self.htol_label.setMinimumWidth(50)
        self.htol_label.setStyleSheet("color: #aaa; font-weight: 600;")
        htol_slider = QSlider(Qt.Orientation.Horizontal)
        htol_slider.setMinimum(0)
        htol_slider.setMaximum(300)
        htol_slider.setValue(td_config.get("merge_horizontal_tolerance", 50))
        htol_spin = QSpinBox()
        htol_spin.setMinimum(0)
        htol_spin.setMaximum(1000)
        htol_spin.setValue(td_config.get("merge_horizontal_tolerance", 50))
        htol_spin.setMaximumWidth(80)
        
        def on_htol_slider_change(v):
            self.on_slider_change("merge_horizontal_tolerance", v, self.htol_label, "{:.0f}")
            htol_spin.blockSignals(True)
            htol_spin.setValue(v)
            htol_spin.blockSignals(False)
            self.update_merge_viz()
        
        def on_htol_spin_change(v):
            clamped = max(0, min(300, v))
            htol_slider.blockSignals(True)
            htol_slider.setValue(clamped)
            htol_slider.blockSignals(False)
            self.on_slider_change("merge_horizontal_tolerance", v, self.htol_label, "{:.0f}")
            self.update_merge_viz()
        
        htol_slider.valueChanged.connect(on_htol_slider_change)
        htol_slider.sliderReleased.connect(self.save_and_refresh)
        htol_spin.valueChanged.connect(on_htol_spin_change)
        htol_spin.editingFinished.connect(self.save_and_refresh)
        
        htol_layout.addWidget(htol_label_text)
        htol_layout.addWidget(htol_slider)
        htol_layout.addWidget(htol_spin)
        htol_layout.addWidget(self.htol_label)
        g_layout.addLayout(htol_layout)
        info_htol = QLabel("Max horizontal offset to consider aligned (px)")
        info_htol.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        info_htol.setWordWrap(True)
        g_layout.addWidget(info_htol)

        # Width Ratio
        ratio_layout = QHBoxLayout()
        ratio_label_text = QLabel("Width Ratio Threshold:")
        self.ratio_label = QLabel(f"{td_config.get('merge_width_ratio_threshold', 0.3):.2f}")
        self.ratio_label.setMinimumWidth(50)
        self.ratio_label.setStyleSheet("color: #aaa; font-weight: 600;")
        ratio_slider = QSlider(Qt.Orientation.Horizontal)
        ratio_slider.setMinimum(0)
        ratio_slider.setMaximum(100)
        ratio_slider.setValue(int(td_config.get("merge_width_ratio_threshold", 0.3) * 100))
        ratio_slider.valueChanged.connect(
            lambda v: self.on_slider_change("merge_width_ratio_threshold", v / 100.0, self.ratio_label, "{:.2f}")
        )
        ratio_slider.sliderReleased.connect(self.save_and_refresh)
        ratio_layout.addWidget(ratio_label_text)
        ratio_layout.addWidget(ratio_slider)
        ratio_layout.addWidget(self.ratio_label)
        g_layout.addLayout(ratio_layout)
        info_ratio = QLabel("Min width similarity ratio (0.0-1.0) to merge boxes")
        info_ratio.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        info_ratio.setWordWrap(True)
        g_layout.addWidget(info_ratio)

        # Store references
        self.vtol_slider = vtol_slider
        self.htol_slider = htol_slider
        self.vtol_spin = vtol_spin
        self.htol_spin = htol_spin

        # Initialize visualizer
        self.update_merge_viz()

        group.setLayout(g_layout)
        layout.addWidget(group)

    def on_slider_change(self, key: str, value: float, label: QLabel, fmt: str):
        """Update label and config when slider changes"""
        label.setText(fmt.format(value))
        if "text_detection" not in self.config:
            self.config["text_detection"] = {}
        self.config["text_detection"][key] = value

    def on_confidence_change(self, value: float):
        """Handle confidence slider change"""
        self.on_slider_change("min_confidence", value, self.conf_label, "{:.2f}")
        self.update_paddle_viz()

    def update_paddle_viz(self):
        """Update paddle visualizer"""
        if hasattr(self, 'paddle_viz'):
            td_config = self.config.get("text_detection", {})
            conf = td_config.get("min_confidence", 0.6)
            width = td_config.get("min_width", 30)
            height = td_config.get("min_height", 30)
            self.paddle_viz.update_values(conf, width, height)

    def update_merge_viz(self):
        """Update merge visualizer"""
        if hasattr(self, 'merge_viz'):
            td_config = self.config.get("text_detection", {})
            v_tol = td_config.get("merge_vertical_tolerance", 30)
            h_tol = td_config.get("merge_horizontal_tolerance", 50)
            self.merge_viz.update_values(v_tol, h_tol)

    def update_api_config(self):
        """Update API config from input fields"""
        self.config["api_url"] = self.api_url_input.text()
        self.config["api_key"] = self.api_key_input.text()
        self.config["model"] = self.model_input.text()
        self.save_config()

    def save_and_refresh(self):
        """Save config and refresh detection if image exists"""
        self.save_config()
        if state.last_image:
            # Debounce: wait a bit before refreshing
            QTimer.singleShot(300, self.run_detection_preview)

    def save_config(self):
        """Save config to file"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def setup_hotkey_listener(self):
        """Setup global hotkey listener in background thread"""
        if not KEYBOARD_AVAILABLE:
            print("[Hotkey] Keyboard module not available. Hotkey disabled.")
            return

        def listen():
            try:
                keyboard.add_hotkey("ctrl+shift+alt+z", self.trigger_hotkey)
                print("[Hotkey] Registered Ctrl+Shift+Alt+Z for screenshot capture")
                keyboard.wait()
            except Exception as e:
                print(f"[Hotkey] Error: {e}")

        t = threading.Thread(target=listen, daemon=True)
        t.start()

    def trigger_hotkey(self):
        """Called when hotkey is pressed (runs in background thread)"""
        # Use QTimer to safely call GUI method from background thread
        QTimer.singleShot(0, self.run_capture_and_detect)

    def run_capture_and_detect(self):
        """Capture screenshot and run detection"""
        self.status_label.setText("Capturing screenshot...")
        self.progress_bar.setValue(5)

        img = capture_screenshot()
        if img:
            state.last_image = img
            state.reset_detections()  # Clear old cached detections
            state.screenshot_version += 1
            self.display_image(img)
            self.run_detection_preview()
        else:
            self.status_label.setText("Failed to capture screenshot")
            self.progress_bar.setValue(0)

    def display_image(self, pil_image):
        """Display PIL image in graphics view"""
        self.scene.clear()
        self.filtered_boxes = []
        self.merged_boxes = []
        self.pixmap_item = None  # Reset reference

        # Convert PIL to QPixmap
        im_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QImage(im_data, pil_image.size[0], pil_image.size[1], QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)
        self.pixmap_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fit_image_to_view()

    def clear_all_boxes(self):
        """Remove all box items from scene (except pixmap)"""
        if not self.pixmap_item:
            return  # No image loaded yet
            
        items_to_remove = []
        items_list = self.scene.items()
        
        for item in items_list:
            # Keep only the pixmap item (by reference and type check)
            if item != self.pixmap_item and not isinstance(item, QGraphicsPixmapItem):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.scene.removeItem(item)

    def draw_filtered_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        """Draw red filtered boxes (raw detections)"""
        pen = QPen(QColor(255, 0, 0), 2)  # Red outline
        brush = QBrush(QColor(255, 0, 0, 76))  # Semi-transparent red fill (30% of 255)

        for (x1, y1, x2, y2) in boxes:
            self.scene.addRect(
                float(x1), float(y1),
                float(x2 - x1), float(y2 - y1),
                pen, brush
            )

    def draw_merged_boxes(self, merged_boxes_info: List[dict]):
        """Draw blue merged boxes with count labels"""
        pen = QPen(QColor(0, 100, 255), 2)  # Blue outline
        brush = QBrush(QColor(0, 100, 255, 51))  # Semi-transparent blue fill (20% of 255)

        for box_info in merged_boxes_info:
            rect = box_info["rect"]
            count = box_info.get("count", 1)
            x1, y1, x2, y2 = rect
            width = x2 - x1
            height = y2 - y1

            # Draw merged box
            rect_item = self.scene.addRect(
                float(x1), float(y1),
                float(width), float(height),
                pen, brush
            )

            # Add count label if merged multiple boxes
            if count > 1:
                # Add background rectangle first (so it's behind text)
                bg_rect = self.scene.addRect(
                    float(x1) + 1, float(y1) + 1,
                    20, 16,
                    QPen(Qt.PenStyle.NoPen),
                    QBrush(QColor(0, 100, 255))
                )
                bg_rect.setZValue(100)
                
                # Add text on top
                text_item = QGraphicsTextItem(str(count))
                text_item.setDefaultTextColor(QColor(255, 255, 255))
                font = QFont()
                font.setBold(True)
                font.setPointSize(10)
                text_item.setFont(font)
                text_item.setPos(float(x1) + 2, float(y1) + 2)
                text_item.setZValue(101)
                self.scene.addItem(text_item)

    def draw_tolerance_zones(self, merged_boxes_info: List[dict]):
        """Draw yellow tolerance zones around original boxes"""
        td_config = self.config.get("text_detection", {})
        v_tol = td_config.get("merge_vertical_tolerance", 30)
        h_tol = td_config.get("merge_horizontal_tolerance", 50)

        pen = QPen(QColor(255, 255, 0), 1)  # Yellow outline
        pen.setStyle(Qt.PenStyle.DashLine)
        brush = QBrush(QColor(255, 255, 0, 38))  # Semi-transparent yellow fill (15% of 255)

        for box_info in merged_boxes_info:
            count = box_info.get("count", 1)
            original_boxes = box_info.get("originalBoxes", [])
            
            # Only draw tolerance zones for merged boxes (count > 1)
            if count > 1 and original_boxes:
                for orig_box in original_boxes:
                    ox1, oy1, ox2, oy2 = orig_box
                    # Expand by tolerance
                    tol_x1 = ox1 - h_tol
                    tol_y1 = oy1 - v_tol
                    tol_x2 = ox2 + h_tol
                    tol_y2 = oy2 + v_tol
                    
                    self.scene.addRect(
                        float(tol_x1), float(tol_y1),
                        float(tol_x2 - tol_x1), float(tol_y2 - tol_y1),
                        pen, brush
                    )

    def draw_all_boxes(self):
        """Draw all box types: red (filtered), yellow (tolerance), blue (merged)"""
        self.clear_all_boxes()
        
        # Draw in order: tolerance zones (bottom), filtered boxes, merged boxes (top)
        if self.merged_boxes:
            self.draw_tolerance_zones(self.merged_boxes)
        if self.filtered_boxes:
            self.draw_filtered_boxes(self.filtered_boxes)
        if self.merged_boxes:
            self.draw_merged_boxes(self.merged_boxes)

    def run_detection_preview(self):
        """Run detection only (for preview)"""
        if not state.last_image:
            self.status_label.setText("No image captured")
            return
        self.start_worker(mode="detection_only")

    def run_extraction(self):
        """Run full extraction (detection + OCR)"""
        if not state.last_image:
            self.status_label.setText("No image captured")
            return
        self.start_worker(mode="full")

    def start_worker(self, mode: str):
        """Start OCR worker thread"""
        # Cancel existing worker if running
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)

        # Update UI state
        self.btn_cancel.setEnabled(True)
        self.btn_capture.setEnabled(False)
        self.btn_extract.setEnabled(False)
        self.progress_bar.setValue(0)

        # Create and start worker
        self.worker = OCRWorker(mode=mode, config=self.config)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.start()

    def cancel_process(self):
        """Cancel current OCR process"""
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")

    def update_progress(self, msg: str, percent: int):
        """Update progress bar and status"""
        self.status_label.setText(msg)
        self.progress_bar.setValue(percent)

    def on_worker_finished(self, filtered_boxes: List, merged_boxes_info: List, text: Optional[str]):
        """Handle worker completion"""
        self.btn_cancel.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_extract.setEnabled(True)
        self.progress_bar.setValue(100)

        # Store boxes
        self.filtered_boxes = filtered_boxes if filtered_boxes else []
        self.merged_boxes = merged_boxes_info if merged_boxes_info else []

        # Draw all box types
        self.draw_all_boxes()

        # Update text output
        if text:
            self.text_output.setText(text)
            # Copy to clipboard
            if CLIPBOARD_AVAILABLE:
                try:
                    pyperclip.copy(text)
                    self.status_label.setText(f"Done! Text copied to clipboard ({len(text)} chars)")
                except Exception as e:
                    self.status_label.setText(f"Done! (Clipboard error: {e})")
            else:
                self.status_label.setText("Done! (Clipboard not available)")
        else:
            self.status_label.setText("Detection complete (no text extracted)")

    def on_worker_error(self, err: str):
        """Handle worker error"""
        self.status_label.setText(f"Error: {err}")
        self.btn_cancel.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_extract.setEnabled(True)
        self.progress_bar.setValue(0)

    def fit_image_to_view(self):
        """Fit the image to the view while maintaining aspect ratio"""
        if self.pixmap_item:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        """Handle window resize - refit image to maintain aspect ratio"""
        super().resizeEvent(event)
        # Use QTimer to delay fitInView until after resize is complete
        QTimer.singleShot(0, self.fit_image_to_view)

    def eventFilter(self, obj, event):
        """Filter events to handle viewport resize"""
        if obj == self.view.viewport() and event.type() == QEvent.Type.Resize:
            # Viewport resized, refit image
            QTimer.singleShot(0, self.fit_image_to_view)
        return super().eventFilter(obj, event)

    def apply_dark_theme(self):
        """Apply dark theme using built-in QSS (no external dependencies)"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #888;
            }
            QTextEdit, QLineEdit {
                background-color: #252526;
                border: 1px solid #3e3e42;
                color: #ffffff;
                padding: 4px;
                border-radius: 3px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #007acc;
            }
            QLabel {
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                background: #3e3e42;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #0098ff;
            }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 3px;
                text-align: center;
                background-color: #252526;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 2px;
            }
            QGraphicsView {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 3px;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                background: #252526;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #3e3e42;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #505050;
            }
            QScrollBar:horizontal {
                background: #252526;
                height: 12px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #3e3e42;
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #505050;
            }
        """)


def main():
    app = QApplication(sys.argv)
    window = OCRWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

