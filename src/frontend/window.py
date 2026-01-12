"""Main window for the OCR GUI"""
import json
import threading
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGroupBox, QPushButton, QTextEdit, QProgressBar,
    QGraphicsView, QGraphicsScene, QSplitter, QLineEdit, QSpinBox,
    QGraphicsTextItem, QGraphicsPixmapItem, QScrollArea, QComboBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QRect, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont, QPainterPath

from src.backend.core.config import load_config, CONFIG_FILE
from src.backend.core.capture import capture_screenshot
from src.backend.state import state
from src.frontend.widgets import DetectionVizWidget, MergeVizWidget, ResizeVizWidget
from src.frontend.worker import OCRWorker
from src.frontend.theme import DARK_THEME_STYLESHEET
from src.frontend.constants import KEYBOARD_AVAILABLE, CLIPBOARD_AVAILABLE, keyboard, pyperclip


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

        # New Preprocessing Group
        self.create_image_preprocessing_group(controls_layout)

        # Preprocessing Group (Legacy/Size)
        self.create_preprocessing_group(controls_layout)

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

    def create_image_preprocessing_group(self, layout):
        """Create OpenCV-based Image Preprocessing settings group"""
        group = QGroupBox("Image Adjustments")
        g_layout = QVBoxLayout()
        pp_config = self.config.get("preprocessing", {})

        # 1. Invert Checkbox
        chk_layout = QHBoxLayout()
        self.chk_invert = QComboBox()
        self.chk_invert.addItem("Normal Colors", False)
        self.chk_invert.addItem("Invert Colors", True)
        self.chk_invert.setCurrentIndex(1 if pp_config.get("invert", False) else 0)
        
        self.chk_invert.currentIndexChanged.connect(
            lambda idx: self.update_pp("invert", bool(self.chk_invert.currentData()))
        )
        chk_layout.addWidget(QLabel("Colors:"))
        chk_layout.addWidget(self.chk_invert)
        g_layout.addLayout(chk_layout)

        # 2. Binary Threshold (0 = Disabled)
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Binary Threshold:")
        self.thresh_val = QLabel(str(pp_config.get("binary_threshold", 0)))
        self.thresh_val.setMinimumWidth(30)
        
        thresh_slider = QSlider(Qt.Orientation.Horizontal)
        thresh_slider.setRange(0, 255)
        thresh_slider.setValue(pp_config.get("binary_threshold", 0))
        
        thresh_slider.valueChanged.connect(
            lambda v: self.update_pp_slider("binary_threshold", v, self.thresh_val)
        )
        thresh_slider.sliderReleased.connect(self.save_and_refresh)
        
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(thresh_slider)
        thresh_layout.addWidget(self.thresh_val)
        g_layout.addLayout(thresh_layout)
        g_layout.addWidget(QLabel("(0 = Disabled). Helps separate text from background."))

        # 3. Contrast (1.0 = Normal)
        cont_layout = QHBoxLayout()
        cont_label = QLabel("Contrast:")
        self.cont_val = QLabel(f"{pp_config.get('contrast', 1.0):.1f}")
        self.cont_val.setMinimumWidth(30)
        
        cont_slider = QSlider(Qt.Orientation.Horizontal)
        cont_slider.setRange(5, 30) # 0.5 to 3.0
        cont_slider.setValue(int(pp_config.get("contrast", 1.0) * 10))
        
        cont_slider.valueChanged.connect(
            lambda v: self.update_pp_slider("contrast", v/10.0, self.cont_val, "{:.1f}")
        )
        cont_slider.sliderReleased.connect(self.save_and_refresh)
        
        cont_layout.addWidget(cont_label)
        cont_layout.addWidget(cont_slider)
        cont_layout.addWidget(self.cont_val)
        g_layout.addLayout(cont_layout)

        # 4. Brightness
        bright_layout = QHBoxLayout()
        bright_label = QLabel("Brightness:")
        self.bright_val = QLabel(str(pp_config.get("brightness", 0)))
        self.bright_val.setMinimumWidth(30)
        
        bright_slider = QSlider(Qt.Orientation.Horizontal)
        bright_slider.setRange(-100, 100)
        bright_slider.setValue(pp_config.get("brightness", 0))
        
        bright_slider.valueChanged.connect(
            lambda v: self.update_pp_slider("brightness", v, self.bright_val)
        )
        bright_slider.sliderReleased.connect(self.save_and_refresh)
        
        bright_layout.addWidget(bright_label)
        bright_layout.addWidget(bright_slider)
        bright_layout.addWidget(self.bright_val)
        g_layout.addLayout(bright_layout)

        # 5. Dilation
        dil_layout = QHBoxLayout()
        dil_layout.addWidget(QLabel("Text Thickness (Dilation):"))
        self.dil_spin = QSpinBox()
        self.dil_spin.setRange(0, 5)
        self.dil_spin.setValue(pp_config.get("dilation", 0))
        self.dil_spin.valueChanged.connect(lambda v: self.update_pp("dilation", v))
        dil_layout.addWidget(self.dil_spin)
        g_layout.addLayout(dil_layout)

        group.setLayout(g_layout)
        layout.addWidget(group)

    def update_pp(self, key, value):
        """Update preprocessing config"""
        if "preprocessing" not in self.config:
            self.config["preprocessing"] = {}
        self.config["preprocessing"][key] = value
        self.save_and_refresh()

    def update_pp_slider(self, key, value, label, fmt="{}"):
        """Update label and config for sliders"""
        label.setText(fmt.format(value))
        if "preprocessing" not in self.config:
            self.config["preprocessing"] = {}
        self.config["preprocessing"][key] = value

    def create_preprocessing_group(self, layout):
        """Create Image Processing settings group with pixelation preview"""
        group = QGroupBox("Image Size (Tokens)")
        g_layout = QVBoxLayout()

        # Visualizer
        viz_container = QWidget()
        viz_container.setStyleSheet("background: #000; border: 1px dashed #555; border-radius: 4px;")
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 5, 0, 5)
        self.resize_viz = ResizeVizWidget()
        viz_layout.addWidget(self.resize_viz, alignment=Qt.AlignmentFlag.AlignCenter)
        g_layout.addWidget(viz_container)

        # Max Dimension Controls
        dim_layout = QHBoxLayout()
        dim_label = QLabel("Max Dimension:")
        
        current_dim = self.config.get("max_image_dimension", 1080)
        self.dim_val_label = QLabel(f"{current_dim}px")
        self.dim_val_label.setMinimumWidth(50)
        self.dim_val_label.setStyleSheet("color: #aaa; font-weight: 600;")
        
        dim_slider = QSlider(Qt.Orientation.Horizontal)
        dim_slider.setMinimum(320)
        dim_slider.setMaximum(2560)
        dim_slider.setValue(current_dim)
        
        dim_spin = QSpinBox()
        dim_spin.setMinimum(1)  # Only positive numbers
        dim_spin.setMaximum(999999)  # Very high limit, effectively unlimited
        dim_spin.setValue(current_dim)
        dim_spin.setMaximumWidth(80)

        def on_dim_change(v):
            """Update label and visualizer when dimension changes"""
            self.dim_val_label.setText(f"{v}px")
            self.config["max_image_dimension"] = v
            self.resize_viz.update_value(v)
            
        def on_slider_move(v):
            """Handle slider movement"""
            dim_spin.blockSignals(True)
            dim_spin.setValue(v)
            dim_spin.blockSignals(False)
            on_dim_change(v)

        def on_spin_change(v):
            """Handle spinbox change"""
            # Only update slider if value is within slider range
            if dim_slider.minimum() <= v <= dim_slider.maximum():
                dim_slider.blockSignals(True)
                dim_slider.setValue(v)
                dim_slider.blockSignals(False)
            on_dim_change(v)
            self.save_config()  # Save immediately on spinbox finish

        dim_slider.valueChanged.connect(on_slider_move)
        dim_slider.sliderReleased.connect(self.save_config)
        dim_spin.valueChanged.connect(on_spin_change)

        dim_layout.addWidget(dim_label)
        dim_layout.addWidget(dim_slider)
        dim_layout.addWidget(dim_spin)
        dim_layout.addWidget(self.dim_val_label)
        
        g_layout.addLayout(dim_layout)
        
        info_label = QLabel("Resizes image before sending to API to save tokens/costs. Lower values = more pixelated.")
        info_label.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        info_label.setWordWrap(True)
        g_layout.addWidget(info_label)

        # Initialize viz
        self.resize_viz.update_value(current_dim)

        group.setLayout(g_layout)
        layout.addWidget(group)

    def create_separator(self):
        """Create a horizontal separator line"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #555;")
        return line

    def on_order_changed(self, index):
        """Update reading direction config"""
        direction = self.order_combo.currentData()
        if "text_sorting" not in self.config:
            self.config["text_sorting"] = {}
        self.config["text_sorting"]["direction"] = direction
        # Keep legacy reading_direction for backward compatibility
        if direction in ["horizontal_ltr", "vertical_ltr"]:
            self.config["reading_direction"] = "ltr"
        else:
            self.config["reading_direction"] = "rtl"
        self.save_and_refresh()

    def create_detection_group(self, layout):
        group = QGroupBox("Detection Settings (RapidOCR)")
        g_layout = QVBoxLayout()

        td_config = self.config.get("text_detection", {})

        # Mini Visualizer
        viz_container = QWidget()
        viz_container.setMinimumHeight(120)
        viz_container.setMaximumHeight(120)
        viz_container.setStyleSheet("background: #000; border: 1px dashed #555; border-radius: 4px;")
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        self.detection_viz = DetectionVizWidget()
        viz_layout.addWidget(self.detection_viz, alignment=Qt.AlignmentFlag.AlignCenter)
        g_layout.addWidget(viz_container)

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
            self.update_detection_viz()
        
        def on_width_spin_change(v):
            clamped = max(5, min(300, v))
            width_slider.blockSignals(True)
            width_slider.setValue(clamped)
            width_slider.blockSignals(False)
            self.on_slider_change("min_width", v, self.width_label, "{:.0f}")
            self.update_detection_viz()
        
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
            self.update_detection_viz()
        
        def on_height_spin_change(v):
            clamped = max(5, min(300, v))
            height_slider.blockSignals(True)
            height_slider.setValue(clamped)
            height_slider.blockSignals(False)
            self.on_slider_change("min_height", v, self.height_label, "{:.0f}")
            self.update_detection_viz()
        
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
        self.width_slider = width_slider
        self.height_slider = height_slider
        self.width_spin = width_spin
        self.height_spin = height_spin

        # Initialize visualizer
        self.update_detection_viz()

        group.setLayout(g_layout)
        layout.addWidget(group)

    def create_merge_group(self, layout):
        group = QGroupBox("Merge & Ordering")
        g_layout = QVBoxLayout()

        td_config = self.config.get("text_detection", {})

        # Reading Direction Control (New enhanced version)
        sort_config = self.config.get("text_sorting", {})
        if not sort_config:
            # Initialize if missing
            if "text_sorting" not in self.config:
                self.config["text_sorting"] = {}
            # Migrate from legacy reading_direction
            legacy_dir = self.config.get("reading_direction", "ltr")
            self.config["text_sorting"]["direction"] = "horizontal_ltr" if legacy_dir == "ltr" else "horizontal_rtl"
            self.config["text_sorting"]["group_tolerance"] = 0.5
            sort_config = self.config["text_sorting"]
        
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Reading Direction:"))
        
        self.order_combo = QComboBox()
        self.order_combo.addItem("Left to Right (Standard)", "horizontal_ltr")
        self.order_combo.addItem("Right to Left (Manga)", "horizontal_rtl")
        self.order_combo.addItem("Vertical Columns (Traditional)", "vertical_rtl")
        self.order_combo.addItem("Vertical Columns (LTR)", "vertical_ltr")
        
        # Set current value
        current_dir = sort_config.get("direction", "horizontal_ltr")
        index = self.order_combo.findData(current_dir)
        if index >= 0:
            self.order_combo.setCurrentIndex(index)
        
        self.order_combo.currentIndexChanged.connect(self.on_order_changed)
        order_layout.addWidget(self.order_combo)
        order_layout.addStretch()
        
        g_layout.addLayout(order_layout)
        
        # Grouping Tolerance Control
        grp_layout = QHBoxLayout()
        grp_label = QLabel("Line Grouping:")
        self.grp_val_label = QLabel(f"{sort_config.get('group_tolerance', 0.5):.2f}")
        self.grp_val_label.setMinimumWidth(50)
        self.grp_val_label.setStyleSheet("color: #aaa; font-weight: 600;")
        
        grp_slider = QSlider(Qt.Orientation.Horizontal)
        grp_slider.setMinimum(1)   # 0.1
        grp_slider.setMaximum(20)  # 2.0
        grp_slider.setValue(int(sort_config.get("group_tolerance", 0.5) * 10))
        
        grp_spin = QSpinBox()
        grp_spin.setMinimum(1)
        grp_spin.setMaximum(20)
        grp_spin.setValue(int(sort_config.get("group_tolerance", 0.5) * 10))
        grp_spin.setMaximumWidth(80)
        grp_spin.setSuffix(" (×0.1)")
        
        def on_grp_slider_change(v):
            val = v / 10.0
            self.grp_val_label.setText(f"{val:.2f}")
            grp_spin.blockSignals(True)
            grp_spin.setValue(v)
            grp_spin.blockSignals(False)
            if "text_sorting" not in self.config:
                self.config["text_sorting"] = {}
            self.config["text_sorting"]["group_tolerance"] = val
        
        def on_grp_spin_change(v):
            clamped = max(1, min(20, v))
            grp_slider.blockSignals(True)
            grp_slider.setValue(clamped)
            grp_slider.blockSignals(False)
            val = clamped / 10.0
            self.grp_val_label.setText(f"{val:.2f}")
            if "text_sorting" not in self.config:
                self.config["text_sorting"] = {}
            self.config["text_sorting"]["group_tolerance"] = val
        
        grp_slider.valueChanged.connect(on_grp_slider_change)
        grp_slider.sliderReleased.connect(self.save_and_refresh)
        grp_spin.valueChanged.connect(on_grp_spin_change)
        grp_spin.editingFinished.connect(self.save_and_refresh)
        
        grp_layout.addWidget(grp_label)
        grp_layout.addWidget(grp_slider)
        grp_layout.addWidget(grp_spin)
        grp_layout.addWidget(self.grp_val_label)
        g_layout.addLayout(grp_layout)
        
        info_grp = QLabel("Controls how strictly boxes must align to be in the same row/column. Lower = stricter grouping.")
        info_grp.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        info_grp.setWordWrap(True)
        g_layout.addWidget(info_grp)
        
        g_layout.addWidget(self.create_separator())

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
        def on_ratio_change(v):
            val = v / 100.0
            self.on_slider_change("merge_width_ratio_threshold", val, self.ratio_label, "{:.2f}")
            self.update_merge_viz()  # Update visualizer when ratio changes
        
        ratio_slider.valueChanged.connect(on_ratio_change)
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

    def update_detection_viz(self):
        """Update detection visualizer"""
        if hasattr(self, 'detection_viz'):
            td_config = self.config.get("text_detection", {})
            width = td_config.get("min_width", 30)
            height = td_config.get("min_height", 30)
            # Pass default confidence of 1.0 for visualization (not used for filtering)
            self.detection_viz.update_values(1.0, width, height)

    def update_merge_viz(self):
        """Update merge visualizer"""
        if hasattr(self, 'merge_viz'):
            td_config = self.config.get("text_detection", {})
            v_tol = td_config.get("merge_vertical_tolerance", 30)
            h_tol = td_config.get("merge_horizontal_tolerance", 50)
            ratio = td_config.get("merge_width_ratio_threshold", 0.3)
            self.merge_viz.update_values(v_tol, h_tol, ratio)

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
        """Draw blue merged boxes with count labels AND sequence numbers"""
        pen = QPen(QColor(0, 100, 255), 2)  # Blue outline
        brush = QBrush(QColor(0, 100, 255, 51))  # Semi-transparent blue fill (20% of 255)
        
        # Font for the sequence number
        seq_font = QFont("Arial")
        seq_font.setBold(True)
        seq_font.setPixelSize(14)

        for i, box_info in enumerate(merged_boxes_info):
            rect = box_info["rect"]
            count = box_info.get("count", 1)
            x1, y1, x2, y2 = rect
            width = x2 - x1
            height = y2 - y1

            # 1. Draw the box rectangle
            rect_item = self.scene.addRect(
                float(x1), float(y1),
                float(width), float(height),
                pen, brush
            )

            # 2. Draw Sequence Number Badge (Top-Left corner)
            # Circle background
            badge_size = 24
            badge_x = float(x1) - (badge_size / 2)
            badge_y = float(y1) - (badge_size / 2)
            
            badge_item = self.scene.addEllipse(
                badge_x, badge_y, badge_size, badge_size,
                QPen(QColor(255, 255, 255), 1),  # White border
                QBrush(QColor(0, 120, 215))      # Solid blue fill
            )
            badge_item.setZValue(200)  # Ensure it's on top

            # Number text
            seq_num = str(i + 1)
            text_item = QGraphicsTextItem(seq_num)
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            text_item.setFont(seq_font)
            
            # Center text in badge
            text_rect = text_item.boundingRect()
            text_x = badge_x + (badge_size - text_rect.width()) / 2
            text_y = badge_y + (badge_size - text_rect.height()) / 2
            
            text_item.setPos(text_x, text_y)
            text_item.setZValue(201)
            self.scene.addItem(text_item)

            # 3. Add count label if merged multiple boxes (bottom-right corner)
            if count > 1:
                # Add background rectangle first (so it's behind text)
                bg_rect = self.scene.addRect(
                    float(x2) - 21, float(y2) - 17,
                    20, 16,
                    QPen(Qt.PenStyle.NoPen),
                    QBrush(QColor(0, 100, 255))
                )
                bg_rect.setZValue(100)
                
                # Add text on top
                count_text = QGraphicsTextItem(str(count))
                count_text.setDefaultTextColor(QColor(255, 255, 255))
                count_font = QFont()
                count_font.setBold(True)
                count_font.setPointSize(10)
                count_text.setFont(count_font)
                count_text.setPos(float(x2) - 19, float(y2) - 15)
                count_text.setZValue(101)
                self.scene.addItem(count_text)

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

    def draw_ordering_visualization(self, merged_boxes_info: List[dict]):
        """Draw flow path, guide lines, and order numbers for reading order visualization"""
        if not merged_boxes_info:
            return
        
        sort_config = self.config.get("text_sorting", {})
        direction = sort_config.get("direction", "horizontal_ltr")
        group_tol = sort_config.get("group_tolerance", 0.5)
        
        # 1. Draw Flow Path (Connecting centers with arrows)
        path_pen = QPen(QColor(255, 0, 255), 2)  # Magenta/Purple
        path_pen.setStyle(Qt.PenStyle.DashLine)
        path_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        
        centers = []
        for box_info in merged_boxes_info:
            rect = box_info["rect"]
            x1, y1, x2, y2 = rect
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        if len(centers) > 1:
            path = QPainterPath()
            path.moveTo(*centers[0])
            for cx, cy in centers[1:]:
                path.lineTo(cx, cy)
            
            path_item = self.scene.addPath(path, path_pen)
            path_item.setZValue(150)  # Above tolerance zones, below boxes
            
            # Draw arrow heads at each segment
            arrow_pen = QPen(QColor(255, 0, 255), 2)
            arrow_brush = QBrush(QColor(255, 0, 255))
            for i in range(len(centers) - 1):
                x1, y1 = centers[i]
                x2, y2 = centers[i + 1]
                
                # Calculate arrow direction
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    # Normalize
                    dx /= length
                    dy /= length
                    
                    # Arrow size
                    arrow_size = 8
                    arrow_back = 12
                    
                    # Arrow tip
                    tip_x = x2
                    tip_y = y2
                    
                    # Arrow base points
                    perp_dx = -dy
                    perp_dy = dx
                    
                    base_x = tip_x - dx * arrow_back
                    base_y = tip_y - dy * arrow_back
                    
                    left_x = base_x + perp_dx * arrow_size
                    left_y = base_y + perp_dy * arrow_size
                    right_x = base_x - perp_dx * arrow_size
                    right_y = base_y - perp_dy * arrow_size
                    
                    # Draw arrow triangle
                    arrow_path = QPainterPath()
                    arrow_path.moveTo(tip_x, tip_y)
                    arrow_path.lineTo(left_x, left_y)
                    arrow_path.lineTo(right_x, right_y)
                    arrow_path.closeSubpath()
                    
                    arrow_item = self.scene.addPath(arrow_path, arrow_pen, arrow_brush)
                    arrow_item.setZValue(151)
        
        # 2. Draw Grouping Guide Lines (Axes)
        # These show the "bands" that boxes are grouped into
        guide_pen = QPen(QColor(255, 255, 255, 80), 1)  # Semi-transparent white
        guide_pen.setStyle(Qt.PenStyle.DotLine)
        
        # Get scene dimensions from scene rect
        scene_rect = self.scene.sceneRect()
        scene_width = scene_rect.width() if scene_rect.width() > 0 else 1000
        scene_height = scene_rect.height() if scene_rect.height() > 0 else 1000
        
        if "horizontal" in direction:
            # Horizontal guides: group by Y-coordinate
            # Find unique Y positions (grouped lines)
            y_positions = set()
            for box_info in merged_boxes_info:
                rect = box_info["rect"]
                x1, y1, x2, y2 = rect
                center_y = (y1 + y2) / 2
                y_positions.add(center_y)
            
            # Draw horizontal guide lines
            for y_pos in y_positions:
                guide_line = self.scene.addLine(0, y_pos, scene_width, y_pos, guide_pen)
                guide_line.setZValue(50)  # Below everything else
        else:
            # Vertical guides: group by X-coordinate
            x_positions = set()
            for box_info in merged_boxes_info:
                rect = box_info["rect"]
                x1, y1, x2, y2 = rect
                center_x = (x1 + x2) / 2
                x_positions.add(center_x)
            
            # Draw vertical guide lines
            for x_pos in x_positions:
                guide_line = self.scene.addLine(x_pos, 0, x_pos, scene_height, guide_pen)
                guide_line.setZValue(50)  # Below everything else
    
    def draw_all_boxes(self):
        """Draw all box types: red (filtered), yellow (tolerance), blue (merged), and ordering visualization"""
        self.clear_all_boxes()
        
        # Draw in order: guide lines (bottom), tolerance zones, filtered boxes, merged boxes, flow path (top)
        if self.merged_boxes:
            self.draw_ordering_visualization(self.merged_boxes)
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
        self.worker.image_processed_signal.connect(self.display_image)
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
        self.setStyleSheet(DARK_THEME_STYLESHEET)

