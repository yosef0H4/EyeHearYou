"""Main window for the OCR GUI"""
import json
import threading
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGroupBox, QPushButton, QTextEdit, QProgressBar,
    QGraphicsView, QGraphicsScene, QSplitter, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGraphicsTextItem, QGraphicsPixmapItem, QScrollArea, QComboBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QRect, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont, QPainterPath

from src.backend.core.config import load_config, CONFIG_FILE
from src.backend.core.capture import capture_screenshot
from src.backend.state import state
from src.backend.core.filtering import filter_text_regions, sort_text_regions_by_reading_order
from src.backend.core.merging import merge_close_text_boxes
from src.frontend.widgets import UnifiedSettingsViz, ResizeVizWidget
from src.frontend.worker import OCRWorker
from src.frontend.theme import DARK_THEME_STYLESHEET
from src.frontend.constants import KEYBOARD_AVAILABLE, CLIPBOARD_AVAILABLE, keyboard, pyperclip


class OCRWindow(QMainWindow):
    # Default values for reset functionality
    DEFAULT_CONFIG = {
        "max_image_dimension": 1080,
        "preprocessing": {
            "binary_threshold": 0,
            "invert": False,
            "dilation": 0,
            "contrast": 1.0,
            "brightness": 0
        },
        "text_detection": {
            "min_width": 30,
            "min_height": 30,
            "merge_vertical_tolerance": 4,
            "merge_horizontal_tolerance": 50,
            "merge_width_ratio_threshold": 0.3
        },
        "text_sorting": {
            "direction": "horizontal_ltr",
            "group_tolerance": 0.8
        }
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Novel OCR - Settings Tuner")
        self.resize(1400, 950)
        self.config = load_config()
        self.worker: Optional[OCRWorker] = None
        self.raw_boxes: List[Tuple[int, int, int, int]] = []  # All detections (unfiltered)
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
        scroll_area.setMinimumWidth(420)
        scroll_area.setMaximumWidth(520)
        
        # Create the controls widget
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(15)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # 1. Model Info Label
        model_info = QLabel("Model: H2OVL-Mississippi-0.8B (Local)")
        model_info.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px;")
        controls_layout.addWidget(model_info)

        # 2. Image Adjustments Group (Consolidated)
        self.create_image_settings_group(controls_layout)

        # 3. Text Processing Group (Consolidated Detection + Merge)
        self.create_text_processing_group(controls_layout)

        # 4. TTS Group
        self.create_tts_group(controls_layout)

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
        splitter.setSizes([460, 940])

        # Apply dark theme
        self.apply_dark_theme()


    def create_image_settings_group(self, layout):
        """Combined Image Settings (Size + Preprocessing)"""
        group = QGroupBox("Image Adjustments")
        g_layout = QVBoxLayout()
        pp_config = self.config.get("preprocessing", {})

        # --- Sub-section: Resizing Visualizer ---
        # Visualizer (centered, full width)
        viz_container = QWidget()
        viz_container.setStyleSheet("background: #000; border: 1px dashed #555; border-radius: 4px;")
        viz_box = QVBoxLayout(viz_container)
        viz_box.setContentsMargins(2, 2, 2, 2)
        self.resize_viz = ResizeVizWidget()
        viz_box.addWidget(self.resize_viz, alignment=Qt.AlignmentFlag.AlignCenter)
        g_layout.addWidget(viz_container)
        
        g_layout.addWidget(self.create_separator())

        # --- Sub-section: Preprocessing ---
        
        # Max Dimension slider (moved here from next to visualizer)
        current_dim = self.config.get("max_image_dimension", 1080)
        dim_row = QHBoxLayout()
        dim_row.setSpacing(8)
        dim_row.addWidget(QLabel("Max Dimension:"))
        
        dim_slider = QSlider(Qt.Orientation.Horizontal)
        dim_slider.setRange(320, 2560)
        dim_slider.setValue(current_dim)
        
        dim_spin = QSpinBox()
        dim_spin.setRange(1, 99999)
        dim_spin.setValue(current_dim)
        dim_spin.setSuffix(" px")
        dim_spin.setMaximumWidth(70)

        # Reset button
        def reset_dim():
            default_val = self.DEFAULT_CONFIG["max_image_dimension"]
            dim_slider.setValue(default_val)
            dim_spin.setValue(default_val)
            self.config["max_image_dimension"] = default_val
            self.resize_viz.update_value(default_val)
            self.save_config()
        dim_reset = self.create_reset_button(reset_dim)

        # Callbacks
        def update_dim(v):
            self.config["max_image_dimension"] = v
            self.resize_viz.update_value(v)

        dim_slider.valueChanged.connect(lambda v: [dim_spin.setValue(v), update_dim(v)])
        dim_slider.sliderReleased.connect(self.save_config)
        dim_spin.valueChanged.connect(lambda v: [dim_slider.setValue(v), update_dim(v), self.save_config()])
        
        # Init Viz
        self.resize_viz.update_value(current_dim)

        dim_row.addWidget(dim_slider)
        dim_row.addWidget(dim_spin)
        dim_row.addWidget(dim_reset)
        g_layout.addLayout(dim_row)
        
        # Helper to create slider rows
        def add_slider_row(label, key, min_v, max_v, default, scale=1.0, is_float=False):
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(QLabel(label))
            
            slider = QSlider(Qt.Orientation.Horizontal)
            # Map float to int for slider if needed
            s_min = int(min_v * scale) if is_float else int(min_v)
            s_max = int(max_v * scale) if is_float else int(max_v)
            cur_val = pp_config.get(key, default)
            s_val = int(cur_val * scale) if is_float else int(cur_val)
            
            slider.setRange(s_min, s_max)
            slider.setValue(s_val)
            
            if is_float:
                spin = QDoubleSpinBox()
                spin.setRange(min_v, max_v)
                spin.setSingleStep(0.1)
                spin.setValue(cur_val)
            else:
                spin = QSpinBox()
                spin.setRange(min_v, max_v)
                spin.setValue(cur_val)
            spin.setMaximumWidth(70)

            # Sync logic
            def on_slider(v):
                val = v / scale if is_float else v
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
                if "preprocessing" not in self.config: self.config["preprocessing"] = {}
                self.config["preprocessing"][key] = val

            def on_spin(v):
                s_v = int(v * scale) if is_float else int(v)
                slider.blockSignals(True)
                slider.setValue(s_v)
                slider.blockSignals(False)
                if "preprocessing" not in self.config: self.config["preprocessing"] = {}
                self.config["preprocessing"][key] = v
                self.save_and_refresh()

            slider.valueChanged.connect(on_slider)
            slider.sliderReleased.connect(self.save_and_refresh)
            spin.valueChanged.connect(on_spin)
            spin.editingFinished.connect(self.save_and_refresh)
            
            # Reset button
            def reset_val():
                default_val = self.DEFAULT_CONFIG["preprocessing"].get(key, default)
                if is_float:
                    slider.setValue(int(default_val * scale))
                    spin.setValue(default_val)
                else:
                    slider.setValue(int(default_val))
                    spin.setValue(int(default_val))
                if "preprocessing" not in self.config: self.config["preprocessing"] = {}
                self.config["preprocessing"][key] = default_val
                self.save_and_refresh()
            reset_btn = self.create_reset_button(reset_val)
            
            row.addWidget(slider)
            row.addWidget(spin)
            row.addWidget(reset_btn)
            g_layout.addLayout(row)

        # 1. Invert
        chk_invert = QComboBox()
        chk_invert.addItem("Normal Colors", False)
        chk_invert.addItem("Invert Colors", True)
        chk_invert.setCurrentIndex(1 if pp_config.get("invert", False) else 0)
        chk_invert.currentIndexChanged.connect(lambda: self.update_pp("invert", bool(chk_invert.currentData())))
        inv_layout = QHBoxLayout()
        inv_layout.setSpacing(8)
        inv_layout.addWidget(QLabel("Colors:"))
        inv_layout.addWidget(chk_invert)
        
        # Reset button for Colors
        def reset_invert():
            default_val = self.DEFAULT_CONFIG["preprocessing"].get("invert", False)
            chk_invert.setCurrentIndex(1 if default_val else 0)
            self.update_pp("invert", default_val)
        inv_reset = self.create_reset_button(reset_invert)
        inv_layout.addWidget(inv_reset)
        
        g_layout.addLayout(inv_layout)

        # 2. Threshold
        add_slider_row("Bin. Threshold:", "binary_threshold", 0, 255, 0)
        
        # 3. Contrast
        add_slider_row("Contrast:", "contrast", 0.5, 3.0, 1.0, scale=10.0, is_float=True)
        
        # 4. Brightness
        add_slider_row("Brightness:", "brightness", -100, 100, 0)
        
        # 5. Dilation
        add_slider_row("Thicken Text:", "dilation", 0, 5, 0)

        group.setLayout(g_layout)
        layout.addWidget(group)

    def update_pp(self, key, value):
        """Update preprocessing config"""
        if "preprocessing" not in self.config:
            self.config["preprocessing"] = {}
        self.config["preprocessing"][key] = value
        self.save_and_refresh()


    def create_separator(self):
        """Create a horizontal separator line"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #555;")
        return line

    def create_reset_button(self, callback):
        """Create a reset button with reload emoji icon"""
        btn = QPushButton("🔄")
        btn.setMaximumWidth(30)
        btn.setMaximumHeight(30)
        btn.setToolTip("Reset to default")
        btn.clicked.connect(callback)
        btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #555;
                border-radius: 4px;
                background: #2a2a2a;
                padding: 2px;
            }
            QPushButton:hover {
                background: #3a3a3a;
                border: 1px solid #666;
            }
            QPushButton:pressed {
                background: #1a1a1a;
            }
        """)
        return btn

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
        if self.raw_boxes:
            self.preview_live_merging()
        else:
            self.save_and_refresh()

    def create_text_processing_group(self, layout):
        """Combined Detection and Merging Settings"""
        group = QGroupBox("Text Detection & Merging")
        g_layout = QVBoxLayout()
        td_config = self.config.get("text_detection", {})
        
        # --- Shared Visualizer ---
        viz_container = QWidget()
        viz_container.setMinimumHeight(160)
        viz_container.setStyleSheet("background: #151515; border: 1px solid #333; border-radius: 6px;")
        viz_box = QHBoxLayout(viz_container)
        
        # Add the visualizer
        self.settings_viz = UnifiedSettingsViz()
        viz_box.addStretch()
        viz_box.addWidget(self.settings_viz)
        viz_box.addStretch()
        
        # Add legend/info next to it
        legend = QLabel("🟥 Red: Min Size Filter\n🟨 Yellow: Merge Zone\n🟦 Blue: Reference Box")
        legend.setStyleSheet("color: #aaa; font-size: 10px;")
        viz_box.addWidget(legend)
        
        g_layout.addWidget(viz_container)
        g_layout.addSpacing(10)

        # --- Helper for sections ---
        def add_header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 5px;")
            g_layout.addWidget(lbl)

        # --- Section 1: Detection (Size) ---
        add_header("1. Detection Filter (RapidOCR)")
        
        def update_viz_detection():
            w = self.config["text_detection"].get("min_width", 30)
            h = self.config["text_detection"].get("min_height", 30)
            self.settings_viz.update_detection(w, h)

        # Min Width
        w_row = QHBoxLayout()
        w_row.setSpacing(8)
        w_row.addWidget(QLabel("Min Width:"))
        w_sl = QSlider(Qt.Orientation.Horizontal)
        w_sl.setRange(5, 300)
        w_sl.setValue(td_config.get("min_width", 30))
        w_sp = QSpinBox()
        w_sp.setRange(5, 1000)
        w_sp.setValue(td_config.get("min_width", 30))
        w_sp.setSuffix(" px")
        w_sp.setMaximumWidth(70)

        def on_w_change(v):
            if "text_detection" not in self.config: self.config["text_detection"] = {}
            self.config["text_detection"]["min_width"] = v
            w_sl.blockSignals(True); w_sl.setValue(v); w_sl.blockSignals(False)
            w_sp.blockSignals(True); w_sp.setValue(v); w_sp.blockSignals(False)
            update_viz_detection()
            self.preview_live_filtering()
        
        w_sl.valueChanged.connect(on_w_change)
        w_sl.sliderReleased.connect(self.finalize_live_preview)
        w_sp.valueChanged.connect(on_w_change)
        w_sp.editingFinished.connect(self.finalize_live_preview)
        
        # Reset button
        def reset_w():
            default_val = self.DEFAULT_CONFIG["text_detection"].get("min_width", 30)
            w_sl.setValue(default_val)
            w_sp.setValue(default_val)
            on_w_change(default_val)
        w_reset = self.create_reset_button(reset_w)
        
        w_row.addWidget(w_sl); w_row.addWidget(w_sp); w_row.addWidget(w_reset)
        g_layout.addLayout(w_row)

        # Min Height
        h_row = QHBoxLayout()
        h_row.setSpacing(8)
        h_row.addWidget(QLabel("Min Height:"))
        h_sl = QSlider(Qt.Orientation.Horizontal)
        h_sl.setRange(5, 300)
        h_sl.setValue(td_config.get("min_height", 30))
        h_sp = QSpinBox()
        h_sp.setRange(5, 1000)
        h_sp.setValue(td_config.get("min_height", 30))
        h_sp.setSuffix(" px")
        h_sp.setMaximumWidth(70)

        def on_h_change(v):
            if "text_detection" not in self.config: self.config["text_detection"] = {}
            self.config["text_detection"]["min_height"] = v
            h_sl.blockSignals(True); h_sl.setValue(v); h_sl.blockSignals(False)
            h_sp.blockSignals(True); h_sp.setValue(v); h_sp.blockSignals(False)
            update_viz_detection()
            self.preview_live_filtering()
            
        h_sl.valueChanged.connect(on_h_change)
        h_sl.sliderReleased.connect(self.finalize_live_preview)
        h_sp.valueChanged.connect(on_h_change)
        h_sp.editingFinished.connect(self.finalize_live_preview)
        
        # Reset button
        def reset_h():
            default_val = self.DEFAULT_CONFIG["text_detection"].get("min_height", 30)
            h_sl.setValue(default_val)
            h_sp.setValue(default_val)
            on_h_change(default_val)
        h_reset = self.create_reset_button(reset_h)
        
        h_row.addWidget(h_sl); h_row.addWidget(h_sp); h_row.addWidget(h_reset)
        g_layout.addLayout(h_row)

        g_layout.addWidget(self.create_separator())

        # --- Section 2: Merging ---
        add_header("2. Merging & Reading Order")
        
        # Reading Direction
        sort_config = self.config.get("text_sorting", {})
        if not sort_config:
            # Initialize if missing
            if "text_sorting" not in self.config:
                self.config["text_sorting"] = {}
            # Migrate from legacy reading_direction
            legacy_dir = self.config.get("reading_direction", "ltr")
            self.config["text_sorting"]["direction"] = "horizontal_ltr" if legacy_dir == "ltr" else "horizontal_rtl"
            self.config["text_sorting"]["group_tolerance"] = 0.8
            sort_config = self.config["text_sorting"]
        dir_row = QHBoxLayout()
        dir_row.setSpacing(8)
        dir_row.addWidget(QLabel("Order:"))
        self.order_combo = QComboBox()
        self.order_combo.addItem("Left to Right (Standard)", "horizontal_ltr")
        self.order_combo.addItem("Right to Left (Manga)", "horizontal_rtl")
        self.order_combo.addItem("Vertical Columns (RTL)", "vertical_rtl")
        self.order_combo.addItem("Vertical Columns (LTR)", "vertical_ltr")
        
        cur_dir = sort_config.get("direction", "horizontal_ltr")
        idx = self.order_combo.findData(cur_dir)
        if idx >= 0: self.order_combo.setCurrentIndex(idx)
        self.order_combo.currentIndexChanged.connect(self.on_order_changed)
        
        # Reset button
        def reset_order():
            default_val = self.DEFAULT_CONFIG["text_sorting"].get("direction", "horizontal_ltr")
            idx = self.order_combo.findData(default_val)
            if idx >= 0:
                self.order_combo.setCurrentIndex(idx)
                self.on_order_changed(idx)
        dir_reset = self.create_reset_button(reset_order)
        
        dir_row.addWidget(self.order_combo, 1)
        dir_row.addWidget(dir_reset)
        g_layout.addLayout(dir_row)

        # Viz Update Logic for Merge
        def update_viz_merge():
            vt = self.config["text_detection"].get("merge_vertical_tolerance", 4)
            ht = self.config["text_detection"].get("merge_horizontal_tolerance", 50)
            rat = self.config["text_detection"].get("merge_width_ratio_threshold", 0.3)
            self.settings_viz.update_merge(vt, ht, rat)

        # Helper for merge sliders
        def add_merge_slider(label, key, default, max_val=300, is_float=False):
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(QLabel(label))
            
            sl = QSlider(Qt.Orientation.Horizontal)
            sp = QDoubleSpinBox() if is_float else QSpinBox()
            sp.setMaximumWidth(70)
            
            if is_float:
                sl.setRange(0, 100)
                sp.setRange(0.0, 1.0)
                sp.setSingleStep(0.01)
                val = td_config.get(key, default)
                sl.setValue(int(val * 100))
                sp.setValue(val)
            else:
                sl.setRange(0, max_val)
                sp.setRange(0, 1000)
                sp.setSuffix(" px")
                val = td_config.get(key, default)
                sl.setValue(val)
                sp.setValue(val)
            
            def on_change(v):
                real_val = v / 100.0 if is_float else v
                if "text_detection" not in self.config: self.config["text_detection"] = {}
                self.config["text_detection"][key] = real_val
                
                # Update UI counterparts
                if is_float:
                    sl.blockSignals(True); sl.setValue(int(real_val*100)); sl.blockSignals(False)
                    sp.blockSignals(True); sp.setValue(real_val); sp.blockSignals(False)
                else:
                    sl.blockSignals(True); sl.setValue(real_val); sl.blockSignals(False)
                    sp.blockSignals(True); sp.setValue(real_val); sp.blockSignals(False)
                
                update_viz_merge()
                self.preview_live_merging()
            
            if is_float:
                sl.valueChanged.connect(on_change)
                # Double spin box emits float
                sp.valueChanged.connect(on_change) 
            else:
                sl.valueChanged.connect(on_change)
                sp.valueChanged.connect(on_change)

            sl.sliderReleased.connect(self.finalize_live_preview)
            sp.editingFinished.connect(self.finalize_live_preview)
            
            # Reset button
            def reset_val():
                default_val = self.DEFAULT_CONFIG["text_detection"].get(key, default)
                if is_float:
                    sl.setValue(int(default_val * 100))
                    sp.setValue(default_val)
                else:
                    sl.setValue(int(default_val))
                    sp.setValue(int(default_val))
                on_change(default_val if not is_float else int(default_val * 100))
            reset_btn = self.create_reset_button(reset_val)
            
            row.addWidget(sl)
            row.addWidget(sp)
            row.addWidget(reset_btn)
            g_layout.addLayout(row)

        add_merge_slider("V. Tolerance:", "merge_vertical_tolerance", 4)
        add_merge_slider("H. Tolerance:", "merge_horizontal_tolerance", 50)
        add_merge_slider("Width Ratio:", "merge_width_ratio_threshold", 0.3, is_float=True)

        # Line Grouping (Sorting)
        grp_row = QHBoxLayout()
        grp_row.setSpacing(8)
        grp_row.addWidget(QLabel("Line Grouping:"))
        g_sl = QSlider(Qt.Orientation.Horizontal)
        g_sl.setRange(1, 20)
        grp_val = sort_config.get("group_tolerance", 0.8)
        g_sl.setValue(int(grp_val * 10))
        g_sp = QDoubleSpinBox()
        g_sp.setRange(0.1, 2.0)
        g_sp.setSingleStep(0.1)
        g_sp.setValue(grp_val)
        g_sp.setMaximumWidth(70)
        
        def on_grp_change(v):
            # v can be int (slider) or float (spinbox)
            real_val = v / 10.0 if isinstance(v, int) else v
            if "text_sorting" not in self.config: self.config["text_sorting"] = {}
            self.config["text_sorting"]["group_tolerance"] = real_val
            
            g_sl.blockSignals(True); g_sl.setValue(int(real_val*10)); g_sl.blockSignals(False)
            g_sp.blockSignals(True); g_sp.setValue(real_val); g_sp.blockSignals(False)
            self.preview_live_merging()
            
        g_sl.valueChanged.connect(on_grp_change)
        g_sl.sliderReleased.connect(self.finalize_live_preview)
        g_sp.valueChanged.connect(on_grp_change)
        g_sp.editingFinished.connect(self.finalize_live_preview)
        
        # Reset button
        def reset_grp():
            default_val = self.DEFAULT_CONFIG["text_sorting"].get("group_tolerance", 0.8)
            g_sl.setValue(int(default_val * 10))
            g_sp.setValue(default_val)
            on_grp_change(default_val)
        grp_reset = self.create_reset_button(reset_grp)
        
        grp_row.addWidget(g_sl)
        grp_row.addWidget(g_sp)
        grp_row.addWidget(grp_reset)
        g_layout.addLayout(grp_row)

        # Initial Viz Update
        update_viz_detection()
        update_viz_merge()

        group.setLayout(g_layout)
        layout.addWidget(group)

    def create_tts_group(self, layout):
        """Text-to-Speech Settings"""
        group = QGroupBox("🔊 Text-to-Speech (Kokoro)")
        g_layout = QVBoxLayout()
        tts_config = self.config.get("tts", {})

        # Enable Checkbox
        row1 = QHBoxLayout()
        chk_enable = QComboBox()
        chk_enable.addItem("Disabled", False)
        chk_enable.addItem("Enabled (Read Aloud)", True)
        is_enabled = tts_config.get("enabled", False)
        chk_enable.setCurrentIndex(1 if is_enabled else 0)

        def on_enable(idx):
            val = chk_enable.currentData()
            if "tts" not in self.config: self.config["tts"] = {}
            self.config["tts"]["enabled"] = val
            self.save_config()

        chk_enable.currentIndexChanged.connect(on_enable)
        row1.addWidget(QLabel("Status:"))
        row1.addWidget(chk_enable)
        g_layout.addLayout(row1)

        # Speed Slider
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Speed:"))
        
        sl_speed = QSlider(Qt.Orientation.Horizontal)
        sl_speed.setRange(5, 20)  # 0.5x to 2.0x
        current_speed = tts_config.get("speed", 1.0)
        sl_speed.setValue(int(current_speed * 10))
        
        sp_speed = QDoubleSpinBox()
        sp_speed.setRange(0.5, 2.0)
        sp_speed.setSingleStep(0.1)
        sp_speed.setValue(current_speed)
        sp_speed.setMaximumWidth(60)

        def on_speed(v):
            real_val = v / 10.0 if isinstance(v, int) else v
            if "tts" not in self.config: self.config["tts"] = {}
            self.config["tts"]["speed"] = real_val
            
            sl_speed.blockSignals(True); sl_speed.setValue(int(real_val*10)); sl_speed.blockSignals(False)
            sp_speed.blockSignals(True); sp_speed.setValue(real_val); sp_speed.blockSignals(False)
            self.save_config()

        sl_speed.valueChanged.connect(on_speed)
        sp_speed.valueChanged.connect(on_speed)

        row2.addWidget(sl_speed)
        row2.addWidget(sp_speed)
        g_layout.addLayout(row2)

        group.setLayout(g_layout)
        layout.addWidget(group)

    def save_and_refresh(self):
        """Save config and refresh detection if image exists"""
        self.save_config()
        if state.last_image:
            # Debounce: wait a bit before refreshing
            QTimer.singleShot(300, self.run_detection_preview)

    def finalize_live_preview(self):
        """Called when slider is released: Saves config and redraws clean state"""
        self.save_config()
        # We don't need to re-run detection if we have raw_boxes.
        # We can just apply the filters locally and "commit" the view.
        if self.raw_boxes:
            self.preview_live_merging(show_tolerance_zones=False)
            self.status_label.setText("Settings applied.")
        else:
            self.run_detection_preview()

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
                # Z = Capture + Detect + Extract + Copy
                keyboard.add_hotkey("ctrl+shift+alt+z", lambda: self.trigger_hotkey(mode="extract"))
                # X = Capture + Detect Only (Preview)
                keyboard.add_hotkey("ctrl+shift+alt+x", lambda: self.trigger_hotkey(mode="detect"))
                print("[Hotkey] Registered Ctrl+Shift+Alt+Z (Extract) and Ctrl+Shift+Alt+X (Detect)")
                keyboard.wait()
            except Exception as e:
                print(f"[Hotkey] Error: {e}")

        t = threading.Thread(target=listen, daemon=True)
        t.start()

    def trigger_hotkey(self, mode="extract"):
        """Called when hotkey is pressed (runs in background thread)"""
        # Use QTimer to safely call GUI method from background thread
        if mode == "extract":
            QTimer.singleShot(0, self.run_capture_and_extract)
        else:
            QTimer.singleShot(0, self.run_capture_and_detect)

    def run_capture_and_extract(self):
        """Capture screenshot and run full extraction (Z)"""
        self.status_label.setText("Capturing screenshot...")
        self.progress_bar.setValue(5)

        img = capture_screenshot()
        if img:
            state.last_image = img
            state.reset_detections()
            state.screenshot_version += 1
            self.display_image(img)
            self.run_extraction()
        else:
            self.status_label.setText("Failed to capture screenshot")
            self.progress_bar.setValue(0)

    def run_capture_and_detect(self):
        """Capture screenshot and run detection preview (X)"""
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
        self.raw_boxes = []
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

    def preview_live_filtering(self):
        """Show raw boxes: Green (Keep) vs Red (Discard) based on current min_width/min_height"""
        if not self.raw_boxes or not self.pixmap_item:
            return

        self.clear_all_boxes()
        
        min_w = self.config["text_detection"].get("min_width", 30)
        min_h = self.config["text_detection"].get("min_height", 30)
        img_w = self.pixmap_item.pixmap().width()
        img_h = self.pixmap_item.pixmap().height()
        
        # Prepare pens/brushes
        pen_keep = QPen(QColor(0, 255, 0), 2)  # Green solid
        brush_keep = QBrush(QColor(0, 255, 0, 50))  # Semi-transparent green
        
        pen_discard = QPen(QColor(255, 0, 0), 1)  # Red dashed
        pen_discard.setStyle(Qt.PenStyle.DashLine)
        brush_discard = QBrush(QColor(255, 0, 0, 30))  # Semi-transparent red
        
        # Filter logic (duplicated from backend for speed)
        for (x1, y1, x2, y2) in self.raw_boxes:
            # Clamp
            x1 = max(0, min(x1, img_w)); x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h)); y2 = max(0, min(y2, img_h))
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0: continue
            
            if w > min_w and h > min_h:
                # Keep (Green)
                self.scene.addRect(float(x1), float(y1), float(w), float(h), pen_keep, brush_keep)
            else:
                # Discard (Red Ghost)
                self.scene.addRect(float(x1), float(y1), float(w), float(h), pen_discard, brush_discard)

    def preview_live_merging(self, show_tolerance_zones=True):
        """Calculate and show merges live using Python (fast enough for typical OCR)"""
        if not self.raw_boxes or not self.pixmap_item:
            return

        # 1. Local Filtering
        min_w = self.config["text_detection"].get("min_width", 30)
        min_h = self.config["text_detection"].get("min_height", 30)
        img_w = self.pixmap_item.pixmap().width() if self.pixmap_item else 2000
        img_h = self.pixmap_item.pixmap().height() if self.pixmap_item else 2000
        
        filtered = filter_text_regions(self.raw_boxes, (img_h, img_w), min_width=min_w, min_height=min_h)
        
        # 2. Local Sorting
        sort_config = self.config.get("text_sorting", {})
        direction = sort_config.get("direction", "horizontal_ltr")
        group_tol = sort_config.get("group_tolerance", 0.8)
        
        sorted_regions = sort_text_regions_by_reading_order(
            filtered, direction=direction, group_tolerance=group_tol
        )
        
        # 3. Local Merging
        td_config = self.config.get("text_detection", {})
        merged, is_merged_flags, original_groups = merge_close_text_boxes(
            sorted_regions,
            vertical_tolerance=td_config.get("merge_vertical_tolerance", 4),
            horizontal_tolerance=td_config.get("merge_horizontal_tolerance", 50),
            width_ratio_threshold=td_config.get("merge_width_ratio_threshold", 0.3)
        )
        
        # 4. Re-Sort Results
        merged_regions = sort_text_regions_by_reading_order(
            merged, direction=direction, group_tolerance=group_tol
        )
        
        # Format for drawing
        # Create map to find original groups for the re-sorted merged regions
        # This is a lightweight approximate mapping for visualization
        merged_boxes_info = []
        
        # Helper to map back (simple version for viz)
        # We need to reconstruct the info dicts
        unsorted_info = []
        for m_box, flag, group in zip(merged, is_merged_flags, original_groups):
            if flag and group:
                unsorted_info.append({"rect": m_box, "count": len(group), "originalBoxes": group})
            else:
                unsorted_info.append({"rect": m_box, "count": 1, "originalBoxes": [m_box]})
        
        # Map rect -> info
        rect_map = {tuple(info["rect"]): info for info in unsorted_info}
        
        final_info_list = []
        for r in merged_regions:
            t = tuple(r)
            if t in rect_map:
                final_info_list.append(rect_map[t])
        
        # Draw
        self.clear_all_boxes()
        self.merged_boxes = final_info_list
        self.filtered_boxes = filtered  # Update state
        
        # Always draw ordering visualization
        if final_info_list:
            self.draw_ordering_visualization(final_info_list)
        
        # Draw tolerance zones if dragging (Yellow)
        if show_tolerance_zones and final_info_list:
            # Draw yellow zones for ALL boxes (even single ones) to show search area
            self.draw_tolerance_zones(final_info_list, force_draw_all=True)
            # Draw Width Ratio "Anchor Bars" (Cyan) to show alignment strictness
            self.draw_ratio_bars(final_info_list)
            
        # Draw Merged Result (Blue)
        self.draw_merged_boxes(final_info_list)

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

    def draw_tolerance_zones(self, merged_boxes_info: List[dict], force_draw_all=False):
        """Draw yellow tolerance zones around original boxes"""
        td_config = self.config.get("text_detection", {})
        v_tol = td_config.get("merge_vertical_tolerance", 4)
        h_tol = td_config.get("merge_horizontal_tolerance", 50)

        pen = QPen(QColor(255, 255, 0), 1)  # Yellow outline
        pen.setStyle(Qt.PenStyle.DashLine)
        brush = QBrush(QColor(255, 255, 0, 38))  # Semi-transparent yellow fill (15% of 255)

        for box_info in merged_boxes_info:
            count = box_info.get("count", 1)
            original_boxes = box_info.get("originalBoxes", [])
            
            # Draw if it's a merged group OR if we are in "live tuning" mode (force_draw_all)
            if (count > 1 or force_draw_all) and original_boxes:
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

    def draw_ratio_bars(self, merged_boxes_info: List[dict]):
        """
        Draw Cyan 'Anchor Bars' inside boxes to visualize width_ratio_threshold.
        This helps users understand how much vertical alignment/overlap is required.
        The bar represents the minimum horizontal overlap needed for vertical merging.
        """
        ratio = self.config["text_detection"].get("merge_width_ratio_threshold", 0.3)
        
        # Cyan pen/brush - semi-transparent for visibility
        brush = QBrush(QColor(0, 255, 255, 120))  # Cyan, semi-transparent
        
        for box_info in merged_boxes_info:
            # We want to draw this on the ORIGINAL boxes so users see why they are/aren't merging
            original_boxes = box_info.get("originalBoxes", [])
            
            for (x1, y1, x2, y2) in original_boxes:
                w = x2 - x1
                h = y2 - y1
                
                if w <= 0 or h <= 0:
                    continue
                
                # Calculate the anchor bar width based on ratio
                # Logic: x_overlap > min_w * ratio
                # So we visualize this required width in the center of the box
                bar_w = w * ratio
                bar_h = 4  # Thin bar
                
                # Center it horizontally
                bar_x = x1 + (w - bar_w) / 2
                
                # Draw at bottom (where it would connect to the next line)
                # Position it slightly above the bottom edge for visibility
                bar_y = y2 - 6
                
                # Make sure bar doesn't go outside the box
                if bar_y < y1:
                    bar_y = y1
                
                self.scene.addRect(
                    float(bar_x), float(bar_y), 
                    float(bar_w), float(bar_h), 
                    QPen(Qt.PenStyle.NoPen), brush
                )

    def draw_ordering_visualization(self, merged_boxes_info: List[dict]):
        """Draw flow path, guide lines, and order numbers for reading order visualization"""
        if not merged_boxes_info:
            return
        
        sort_config = self.config.get("text_sorting", {})
        direction = sort_config.get("direction", "horizontal_ltr")
        group_tol = sort_config.get("group_tolerance", 0.8)
        
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

    def on_worker_finished(self, raw_boxes: List, filtered_boxes: List, merged_boxes_info: List, text: Optional[str]):
        """Handle worker completion"""
        self.btn_cancel.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_extract.setEnabled(True)
        self.progress_bar.setValue(100)

        # Store boxes
        self.raw_boxes = raw_boxes if raw_boxes else []
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
            
            # Trigger TTS
            from src.backend.core.tts import speak_text
            speak_text(text)
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

