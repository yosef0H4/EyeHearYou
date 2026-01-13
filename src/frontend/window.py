"""Main window for the OCR GUI"""
import json
import threading
from typing import Optional, List, Tuple

import sounddevice as sd
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGroupBox, QPushButton, QTextEdit, QProgressBar,
    QGraphicsView, QGraphicsScene, QSplitter, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGraphicsTextItem, QGraphicsPixmapItem, QScrollArea, QComboBox, QFrame,
    QRadioButton, QButtonGroup, QCheckBox, QToolButton
)
from PyQt6.QtCore import Qt, QTimer, QRect, QEvent, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont, QPainterPath

from src.backend.core.config import (
    load_config, CONFIG_FILE, load_profiles, save_profiles,
    load_app_settings, save_app_settings
)
from src.backend.core.capture import capture_screenshot
from src.backend.state import state
from src.backend.core.filtering import (
    filter_text_regions, 
    sort_text_regions_by_reading_order,
    generate_selection_mask
)
from src.backend.core.merging import merge_close_text_boxes
from src.frontend.widgets import UnifiedSettingsViz, ResizeVizWidget, ManualBoxItem, HotkeyRecorder
from src.frontend.worker import OCRWorker
from src.frontend.theme import DARK_THEME_STYLESHEET
from src.frontend.constants import KEYBOARD_AVAILABLE, keyboard


class OCRWindow(QMainWindow):
    # Dictionary for UI translations
    TRANSLATIONS = {
        "en": {
            "window_title": "EyeHearYou - Settings Tuner",
            "model_info": "Model: H2OVL-Mississippi-0.8B (Local)",
            "profile_group": "Configuration Profile",
            "ui_lang_label": "UI Language:",
            "selection_group": "Selection Tools",
            "img_adj_group": "Image Adjustments",
            "text_proc_group": "Text Detection & Merging",
            "tts_group": "🔊 Text-to-Speech (Kokoro)",
            "status_group": "Status",
            "output_group": "Extracted Text",
            "preview_label": "Image Preview",
            "btn_capture": "📸 Capture Screenshot",
            "btn_extract": "🚀 Extract Text",
            "btn_cancel": "⛔ Cancel",
            "status_ready": "Ready",
            "lock_notice": "⚠️ Default profile is locked. Duplicate to edit.",
            "chk_rapid": "Use Smart Detection (RapidOCR)",
            "chk_rapid_tooltip": "Uncheck to manually select text areas",
            "chk_gpu": "Use GPU Acceleration",
            "chk_gpu_tooltip": "Enable GPU acceleration for RapidOCR (requires onnxruntime-gpu, ~27% faster)",
            "btn_tool_none": "✋ View",
            "btn_tool_add": "➕ Add Area",
            "btn_tool_sub": "➖ Remove Area",
            "btn_tool_manual": "📦 Manual Box",
            "btn_sel_all": "Select All",
            "btn_desel_all": "Deselect All",
            "btn_clear_manual": "Clear Manual",
            "btn_new_profile_tooltip": "Duplicate current profile",
            "btn_rename_profile_tooltip": "Rename current profile",
            "btn_delete_profile_tooltip": "Delete current profile",
            "hotkeys_group": "Hotkeys",
            "label_max_dimension": "Max Dimension:",
            "label_colors": "Colors:",
            "colors_normal": "Normal Colors",
            "colors_invert": "Invert Colors",
            "reset_tooltip": "Reset to default",
            "legend_text": "🟥 Red: Min Size Filter\n🟨 Yellow: Merge Zone\n🟦 Blue: Reference Box",
            "detection_header": "1. Detection Filter (Adaptive)",
            "detection_info": "Uses ratios relative to text size - works across all screen sizes!",
            "label_min_height_ratio": "Min Height Ratio:",
            "label_noise_filter": "Noise Filter:",
            "merging_header": "2. Merging & Reading Order",
            "label_order": "Order:",
            "order_ltr": "Left to Right (Standard)",
            "order_rtl": "Right to Left (Manga)",
            "order_vertical_rtl": "Vertical Columns (RTL)",
            "order_vertical_ltr": "Vertical Columns (LTR)",
            "label_v_ratio": "V. Ratio:",
            "label_h_ratio": "H. Ratio:",
            "label_width_ratio": "Width Ratio:",
            "tooltip_v_ratio": "Vertical gap as multiplier of text height (e.g., 0.07 = tight vertical merging)",
            "tooltip_h_ratio": "Horizontal gap as multiplier of text height (e.g., 0.37 = tight horizontal merging)",
            "tooltip_width_ratio": "Minimum horizontal overlap ratio for vertical merging",
            "label_line_grouping": "Line Grouping:",
            "tts_info": "TTS is always enabled - text will be read aloud automatically",
            "label_language": "Language:",
            "label_gender": "Gender:",
            "label_voice": "Voice:",
            "label_speed": "Speed:",
            "label_volume": "Volume:",
            "label_binary_threshold": "Bin. Threshold:",
            "label_contrast": "Contrast:",
            "label_brightness": "Brightness:",
            "label_dilation": "Text Thickness:",
            "btn_play": "▶ Play",
            "btn_play_tooltip": "Re-generate and play TTS with current voice/speed settings",
            "btn_replay": "🔁 Replay",
            "btn_replay_tooltip": "Replay the last generated audio (same voice/speed)",
            "btn_stop": "⏹ Stop",
            "btn_stop_tooltip": "Stop current audio playback",
            "label_phonemes": "Phonemes (IPA):",
            "phonemes_placeholder": "Phonemes will appear here after text is read..."
        },
        "ar": {
            "window_title": "EyeHearYou - موجه الإعدادات",
            "model_info": "النموذج: H2OVL-Mississippi-0.8B (محلي)",
            "profile_group": "ملف التعريف",
            "ui_lang_label": "لغة الواجهة:",
            "selection_group": "أدوات التحديد",
            "img_adj_group": "تعديلات الصورة",
            "text_proc_group": "كشف النص ودمجه",
            "tts_group": "🔊 تحويل النص إلى كلام (Kokoro)",
            "status_group": "الحالة",
            "output_group": "النص المستخرج",
            "preview_label": "معاينة الصورة",
            "btn_capture": "📸 التقاط الشاشة",
            "btn_extract": "🚀 استخراج النص",
            "btn_cancel": "⛔ إلغاء",
            "status_ready": "جاهز",
            "lock_notice": "⚠️ ملف التعريف الافتراضي مقفل. انسخه للتعديل.",
            "chk_rapid": "استخدام الكشف الذكي (RapidOCR)",
            "chk_rapid_tooltip": "قم بإلغاء التحديد لتحديد مناطق النص يدوياً",
            "chk_gpu": "استخدام تسريع GPU",
            "chk_gpu_tooltip": "تفعيل تسريع GPU لـ RapidOCR (يتطلب onnxruntime-gpu، أسرع بنسبة ~27%)",
            "btn_tool_none": "✋ عرض",
            "btn_tool_add": "➕ إضافة منطقة",
            "btn_tool_sub": "➖ إزالة منطقة",
            "btn_tool_manual": "📦 صندوق يدوي",
            "btn_sel_all": "تحديد الكل",
            "btn_desel_all": "إلغاء تحديد الكل",
            "btn_clear_manual": "مسح اليدوي",
            "btn_new_profile_tooltip": "نسخ ملف التعريف الحالي",
            "btn_rename_profile_tooltip": "إعادة تسمية ملف التعريف الحالي",
            "btn_delete_profile_tooltip": "حذف ملف التعريف الحالي",
            "hotkeys_group": "اختصارات لوحة المفاتيح",
            "label_max_dimension": "الحد الأقصى للأبعاد:",
            "label_colors": "الألوان:",
            "colors_normal": "ألوان عادية",
            "colors_invert": "عكس الألوان",
            "reset_tooltip": "إعادة تعيين إلى الافتراضي",
            "legend_text": "🟥 أحمر: مرشح الحد الأدنى للحجم\n🟨 أصفر: منطقة الدمج\n🟦 أزرق: صندوق مرجعي",
            "detection_header": "1. مرشح الكشف (متكيف)",
            "detection_info": "يستخدم نسباً نسبة لحجم النص - يعمل عبر جميع أحجام الشاشة!",
            "label_min_height_ratio": "نسبة الحد الأدنى للارتفاع:",
            "label_noise_filter": "مرشح الضوضاء:",
            "merging_header": "2. الدمج وترتيب القراءة",
            "label_order": "الترتيب:",
            "order_ltr": "من اليسار إلى اليمين (قياسي)",
            "order_rtl": "من اليمين إلى اليسار (مانجا)",
            "order_vertical_rtl": "أعمدة عمودية (من اليمين إلى اليسار)",
            "order_vertical_ltr": "أعمدة عمودية (من اليسار إلى اليمين)",
            "label_v_ratio": "نسبة عمودية:",
            "label_h_ratio": "نسبة أفقية:",
            "label_width_ratio": "نسبة العرض:",
            "tooltip_v_ratio": "الفجوة العمودية كمضاعف لارتفاع النص (مثلاً، 0.07 = دمج عمودي ضيق)",
            "tooltip_h_ratio": "الفجوة الأفقية كمضاعف لارتفاع النص (مثلاً، 0.37 = دمج أفقي ضيق)",
            "tooltip_width_ratio": "نسبة التداخل الأفقي الأدنى للدمج العمودي",
            "label_line_grouping": "تجميع الأسطر:",
            "tts_info": "تحويل النص إلى كلام مفعّل دائماً - سيتم قراءة النص تلقائياً",
            "label_language": "اللغة:",
            "label_gender": "الجنس:",
            "label_voice": "الصوت:",
            "label_speed": "السرعة:",
            "label_volume": "مستوى الصوت:",
            "label_binary_threshold": "عتبة الثنائية:",
            "label_contrast": "التباين:",
            "label_brightness": "السطوع:",
            "label_dilation": "سُمك النص:",
            "btn_play": "▶ تشغيل",
            "btn_play_tooltip": "إعادة توليد وتشغيل تحويل النص إلى كلام بالإعدادات الحالية للصوت/السرعة",
            "btn_replay": "🔁 إعادة التشغيل",
            "btn_replay_tooltip": "إعادة تشغيل الصوت المولّد آخر مرة (نفس الصوت/السرعة)",
            "btn_stop": "⏹ إيقاف",
            "btn_stop_tooltip": "إيقاف تشغيل الصوت الحالي",
            "label_phonemes": "الفونيمات (IPA):",
            "phonemes_placeholder": "ستظهر الفونيمات هنا بعد قراءة النص..."
        }
    }

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
            # Adaptive parameters (works across all screen sizes and font sizes)
            # Optimized defaults based on testing with various games
            "min_height_ratio": 0.031,
            "min_width_ratio": 0.0,
            "median_height_fraction": 1.0,
            "merge_vertical_ratio": 0.07,
            "merge_horizontal_ratio": 0.37,
            "merge_width_ratio_threshold": 0.75
        },
        "text_sorting": {
            "direction": "horizontal_ltr",
            "group_tolerance": 0.5
        }
    }

    def __init__(self):
        super().__init__()
        app_settings = load_app_settings()
        self.ui_lang = app_settings.get("ui_lang", "en")
        
        # Set layout direction based on language
        if self.ui_lang == "ar":
            self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        else:
            self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        
        self.setWindowTitle(self.TRANSLATIONS[self.ui_lang]["window_title"])
        self.resize(1400, 950)
        self.config = load_config()
        self.worker: Optional[OCRWorker] = None
        self.raw_boxes: List[Tuple[int, int, int, int]] = []  # All detections (unfiltered)
        self.filtered_boxes: List[Tuple[int, int, int, int]] = []
        self.merged_boxes: List[dict] = []  # List of dicts with 'rect', 'count', 'originalBoxes'
        self.pixmap_item = None  # Store reference to pixmap item
        self.manual_box_items = []  # Keep track of visual items for manual boxes
        
        # Widget references for profile refresh - will be populated during UI creation
        self._config_widgets = {}  # Maps config paths to widget tuples (slider, spinbox, etc.)
        
        # Load manual boxes from config (ensure they persist)
        if "manual_boxes" in self.config:
            # Convert lists (from JSON) back to tuples
            try:
                loaded_boxes = [tuple(box) for box in self.config["manual_boxes"]]
                state.manual_boxes = loaded_boxes
            except Exception:
                print("Failed to load manual boxes from config")
        
        # Selection Tool State
        self.tool_mode = "none"  # "none", "add", "sub", "manual"
        self.is_drawing = False
        self.start_point = QPointF()
        self.current_rect_item = None
        self.selection_overlay_item = None

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
        self.model_info = QLabel(self.TRANSLATIONS[self.ui_lang]["model_info"])
        self.model_info.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px;")
        controls_layout.addWidget(self.model_info)

        # Profile Management Group
        self.create_profile_group(controls_layout)

        # Locked Notice (Hidden by default)
        self.lock_notice = QLabel(self.TRANSLATIONS[self.ui_lang]["lock_notice"])
        self.lock_notice.setStyleSheet("color: #ffb74d; font-weight: bold; padding: 5px; background: #332b00; border-radius: 4px;")
        self.lock_notice.setVisible(False)
        controls_layout.addWidget(self.lock_notice)

        # NEW: Selection Tools Group
        self.create_selection_group(controls_layout)

        # NEW: Hotkeys Group
        self.create_hotkeys_group(controls_layout)

        # 2. Image Adjustments Group (Consolidated)
        self.create_image_settings_group(controls_layout)

        # 3. Text Processing Group (Consolidated Detection + Merge)
        self.create_text_processing_group(controls_layout)

        # 4. TTS Group
        self.create_tts_group(controls_layout)

        # Update profile buttons/locks after all groups are created
        self.update_profile_buttons()

        # Buttons
        btn_layout = QVBoxLayout()
        self.btn_capture = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_capture"])
        self.btn_capture.clicked.connect(self.run_capture_and_detect)
        self.btn_cancel = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_cancel"])
        self.btn_cancel.clicked.connect(self.cancel_process)
        self.btn_cancel.setEnabled(False)
        self.btn_extract = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_extract"])
        self.btn_extract.clicked.connect(self.run_extraction)

        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_extract)
        controls_layout.addLayout(btn_layout)

        # Status and Progress
        self.status_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["status_group"])
        status_layout = QVBoxLayout()
        self.status_label = QLabel(self.TRANSLATIONS[self.ui_lang]["status_ready"])
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        self.status_group.setLayout(status_layout)
        controls_layout.addWidget(self.status_group)

        # Output Text
        self.output_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["output_group"])
        output_layout = QVBoxLayout()
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMinimumHeight(150)
        output_layout.addWidget(self.text_output)
        self.output_group.setLayout(output_layout)
        controls_layout.addWidget(self.output_group)

        controls_layout.addStretch()
        
        # Set the controls widget as the scroll area's widget
        scroll_area.setWidget(controls_widget)

        # --- RIGHT PANEL: Image Preview ---
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel(self.TRANSLATIONS[self.ui_lang]["preview_label"])
        self.preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        preview_layout.addWidget(self.preview_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing)
        # Mouse Event Handling for Tools
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        preview_layout.addWidget(self.view)

        # Add to splitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(preview_widget)
        splitter.setSizes([460, 940])

        # Apply dark theme
        self.apply_dark_theme()


    def create_profile_group(self, layout):
        """Create Configuration Profile Management Group"""
        self.profile_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["profile_group"])
        p_layout = QVBoxLayout()
        
        # Language Selector Row (Global setting)
        lang_row = QHBoxLayout()
        self.lang_globe_btn = QPushButton("🌐")
        self.lang_globe_btn.setToolTip(self.TRANSLATIONS[self.ui_lang]["ui_lang_label"])
        self.lang_globe_btn.setFixedWidth(35)
        self.lang_globe_btn.setEnabled(False)  # Visual indicator only
        self.lang_globe_btn.setStyleSheet("font-size: 16px;")
        lang_row.addWidget(self.lang_globe_btn)
        self.lang_label = QLabel(self.TRANSLATIONS[self.ui_lang]["ui_lang_label"])
        lang_row.addWidget(self.lang_label)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("العربية", "ar")
        idx = self.lang_combo.findData(self.ui_lang)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.lang_combo.currentIndexChanged.connect(self.change_ui_language)
        lang_row.addWidget(self.lang_combo, 1)
        p_layout.addLayout(lang_row)

        p_layout.addWidget(self.create_separator())
        
        # Profile selector row
        selector_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)
        selector_row.addWidget(self.profile_combo, 1)

        # Action buttons
        self.btn_new_profile = QPushButton("📋")
        self.btn_new_profile.setToolTip(self.TRANSLATIONS[self.ui_lang]["btn_new_profile_tooltip"])
        self.btn_new_profile.setFixedWidth(35)
        self.btn_new_profile.clicked.connect(self.duplicate_profile)
        
        self.btn_rename_profile = QPushButton("✏️")
        self.btn_rename_profile.setToolTip(self.TRANSLATIONS[self.ui_lang]["btn_rename_profile_tooltip"])
        self.btn_rename_profile.setFixedWidth(35)
        self.btn_rename_profile.clicked.connect(self.rename_profile)

        self.btn_delete_profile = QPushButton("🗑️")
        self.btn_delete_profile.setToolTip(self.TRANSLATIONS[self.ui_lang]["btn_delete_profile_tooltip"])
        self.btn_delete_profile.setFixedWidth(35)
        self.btn_delete_profile.clicked.connect(self.delete_profile)
        
        selector_row.addWidget(self.btn_new_profile)
        selector_row.addWidget(self.btn_rename_profile)
        selector_row.addWidget(self.btn_delete_profile)
        p_layout.addLayout(selector_row)
        
        self.profile_group.setLayout(p_layout)
        layout.addWidget(self.profile_group)
        
        # Initialize profile list after buttons are created
        self.refresh_profile_list()

    def refresh_profile_list(self):
        """Refresh the profile combo box with current profiles"""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        data = load_profiles()
        # Always add Default first
        self.profile_combo.addItem("Default")
        # Add user profiles in sorted order
        for name in sorted(data["profiles"].keys()):
            self.profile_combo.addItem(name)
        
        # Set active profile
        active_name = data.get("active", "Default")
        idx = self.profile_combo.findText(active_name)
        self.profile_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.update_profile_buttons()
        self.profile_combo.blockSignals(False)

    def update_profile_buttons(self):
        """Update button states and lock UI based on current profile"""
        is_default = self.profile_combo.currentText() == "Default"
        self.btn_delete_profile.setEnabled(not is_default)
        self.btn_rename_profile.setEnabled(not is_default)
        
        # Show/Hide lock notice
        if hasattr(self, 'lock_notice'):
            self.lock_notice.setVisible(is_default)
        
        # Lock/Unlock all setting groups
        groups = [
            getattr(self, 'selection_group', None),
            getattr(self, 'hotkeys_group', None),
            getattr(self, 'image_settings_group', None),
            getattr(self, 'text_processing_group', None),
            getattr(self, 'tts_group', None)
        ]
        
        for group in groups:
            if group:
                group.setEnabled(not is_default)

    def on_profile_changed(self, name):
        """Handle profile selection change"""
        if not name:
            return
        
        data = load_profiles()
        data["active"] = name
        save_profiles(data)
        
        # Reload config for new profile
        self.config = load_config()
        
        # Update button states
        self.update_profile_buttons()
        
        # Refresh UI to reflect new config
        self.refresh_ui_from_config()
        
        # Re-trigger detection if image is loaded and RapidOCR is enabled
        if state.last_image and state.use_rapidocr:
            self.run_detection_preview()

    def duplicate_profile(self):
        """Duplicate the current profile with a new name"""
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        
        current_name = self.profile_combo.currentText()
        name, ok = QInputDialog.getText(
            self, 
            "Duplicate Profile", 
            f"Enter name for new profile (duplicated from '{current_name}'):",
            text=f"{current_name} Copy"
        )
        
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # Validate name
        if name == "Default":
            QMessageBox.warning(self, "Invalid Name", "Cannot use 'Default' as a profile name.")
            return
        
        data = load_profiles()
        if name in data["profiles"]:
            reply = QMessageBox.question(
                self,
                "Profile Exists",
                f"Profile '{name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Duplicate current config
        data["profiles"][name] = self.config.copy()
        data["active"] = name
        save_profiles(data)
        
        # Refresh UI
        self.refresh_profile_list()
        self.status_label.setText(f"Profile '{name}' created.")

    def rename_profile(self):
        """Rename the current profile"""
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        
        current_name = self.profile_combo.currentText()
        if current_name == "Default":
            QMessageBox.information(self, "Cannot Rename", "The 'Default' profile cannot be renamed.")
            return
        
        name, ok = QInputDialog.getText(
            self,
            "Rename Profile",
            f"Enter new name for '{current_name}':",
            text=current_name
        )
        
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # Validate name
        if name == "Default":
            QMessageBox.warning(self, "Invalid Name", "Cannot use 'Default' as a profile name.")
            return
        
        data = load_profiles()
        if name in data["profiles"]:
            QMessageBox.warning(self, "Name Exists", f"Profile '{name}' already exists.")
            return
        
        # Rename profile
        data["profiles"][name] = data["profiles"].pop(current_name)
        if data["active"] == current_name:
            data["active"] = name
        save_profiles(data)
        
        # Refresh UI
        self.refresh_profile_list()
        self.status_label.setText(f"Profile renamed to '{name}'.")

    def delete_profile(self):
        """Delete the current profile"""
        from PyQt6.QtWidgets import QMessageBox
        
        current_name = self.profile_combo.currentText()
        if current_name == "Default":
            QMessageBox.information(self, "Cannot Delete", "The 'Default' profile cannot be deleted.")
            return
        
        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Are you sure you want to delete profile '{current_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        data = load_profiles()
        if current_name in data["profiles"]:
            del data["profiles"][current_name]
            # If deleting active profile, switch to Default
            if data["active"] == current_name:
                data["active"] = "Default"
            save_profiles(data)
        
        # Refresh UI and reload Default config
        self.refresh_profile_list()
        self.on_profile_changed("Default")
        self.status_label.setText(f"Profile '{current_name}' deleted. Switched to Default.")


    def create_selection_group(self, layout):
        """Create Selection Tools Group"""
        self.selection_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["selection_group"])
        g_layout = QVBoxLayout()
        
        # 1. RapidOCR Toggle
        self.chk_rapid = QCheckBox(self.TRANSLATIONS[self.ui_lang]["chk_rapid"])
        self.chk_rapid.setChecked(state.use_rapidocr)
        self.chk_rapid.setToolTip(self.TRANSLATIONS[self.ui_lang]["chk_rapid_tooltip"])
        self.chk_rapid.toggled.connect(self.on_rapid_toggled)
        g_layout.addWidget(self.chk_rapid)
        
        # 2. Tools Toolbar
        tools_layout = QHBoxLayout()
        
        self.btn_tool_none = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_tool_none"])
        self.btn_tool_none.setCheckable(True)
        self.btn_tool_none.setChecked(True)
        self.btn_tool_none.clicked.connect(lambda: self.set_tool("none"))
        
        self.btn_tool_add = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_tool_add"])
        self.btn_tool_add.setCheckable(True)
        self.btn_tool_add.clicked.connect(lambda: self.set_tool("add"))
        
        self.btn_tool_sub = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_tool_sub"])
        self.btn_tool_sub.setCheckable(True)
        self.btn_tool_sub.clicked.connect(lambda: self.set_tool("sub"))
        
        self.btn_tool_manual = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_tool_manual"])
        self.btn_tool_manual.setCheckable(True)
        self.btn_tool_manual.clicked.connect(lambda: self.set_tool("manual"))
        
        # Group for exclusive checking
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_tool_none)
        self.tool_group.addButton(self.btn_tool_add)
        self.tool_group.addButton(self.btn_tool_sub)
        self.tool_group.addButton(self.btn_tool_manual)
        
        tools_layout.addWidget(self.btn_tool_none)
        tools_layout.addWidget(self.btn_tool_add)
        tools_layout.addWidget(self.btn_tool_sub)
        tools_layout.addWidget(self.btn_tool_manual)
        g_layout.addLayout(tools_layout)
        
        # 3. Actions
        actions_layout = QHBoxLayout()
        
        self.btn_sel_all = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_sel_all"])
        self.btn_sel_all.clicked.connect(self.select_all)
        
        self.btn_desel_all = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_desel_all"])
        self.btn_desel_all.clicked.connect(self.deselect_all)
        
        self.btn_clear_manual = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_clear_manual"])
        self.btn_clear_manual.clicked.connect(self.clear_manual_boxes)
        self.btn_clear_manual.setStyleSheet("color: #ff9999;")  # Reddish text hint
        
        actions_layout.addWidget(self.btn_sel_all)
        actions_layout.addWidget(self.btn_desel_all)
        actions_layout.addWidget(self.btn_clear_manual)
        g_layout.addLayout(actions_layout)
        
        self.selection_group.setLayout(g_layout)
        layout.addWidget(self.selection_group)

    def create_hotkeys_group(self, layout):
        """Create Hotkeys Configuration Group"""
        self.hotkeys_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["hotkeys_group"])
        g_layout = QVBoxLayout()
        
        hotkeys = self.config.get("hotkeys", {
            "extract": "ctrl+shift+alt+z",
            "replay": "ctrl+shift+alt+x",
            "detect": "ctrl+shift+alt+m"
        })
        
        # Extract Hotkey
        self.btn_hk_extract = HotkeyRecorder("extract", hotkeys.get("extract", "ctrl+shift+alt+z"))
        self.btn_hk_extract.hotkeyChanged.connect(self.update_hotkey)
        self.btn_hk_extract.setToolTip("Full Process: Screenshot -> Detect -> Extract -> Read")
        g_layout.addWidget(self.btn_hk_extract)

        # Replay Hotkey
        self.btn_hk_replay = HotkeyRecorder("replay", hotkeys.get("replay", "ctrl+shift+alt+x"))
        self.btn_hk_replay.hotkeyChanged.connect(self.update_hotkey)
        self.btn_hk_replay.setToolTip("Replay last spoken text")
        g_layout.addWidget(self.btn_hk_replay)

        # Detect Hotkey
        self.btn_hk_detect = HotkeyRecorder("detect", hotkeys.get("detect", "ctrl+shift+alt+m"))
        self.btn_hk_detect.hotkeyChanged.connect(self.update_hotkey)
        self.btn_hk_detect.setToolTip("Preview: Screenshot -> Detect Only")
        g_layout.addWidget(self.btn_hk_detect)
        
        # Store widgets for profile refresh
        self._config_widgets["hotkeys.extract"] = (None, None, self.btn_hk_extract)
        self._config_widgets["hotkeys.replay"] = (None, None, self.btn_hk_replay)
        self._config_widgets["hotkeys.detect"] = (None, None, self.btn_hk_detect)

        self.hotkeys_group.setLayout(g_layout)
        layout.addWidget(self.hotkeys_group)

    def update_hotkey(self, key_type, new_hotkey):
        """Update hotkey in config and re-register listeners"""
        if "hotkeys" not in self.config:
            self.config["hotkeys"] = {}
        
        self.config["hotkeys"][key_type] = new_hotkey
        self.save_config()
        
        # Re-register hotkeys
        self.setup_hotkey_listener()

    def on_rapid_toggled(self, checked):
        state.use_rapidocr = checked
        if state.last_image:
            self.run_detection_preview()

    def set_tool(self, mode):
        self.tool_mode = mode
        if mode == "none":
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)

    def select_all(self):
        state.selection_ops = []
        state.selection_base_state = True
        self.update_selection_overlay()
        self.run_detection_preview()

    def deselect_all(self):
        state.selection_ops = []
        state.selection_base_state = False
        self.update_selection_overlay()
        self.run_detection_preview()

    def clear_manual_boxes(self):
        """Clear all manually drawn boxes"""
        state.manual_boxes = []
        self.save_manual_boxes()
        self.draw_manual_boxes()
        if state.last_image:
            self.run_detection_preview()

    def save_manual_boxes(self):
        """Save manual boxes to config file"""
        # Save as list of lists for JSON compatibility
        self.config["manual_boxes"] = [list(b) for b in state.manual_boxes]
        self.save_config()

    def update_selection_overlay(self):
        """Draw semi-transparent overlay to indicate selection state"""
        if not self.pixmap_item:
            return
            
        # Remove old overlay
        if self.selection_overlay_item:
            self.scene.removeItem(self.selection_overlay_item)
            self.selection_overlay_item = None
            
        img_w = self.pixmap_item.pixmap().width()
        img_h = self.pixmap_item.pixmap().height()
        
        # Generate mask
        mask = generate_selection_mask((img_h, img_w), state.selection_ops, state.selection_base_state)
        if mask is None:
            return

        # Convert mask to QImage
        import numpy as np
        overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        
        # Set alpha: 150 where mask is 0, 0 where mask is 255
        overlay[mask == 0, 3] = 150  # Darken excluded
        overlay[mask == 255, 3] = 0  # Transparent included
        
        qimg = QImage(overlay.data, img_w, img_h, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        
        self.selection_overlay_item = QGraphicsPixmapItem(pix)
        self.selection_overlay_item.setZValue(50)  # Above image, below boxes
        self.scene.addItem(self.selection_overlay_item)


    def create_image_settings_group(self, layout):
        """Combined Image Settings (Size + Preprocessing)"""
        self.image_settings_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["img_adj_group"])
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
        self.label_max_dimension = QLabel(self.TRANSLATIONS[self.ui_lang]["label_max_dimension"])
        dim_row.addWidget(self.label_max_dimension)
        
        dim_slider = QSlider(Qt.Orientation.Horizontal)
        dim_slider.setRange(320, 2560)
        dim_slider.setValue(current_dim)
        
        dim_spin = QSpinBox()
        dim_spin.setRange(1, 99999)
        dim_spin.setValue(current_dim)
        dim_spin.setSuffix(" px")
        dim_spin.setMaximumWidth(70)
        
        # Store references for profile refresh
        self._config_widgets["max_image_dimension"] = (dim_slider, dim_spin, None)  # (slider, spinbox, combo)

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
            label_widget = QLabel(label)
            row.addWidget(label_widget)
            
            # Store label reference for language switching
            label_key = f"label_{key}" if key != "dilation" else "label_dilation"
            if not hasattr(self, '_preprocessing_labels'):
                self._preprocessing_labels = {}
            self._preprocessing_labels[label_key] = label_widget
            
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
            
            # Store references for profile refresh
            self._config_widgets[f"preprocessing.{key}"] = (slider, spin, None)

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
        self.chk_invert = QComboBox()
        self.chk_invert.addItem(self.TRANSLATIONS[self.ui_lang]["colors_normal"], False)
        self.chk_invert.addItem(self.TRANSLATIONS[self.ui_lang]["colors_invert"], True)
        self.chk_invert.setCurrentIndex(1 if pp_config.get("invert", False) else 0)
        self.chk_invert.currentIndexChanged.connect(lambda: self.update_pp("invert", bool(self.chk_invert.currentData())))
        
        # Store reference for profile refresh
        self._config_widgets["preprocessing.invert"] = (None, None, self.chk_invert)
        inv_layout = QHBoxLayout()
        inv_layout.setSpacing(8)
        self.label_colors = QLabel(self.TRANSLATIONS[self.ui_lang]["label_colors"])
        inv_layout.addWidget(self.label_colors)
        inv_layout.addWidget(self.chk_invert)
        
        # Reset button for Colors
        def reset_invert():
            default_val = self.DEFAULT_CONFIG["preprocessing"].get("invert", False)
            self.chk_invert.setCurrentIndex(1 if default_val else 0)
            self.update_pp("invert", default_val)
        inv_reset = self.create_reset_button(reset_invert)
        inv_layout.addWidget(inv_reset)
        
        g_layout.addLayout(inv_layout)

        # 2. Threshold
        add_slider_row(self.TRANSLATIONS[self.ui_lang]["label_binary_threshold"], "binary_threshold", 0, 255, 0)
        
        # 3. Contrast
        add_slider_row(self.TRANSLATIONS[self.ui_lang]["label_contrast"], "contrast", 0.5, 3.0, 1.0, scale=10.0, is_float=True)
        
        # 4. Brightness
        add_slider_row(self.TRANSLATIONS[self.ui_lang]["label_brightness"], "brightness", -100, 100, 0)
        
        # 5. Dilation/Erosion (Thicken/Thin text)
        # Negative values thin text, positive values thicken text
        add_slider_row(self.TRANSLATIONS[self.ui_lang]["label_dilation"], "dilation", -5, 5, 0)

        self.image_settings_group.setLayout(g_layout)
        layout.addWidget(self.image_settings_group)

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
        btn.setToolTip(self.TRANSLATIONS[self.ui_lang]["reset_tooltip"])
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
        self.text_processing_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["text_proc_group"])
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
        self.legend = QLabel(self.TRANSLATIONS[self.ui_lang]["legend_text"])
        self.legend.setStyleSheet("color: #aaa; font-size: 10px;")
        viz_box.addWidget(self.legend)
        
        g_layout.addWidget(viz_container)
        g_layout.addSpacing(10)

        # --- Helper for sections ---
        def add_header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 5px;")
            g_layout.addWidget(lbl)

        # --- Section 1: Detection (Size) ---
        self.detection_header_label = QLabel(self.TRANSLATIONS[self.ui_lang]["detection_header"])
        self.detection_header_label.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 5px;")
        g_layout.addWidget(self.detection_header_label)
        
        # Info label explaining adaptive logic
        self.detection_info_label = QLabel(self.TRANSLATIONS[self.ui_lang]["detection_info"])
        self.detection_info_label.setStyleSheet("color: #4CAF50; font-size: 9px; font-style: italic; padding: 3px;")
        g_layout.addWidget(self.detection_info_label)
        
        # GPU Acceleration Checkbox
        gpu_row = QHBoxLayout()
        self.chk_gpu = QCheckBox(self.TRANSLATIONS[self.ui_lang]["chk_gpu"])
        gpu_val = td_config.get("use_gpu", False)
        self.chk_gpu.setChecked(gpu_val)
        self.chk_gpu.setToolTip(self.TRANSLATIONS[self.ui_lang]["chk_gpu_tooltip"])
        
        def on_gpu_toggled(checked):
            if "text_detection" not in self.config:
                self.config["text_detection"] = {}
            self.config["text_detection"]["use_gpu"] = checked
            self.save_and_refresh()
            # Force reinitialization of RapidOCR with new GPU setting
            from src.backend.core.detection import _rapidocr_instance, _rapidocr_use_gpu
            import src.backend.core.detection as detection_module
            detection_module._rapidocr_instance = None
            detection_module._rapidocr_use_gpu = None
        
        self.chk_gpu.toggled.connect(on_gpu_toggled)
        gpu_row.addWidget(self.chk_gpu)
        gpu_row.addStretch()
        g_layout.addLayout(gpu_row)
        g_layout.addSpacing(5)
        
        # Store reference for profile refresh
        self._config_widgets["text_detection.use_gpu"] = (None, None, self.chk_gpu)
        
        def update_viz_detection():
            # For visualization, estimate pixels from ratios (assume 1920x1080 screen)
            # This is just for the visualizer widget, actual filtering uses ratios
            min_h_ratio = self.config["text_detection"].get("min_height_ratio", 0.031)
            h = int(1080 * min_h_ratio)  # Estimate for visualization
            w = h  # Use same for width
            self.settings_viz.update_detection(w, h)

        # Min Height Ratio (as % of screen height)
        h_ratio_row = QHBoxLayout()
        h_ratio_row.setSpacing(8)
        self.label_min_height_ratio = QLabel(self.TRANSLATIONS[self.ui_lang]["label_min_height_ratio"])
        h_ratio_row.addWidget(self.label_min_height_ratio)
        h_ratio_sl = QSlider(Qt.Orientation.Horizontal)
        h_ratio_sl.setRange(0, 50)  # 0% to 5% of screen height
        h_ratio_val = td_config.get("min_height_ratio", 0.031)
        h_ratio_sl.setValue(int(h_ratio_val * 1000))  # Convert to per-mille
        h_ratio_sp = QDoubleSpinBox()
        h_ratio_sp.setRange(-999999.0, 999999.0)  # Allow any value
        h_ratio_sp.setSingleStep(0.001)
        h_ratio_sp.setValue(h_ratio_val)
        h_ratio_sp.setSuffix(" %")
        h_ratio_sp.setMaximumWidth(70)
        
        # Store references for profile refresh
        self._config_widgets["text_detection.min_height_ratio"] = (h_ratio_sl, h_ratio_sp, None)

        def on_h_ratio_change(v):
            real_val = v / 1000.0 if isinstance(v, int) else v
            if "text_detection" not in self.config: self.config["text_detection"] = {}
            self.config["text_detection"]["min_height_ratio"] = real_val
            # Only update slider if value is within slider range, otherwise just update config
            if 0 <= real_val <= 0.05:
                h_ratio_sl.blockSignals(True); h_ratio_sl.setValue(int(real_val * 1000)); h_ratio_sl.blockSignals(False)
            h_ratio_sp.blockSignals(True); h_ratio_sp.setValue(real_val); h_ratio_sp.blockSignals(False)
            update_viz_detection()
            self.preview_live_filtering()
        
        h_ratio_sl.valueChanged.connect(on_h_ratio_change)
        h_ratio_sl.sliderReleased.connect(self.finalize_live_preview)
        h_ratio_sp.valueChanged.connect(on_h_ratio_change)
        h_ratio_sp.editingFinished.connect(self.finalize_live_preview)
        
        def reset_h_ratio():
            default_val = self.DEFAULT_CONFIG["text_detection"].get("min_height_ratio", 0.031)
            h_ratio_sl.setValue(int(default_val * 1000))
            h_ratio_sp.setValue(default_val)
            on_h_ratio_change(default_val)
        h_ratio_reset = self.create_reset_button(reset_h_ratio)
        
        h_ratio_row.addWidget(h_ratio_sl); h_ratio_row.addWidget(h_ratio_sp); h_ratio_row.addWidget(h_ratio_reset)
        g_layout.addLayout(h_ratio_row)

        # Median Height Fraction (noise filter)
        median_row = QHBoxLayout()
        median_row.setSpacing(8)
        self.label_noise_filter = QLabel(self.TRANSLATIONS[self.ui_lang]["label_noise_filter"])
        median_row.addWidget(self.label_noise_filter)
        median_sl = QSlider(Qt.Orientation.Horizontal)
        median_sl.setRange(10, 100)  # 0.1 to 1.0
        median_val = td_config.get("median_height_fraction", 1.0)
        median_sl.setValue(int(median_val * 100))
        median_sp = QDoubleSpinBox()
        median_sp.setRange(-999999.0, 999999.0)  # Allow any value
        median_sp.setSingleStep(0.05)
        median_sp.setValue(median_val)
        median_sp.setMaximumWidth(70)
        
        # Store references for profile refresh
        self._config_widgets["text_detection.median_height_fraction"] = (median_sl, median_sp, None)

        def on_median_change(v):
            real_val = v / 100.0 if isinstance(v, int) else v
            if "text_detection" not in self.config: self.config["text_detection"] = {}
            self.config["text_detection"]["median_height_fraction"] = real_val
            # Only update slider if value is within slider range, otherwise just update config
            if 0.1 <= real_val <= 1.0:
                median_sl.blockSignals(True); median_sl.setValue(int(real_val * 100)); median_sl.blockSignals(False)
            median_sp.blockSignals(True); median_sp.setValue(real_val); median_sp.blockSignals(False)
            self.preview_live_filtering()
        
        median_sl.valueChanged.connect(on_median_change)
        median_sl.sliderReleased.connect(self.finalize_live_preview)
        median_sp.valueChanged.connect(on_median_change)
        median_sp.editingFinished.connect(self.finalize_live_preview)
        
        def reset_median():
            default_val = self.DEFAULT_CONFIG["text_detection"].get("median_height_fraction", 1.0)
            median_sl.setValue(int(default_val * 100))
            median_sp.setValue(default_val)
            on_median_change(default_val)
        median_reset = self.create_reset_button(reset_median)
        
        median_row.addWidget(median_sl); median_row.addWidget(median_sp); median_row.addWidget(median_reset)
        g_layout.addLayout(median_row)

        g_layout.addWidget(self.create_separator())

        # --- Section 2: Merging ---
        self.merging_header_label = QLabel(self.TRANSLATIONS[self.ui_lang]["merging_header"])
        self.merging_header_label.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 5px;")
        g_layout.addWidget(self.merging_header_label)
        
        # Reading Direction
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
        dir_row = QHBoxLayout()
        dir_row.setSpacing(8)
        self.label_order = QLabel(self.TRANSLATIONS[self.ui_lang]["label_order"])
        dir_row.addWidget(self.label_order)
        self.order_combo = QComboBox()
        self.order_combo.addItem(self.TRANSLATIONS[self.ui_lang]["order_ltr"], "horizontal_ltr")
        self.order_combo.addItem(self.TRANSLATIONS[self.ui_lang]["order_rtl"], "horizontal_rtl")
        self.order_combo.addItem(self.TRANSLATIONS[self.ui_lang]["order_vertical_rtl"], "vertical_rtl")
        self.order_combo.addItem(self.TRANSLATIONS[self.ui_lang]["order_vertical_ltr"], "vertical_ltr")
        
        cur_dir = sort_config.get("direction") or "horizontal_ltr"
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

        # Viz Update Logic for Merge (uses ratios now)
        def update_viz_merge():
            # For visualization, estimate pixel values from ratios (assume 20px text height)
            v_ratio = self.config["text_detection"].get("merge_vertical_ratio", 0.07)
            h_ratio = self.config["text_detection"].get("merge_horizontal_ratio", 0.37)
            rat = self.config["text_detection"].get("merge_width_ratio_threshold", 0.75)
            # Estimate pixels for visualization (assume 20px text height)
            vt = int(v_ratio * 20)
            ht = int(h_ratio * 20)
            self.settings_viz.update_merge(vt, ht, rat)

        # Helper for merge ratio sliders
        def add_merge_ratio_slider(label_key, tooltip_key, key, default, max_val=5.0):
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(self.TRANSLATIONS[self.ui_lang][label_key])
            if tooltip_key:
                lbl.setToolTip(self.TRANSLATIONS[self.ui_lang][tooltip_key])
            # Store label reference for translation updates
            setattr(self, label_key, lbl)
            row.addWidget(lbl)
            
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, int(max_val * 100))  # 0.0 to max_val
            val = td_config.get(key, default)
            sl.setValue(int(val * 100))
            
            sp = QDoubleSpinBox()
            sp.setRange(-999999.0, 999999.0)  # Allow any value
            sp.setSingleStep(0.1)
            sp.setValue(val)
            sp.setSuffix("x")
            sp.setMaximumWidth(70)
            
            # Store references for profile refresh
            self._config_widgets[f"text_detection.{key}"] = (sl, sp, None)
            
            def on_change(v):
                real_val = v / 100.0 if isinstance(v, int) else v
                if "text_detection" not in self.config: self.config["text_detection"] = {}
                self.config["text_detection"][key] = real_val
                
                # Only update slider if value is within slider range, otherwise just update config
                if 0.0 <= real_val <= max_val:
                    sl.blockSignals(True); sl.setValue(int(real_val * 100)); sl.blockSignals(False)
                sp.blockSignals(True); sp.setValue(real_val); sp.blockSignals(False)
                
                update_viz_merge()
                self.preview_live_merging()
            
            sl.valueChanged.connect(on_change)
            sp.valueChanged.connect(on_change)
            sl.sliderReleased.connect(self.finalize_live_preview)
            sp.editingFinished.connect(self.finalize_live_preview)
            
            def reset_val():
                default_val = self.DEFAULT_CONFIG["text_detection"].get(key, default)
                sl.setValue(int(default_val * 100))
                sp.setValue(default_val)
                on_change(default_val)
            reset_btn = self.create_reset_button(reset_val)
            
            row.addWidget(sl)
            row.addWidget(sp)
            row.addWidget(reset_btn)
            g_layout.addLayout(row)

        add_merge_ratio_slider("label_v_ratio", "tooltip_v_ratio", "merge_vertical_ratio", 0.07, max_val=2.0)
        add_merge_ratio_slider("label_h_ratio", "tooltip_h_ratio", "merge_horizontal_ratio", 0.37, max_val=5.0)
        add_merge_ratio_slider("label_width_ratio", "tooltip_width_ratio", "merge_width_ratio_threshold", 0.75, max_val=1.0)

        # Line Grouping (Sorting)
        grp_row = QHBoxLayout()
        grp_row.setSpacing(8)
        self.label_line_grouping = QLabel(self.TRANSLATIONS[self.ui_lang]["label_line_grouping"])
        grp_row.addWidget(self.label_line_grouping)
        g_sl = QSlider(Qt.Orientation.Horizontal)
        g_sl.setRange(1, 20)
        grp_val = sort_config.get("group_tolerance", 0.5)
        g_sl.setValue(int(grp_val * 10))
        g_sp = QDoubleSpinBox()
        g_sp.setRange(0.1, 2.0)
        g_sp.setSingleStep(0.1)
        g_sp.setValue(grp_val)
        g_sp.setMaximumWidth(70)
        
        # Store references for profile refresh
        self._config_widgets["text_sorting.group_tolerance"] = (g_sl, g_sp, None)
        
        def on_grp_change(v):
            # v can be int (slider) or float (spinbox)
            real_val = v / 10.0 if isinstance(v, int) else v
            if "text_sorting" not in self.config: self.config["text_sorting"] = {}
            self.config["text_sorting"]["group_tolerance"] = real_val
            
            # Only update slider if value is within slider range, otherwise just update config
            if 0.1 <= real_val <= 2.0:
                g_sl.blockSignals(True); g_sl.setValue(int(real_val*10)); g_sl.blockSignals(False)
            g_sp.blockSignals(True); g_sp.setValue(real_val); g_sp.blockSignals(False)
            self.preview_live_merging()
            
        g_sl.valueChanged.connect(on_grp_change)
        g_sl.sliderReleased.connect(self.finalize_live_preview)
        g_sp.valueChanged.connect(on_grp_change)
        g_sp.editingFinished.connect(self.finalize_live_preview)
        
        # Reset button
        def reset_grp():
            default_val = self.DEFAULT_CONFIG["text_sorting"].get("group_tolerance", 0.5)
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

        self.text_processing_group.setLayout(g_layout)
        layout.addWidget(self.text_processing_group)

    def create_tts_group(self, layout):
        """Text-to-Speech Settings with Voice Selection and Media Player"""
        self.tts_group = QGroupBox(self.TRANSLATIONS[self.ui_lang]["tts_group"])
        g_layout = QVBoxLayout()
        tts_config = self.config.get("tts", {})

        # Info label (TTS is always enabled)
        self.tts_info_label = QLabel(self.TRANSLATIONS[self.ui_lang]["tts_info"])
        self.tts_info_label.setStyleSheet("color: #4CAF50; font-style: italic; padding: 5px;")
        g_layout.addWidget(self.tts_info_label)

        # Store full voice list for filtering (dynamically fetched from HuggingFace)
        from src.backend.core.tts import get_available_voices
        self.all_voices = get_available_voices()  # Dynamically fetch available voices

        # 0. Voice Filters Row
        filter_row = QHBoxLayout()
        self.label_language_filter = QLabel(self.TRANSLATIONS[self.ui_lang]["label_language"])
        filter_row.addWidget(self.label_language_filter)
        self.lang_filter = QComboBox()
        self.lang_filter.addItems(["All", "🇺🇸 American", "🇬🇧 British"])
        self.lang_filter.currentIndexChanged.connect(self.filter_voices)
        filter_row.addWidget(self.lang_filter)
        
        self.label_gender_filter = QLabel(self.TRANSLATIONS[self.ui_lang]["label_gender"])
        filter_row.addWidget(self.label_gender_filter)
        self.gender_filter = QComboBox()
        self.gender_filter.addItems(["All", "Male", "Female"])
        self.gender_filter.currentIndexChanged.connect(self.filter_voices)
        filter_row.addWidget(self.gender_filter)
        g_layout.addLayout(filter_row)

        # 1. Voice Selection Row
        voice_row = QHBoxLayout()
        self.label_voice = QLabel(self.TRANSLATIONS[self.ui_lang]["label_voice"])
        voice_row.addWidget(self.label_voice)
        self.voice_combo = QComboBox()
        # Initialize voice combo with all voices first
        for name, code in self.all_voices.items():
            self.voice_combo.addItem(name, code)
        
        # Set current voice from config
        current_voice_code = tts_config.get("voice", "af_heart")
        index = self.voice_combo.findData(current_voice_code)
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)
        else:
            # Fallback: set to first item if voice not found
            self.voice_combo.setCurrentIndex(0)
        
        self.voice_combo.currentIndexChanged.connect(self.save_tts_settings)
        voice_row.addWidget(self.voice_combo, 1)
        g_layout.addLayout(voice_row)

        # 2. Speed Slider
        speed_row = QHBoxLayout()
        self.label_speed = QLabel(self.TRANSLATIONS[self.ui_lang]["label_speed"])
        speed_row.addWidget(self.label_speed)
        
        self.sl_speed = QSlider(Qt.Orientation.Horizontal)
        self.sl_speed.setRange(5, 20)  # 0.5x to 2.0x
        current_speed = tts_config.get("speed", 1.0)
        self.sl_speed.setValue(int(current_speed * 10))
        
        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(0.5, 2.0)
        self.sp_speed.setSingleStep(0.1)
        self.sp_speed.setValue(current_speed)
        self.sp_speed.setMaximumWidth(60)

        def on_speed(v):
            real_val = v / 10.0 if isinstance(v, int) else v
            if "tts" not in self.config: self.config["tts"] = {}
            self.config["tts"]["speed"] = real_val
            
            self.sl_speed.blockSignals(True); self.sl_speed.setValue(int(real_val*10)); self.sl_speed.blockSignals(False)
            self.sp_speed.blockSignals(True); self.sp_speed.setValue(real_val); self.sp_speed.blockSignals(False)
            self.save_config()

        self.sl_speed.valueChanged.connect(on_speed)
        self.sp_speed.valueChanged.connect(on_speed)

        speed_row.addWidget(self.sl_speed)
        speed_row.addWidget(self.sp_speed)
        g_layout.addLayout(speed_row)

        # 2b. Volume Slider
        volume_row = QHBoxLayout()
        self.label_volume = QLabel(self.TRANSLATIONS[self.ui_lang]["label_volume"])
        volume_row.addWidget(self.label_volume)
        
        self.sl_volume = QSlider(Qt.Orientation.Horizontal)
        self.sl_volume.setRange(0, 200)  # 0% to 200%
        current_volume = tts_config.get("volume", 1.0) * 100  # Convert 0.0-2.0 to 0-200
        self.sl_volume.setValue(int(current_volume))
        
        self.sp_volume = QSpinBox()
        self.sp_volume.setRange(0, 200)
        self.sp_volume.setSuffix("%")
        self.sp_volume.setValue(int(current_volume))
        self.sp_volume.setMaximumWidth(70)

        def on_volume(v):
            real_val = v / 100.0 if isinstance(v, int) else v  # Convert to 0.0-2.0
            if "tts" not in self.config: self.config["tts"] = {}
            self.config["tts"]["volume"] = real_val
            
            self.sl_volume.blockSignals(True); self.sl_volume.setValue(int(real_val*100)); self.sl_volume.blockSignals(False)
            self.sp_volume.blockSignals(True); self.sp_volume.setValue(int(real_val*100)); self.sp_volume.blockSignals(False)
            self.save_config()

        self.sl_volume.valueChanged.connect(on_volume)
        self.sp_volume.valueChanged.connect(on_volume)

        volume_row.addWidget(self.sl_volume)
        volume_row.addWidget(self.sp_volume)
        g_layout.addLayout(volume_row)

        # 3. Media Player Controls
        player_layout = QHBoxLayout()
        self.btn_play_new = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_play"])
        self.btn_play_new.clicked.connect(self.play_tts_with_settings)
        self.btn_play_new.setToolTip(self.TRANSLATIONS[self.ui_lang]["btn_play_tooltip"])
        
        self.btn_replay = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_replay"])
        self.btn_replay.clicked.connect(self.replay_audio)
        self.btn_replay.setToolTip(self.TRANSLATIONS[self.ui_lang]["btn_replay_tooltip"])
        
        self.btn_stop_audio = QPushButton(self.TRANSLATIONS[self.ui_lang]["btn_stop"])
        self.btn_stop_audio.clicked.connect(self.stop_audio)
        self.btn_stop_audio.setToolTip(self.TRANSLATIONS[self.ui_lang]["btn_stop_tooltip"])
        
        player_layout.addWidget(self.btn_play_new)
        player_layout.addWidget(self.btn_replay)
        player_layout.addWidget(self.btn_stop_audio)
        g_layout.addLayout(player_layout)

        # 4. Phoneme Display (IPA tokens)
        self.label_phonemes = QLabel(self.TRANSLATIONS[self.ui_lang]["label_phonemes"])
        self.label_phonemes.setStyleSheet("color: #aaa; font-size: 11px;")
        g_layout.addWidget(self.label_phonemes)
        
        self.phoneme_output = QLineEdit()
        self.phoneme_output.setReadOnly(True)
        self.phoneme_output.setPlaceholderText(self.TRANSLATIONS[self.ui_lang]["phonemes_placeholder"])
        self.phoneme_output.setStyleSheet("background: #111; color: #888; font-family: monospace; font-size: 10px; padding: 4px;")
        g_layout.addWidget(self.phoneme_output)

        self.tts_group.setLayout(g_layout)
        layout.addWidget(self.tts_group)

    def populate_voice_combo(self):
        """Populate voice combo box based on current filters"""
        # Get current selection to preserve it if possible
        current_voice_code = None
        if hasattr(self, 'voice_combo') and self.voice_combo.count() > 0:
            current_voice_code = self.voice_combo.currentData()
        
        # Clear and repopulate
        self.voice_combo.clear()
        
        # Get filter values
        lang_filter = self.lang_filter.currentText()
        gender_filter = self.gender_filter.currentText()
        
        # Filter voices
        filtered_voices = {}
        for name, code in self.all_voices.items():
            # Language filter
            if lang_filter == "🇺🇸 American" and not code.startswith(('af_', 'am_')):
                continue
            if lang_filter == "🇬🇧 British" and not code.startswith(('bf_', 'bm_')):
                continue
            
            # Gender filter
            if gender_filter == "Male" and not code.startswith(('am_', 'bm_')):
                continue
            if gender_filter == "Female" and not code.startswith(('af_', 'bf_')):
                continue
            
            filtered_voices[name] = code
        
        # Add filtered voices to combo
        for name, code in filtered_voices.items():
            self.voice_combo.addItem(name, code)
        
        # Restore previous selection if it's still available
        if current_voice_code:
            index = self.voice_combo.findData(current_voice_code)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
            elif self.voice_combo.count() > 0:
                # If previous selection not available, select first item
                self.voice_combo.setCurrentIndex(0)
                # Only save if we're not in the middle of initialization
                if hasattr(self, 'config'):
                    self.save_tts_settings()  # Save new selection
        elif self.voice_combo.count() > 0:
            # If no previous selection, select first item
            self.voice_combo.setCurrentIndex(0)

    def filter_voices(self):
        """Filter voice list based on language and gender filters"""
        self.populate_voice_combo()

    def save_tts_settings(self):
        """Save current GUI TTS settings to config"""
        if "tts" not in self.config:
            self.config["tts"] = {}
        self.config["tts"]["voice"] = self.voice_combo.currentData()
        self.config["tts"]["speed"] = self.sp_speed.value()
        self.config["tts"]["volume"] = self.sp_volume.value() / 100.0  # Convert % to 0.0-2.0
        self.save_config()

    def play_tts_with_settings(self):
        """Re-generate TTS with current voice/speed settings"""
        from src.backend.core.tts import speak_text, stop_tts_engine
        
        # Get text from text output
        text = self.text_output.toPlainText().strip()
        if not text:
            # Fallback to state
            text = state.last_extracted_text if hasattr(state, 'last_extracted_text') else ""
        
        if not text:
            self.status_label.setText("No text to read - capture and extract text first")
            return
        
        # Stop any current playback
        stop_tts_engine()
        
        # Queue new TTS with current settings
        speak_text(text, clear_queue=True)
        self.status_label.setText(f"Generating TTS with {self.voice_combo.currentText()}...")

    def replay_audio(self):
        """Replay the last generated audio from state (with current volume)"""
        import sounddevice as sd
        import numpy as np
        if hasattr(state, 'last_audio_data') and state.last_audio_data is not None:
            try:
                # Get current volume setting
                volume = self.config.get("tts", {}).get("volume", 1.0)
                # Apply volume to stored audio
                audio_with_volume = state.last_audio_data * volume
                sd.play(audio_with_volume, 24000)
                self.status_label.setText("Replaying audio...")
            except Exception as e:
                print(f"[TTS] Replay error: {e}")
                self.status_label.setText("Replay failed - no audio data")
        else:
            self.status_label.setText("No audio to replay")

    def stop_audio(self):
        """Stop current audio playback"""
        import sounddevice as sd
        try:
            sd.stop()
            self.status_label.setText("Audio stopped")
        except Exception as e:
            print(f"[TTS] Stop error: {e}")

    def play_beep(self, beep_type="success"):
        """
        Play a beep sound to provide audio feedback.
        
        Args:
            beep_type: "success" for acknowledgment beep, "error" for error beep
        """
        print(f"[Beep] play_beep called: type={beep_type}")
        def _play_beep_thread():
            """Play beep in a separate thread to avoid blocking"""
            try:
                # Get volume from TTS config (same as TTS volume)
                volume = self.config.get("tts", {}).get("volume", 1.0)
                print(f"[Beep] Thread started: Playing {beep_type} beep (volume={volume})")
                
                # Beep parameters
                sample_rate = 44100
                if beep_type == "success":
                    # Success beep: low pitch, short duration (user will hear this often)
                    frequency = 250  # Hz (low pitch but audible)
                    duration = 0.2  # seconds (short but audible)
                else:  # error
                    # Error beep: slightly higher pitch, still short
                    frequency = 400  # Hz
                    duration = 0.2  # seconds
                
                # Generate sine wave
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                # Apply envelope to avoid clicks (fade in/out)
                envelope = np.ones_like(t)
                fade_samples = int(sample_rate * 0.02)  # 20ms fade
                if fade_samples > 0 and len(envelope) > fade_samples * 2:
                    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
                
                # Generate tone with envelope
                # Use 0.5 base amplitude (50%) for audible but subtle beep, then apply volume (0.0-2.0)
                # Ensure volume is at least 0.25 to be audible even at low settings
                effective_volume = max(0.25, volume) * 0.5
                tone = np.sin(2 * np.pi * frequency * t) * envelope * effective_volume
                
                # Ensure proper format (mono, float32)
                tone = tone.astype(np.float32)
                
                # Check sounddevice is available
                try:
                    default_device = sd.query_devices(kind='output')
                    print(f"[Beep] Using audio device: {default_device['name']}")
                except Exception as dev_e:
                    print(f"[Beep] Warning: Could not query audio device: {dev_e}")
                
                # Play beep and wait for completion
                print(f"[Beep] Playing tone: {len(tone)} samples, {duration:.2f}s, {frequency}Hz, volume={effective_volume:.2f}")
                sd.play(tone, sample_rate)
                sd.wait()  # Wait until playback is finished
                print(f"[Beep] ✓ {beep_type} beep completed")
            except Exception as e:
                print(f"[Beep] ✗ Error playing beep: {e}")
                import traceback
                traceback.print_exc()
        
        # Play beep in a separate thread to avoid blocking the UI
        beep_thread = threading.Thread(target=_play_beep_thread, daemon=True, name=f"BeepThread-{beep_type}")
        beep_thread.start()
        print(f"[Beep] Thread started: {beep_thread.name}")

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

    def refresh_ui_from_config(self):
        """Refresh all UI elements to reflect current config values"""
        # Update resize visualizer
        if hasattr(self, 'resize_viz'):
            max_dim = self.config.get("max_image_dimension", 1080)
            self.resize_viz.update_value(max_dim)
        
        # Update max_image_dimension
        if "max_image_dimension" in self._config_widgets:
            slider, spinbox, _ = self._config_widgets["max_image_dimension"]
            if slider and spinbox:
                slider.blockSignals(True)
                spinbox.blockSignals(True)
                slider.setValue(max_dim)
                spinbox.setValue(max_dim)
                slider.blockSignals(False)
                spinbox.blockSignals(False)
        
        # Update preprocessing widgets
        pp_config = self.config.get("preprocessing", {})
        for key in ["binary_threshold", "contrast", "brightness", "dilation"]:
            widget_key = f"preprocessing.{key}"
            if widget_key in self._config_widgets:
                slider, spinbox, _ = self._config_widgets[widget_key]
                if slider and spinbox:
                    val = pp_config.get(key, 0)
                    # Determine scale based on key
                    if key == "contrast":
                        scale = 10.0
                        s_val = int(val * scale)
                    else:
                        scale = 1.0
                        s_val = int(val)
                    
                    slider.blockSignals(True)
                    spinbox.blockSignals(True)
                    slider.setValue(s_val)
                    spinbox.setValue(val)
                    slider.blockSignals(False)
                    spinbox.blockSignals(False)
        
        # Update invert combo
        if "preprocessing.invert" in self._config_widgets:
            _, _, combo = self._config_widgets["preprocessing.invert"]
            if combo:
                invert_val = pp_config.get("invert", False)
                combo.blockSignals(True)
                combo.setCurrentIndex(1 if invert_val else 0)
                combo.blockSignals(False)
        
        # Update text_detection widgets
        td_config = self.config.get("text_detection", {})
        
        # Update GPU checkbox
        if "text_detection.use_gpu" in self._config_widgets:
            _, _, checkbox = self._config_widgets["text_detection.use_gpu"]
            if checkbox:
                gpu_val = td_config.get("use_gpu", False)
                checkbox.blockSignals(True)
                checkbox.setChecked(gpu_val)
                checkbox.blockSignals(False)
        
        for key in ["min_height_ratio", "median_height_fraction", "merge_vertical_ratio", 
                    "merge_horizontal_ratio", "merge_width_ratio_threshold"]:
            widget_key = f"text_detection.{key}"
            if widget_key in self._config_widgets:
                slider, spinbox, _ = self._config_widgets[widget_key]
                if slider and spinbox:
                    val = td_config.get(key, 0.0)
                    # Determine scale based on key
                    if key == "min_height_ratio":
                        scale = 1000  # per-mille
                        s_val = int(val * scale)
                    elif key in ["merge_vertical_ratio", "merge_horizontal_ratio", "merge_width_ratio_threshold"]:
                        scale = 100
                        s_val = int(val * scale)
                    elif key == "median_height_fraction":
                        scale = 100
                        s_val = int(val * scale)
                    else:
                        scale = 1
                        s_val = int(val)
                    
                    slider.blockSignals(True)
                    spinbox.blockSignals(True)
                    if 0 <= s_val <= slider.maximum():
                        slider.setValue(s_val)
                    spinbox.setValue(val)
                    slider.blockSignals(False)
                    spinbox.blockSignals(False)
        
        # Update text_sorting widgets
        sort_config = self.config.get("text_sorting", {})
        if "text_sorting.group_tolerance" in self._config_widgets:
            slider, spinbox, _ = self._config_widgets["text_sorting.group_tolerance"]
            if slider and spinbox:
                val = sort_config.get("group_tolerance", 0.5)
                slider.blockSignals(True)
                spinbox.blockSignals(True)
                slider.setValue(int(val * 10))
                spinbox.setValue(val)
                slider.blockSignals(False)
                spinbox.blockSignals(False)
        
        # Update hotkeys widgets
        hotkeys = self.config.get("hotkeys", {})
        for key in ["extract", "replay", "detect"]:
            widget_key = f"hotkeys.{key}"
            if widget_key in self._config_widgets:
                _, _, btn = self._config_widgets[widget_key]
                if btn:
                    val = hotkeys.get(key, "")
                    btn.current_hotkey = val
                    btn.setText(btn.format_hotkey(val))
        
        # Update TTS widgets
        tts_config = self.config.get("tts", {})
        if hasattr(self, 'voice_combo') and self.voice_combo:
            voice_code = tts_config.get("voice", "af_heart")
            index = self.voice_combo.findData(voice_code)
            if index >= 0:
                self.voice_combo.blockSignals(True)
                self.voice_combo.setCurrentIndex(index)
                self.voice_combo.blockSignals(False)
        
        if hasattr(self, 'sl_speed') and hasattr(self, 'sp_speed'):
            speed = tts_config.get("speed", 1.0)
            self.sl_speed.blockSignals(True)
            self.sp_speed.blockSignals(True)
            self.sl_speed.setValue(int(speed * 10))
            self.sp_speed.setValue(speed)
            self.sl_speed.blockSignals(False)
            self.sp_speed.blockSignals(False)
        
        if hasattr(self, 'sl_volume') and hasattr(self, 'sp_volume'):
            volume = tts_config.get("volume", 1.0) * 100  # Convert to percentage
            self.sl_volume.blockSignals(True)
            self.sp_volume.blockSignals(True)
            self.sl_volume.setValue(int(volume))
            self.sp_volume.setValue(int(volume))
            self.sl_volume.blockSignals(False)
            self.sp_volume.blockSignals(False)
        
        # Update text sorting combo
        if hasattr(self, 'order_combo') and self.order_combo:
            sort_dir = sort_config.get("direction") or "horizontal_ltr"
            idx = self.order_combo.findData(sort_dir)
            if idx >= 0:
                self.order_combo.blockSignals(True)
                self.order_combo.setCurrentIndex(idx)
                self.order_combo.blockSignals(False)
        
        # Update manual boxes
        if "manual_boxes" in self.config:
            try:
                loaded_boxes = [tuple(box) for box in self.config["manual_boxes"]]
                state.manual_boxes = loaded_boxes
                self.draw_manual_boxes()
            except Exception:
                print("Failed to load manual boxes from config")
        
        # Update visualizers
        if hasattr(self, 'settings_viz'):
            # Update detection viz
            min_h_ratio = td_config.get("min_height_ratio", 0.031)
            h = int(1080 * min_h_ratio)  # Estimate for visualization
            w = h
            self.settings_viz.update_detection(w, h)
            
            # Update merge viz
            v_ratio = td_config.get("merge_vertical_ratio", 0.07)
            h_ratio = td_config.get("merge_horizontal_ratio", 0.37)
            rat = td_config.get("merge_width_ratio_threshold", 0.75)
            vt = int(v_ratio * 20)
            ht = int(h_ratio * 20)
            self.settings_viz.update_merge(vt, ht, rat)

    def save_config(self):
        """Save current config to the active profile"""
        # Get current active profile name
        data = load_profiles()
        active_name = data.get("active", "Default")
        
        # Default profile is read-only - don't save changes to it
        if active_name == "Default":
            # Changes to Default are temporary until user duplicates it
            return
        
        # Save to user profile
        try:
            data["profiles"][active_name] = self.config.copy()
            save_profiles(data)
        except Exception as e:
            print(f"Error saving profile '{active_name}': {e}")

    def setup_hotkey_listener(self):
        """Setup global hotkey listener"""
        if not KEYBOARD_AVAILABLE:
            print("[Hotkey] Keyboard module not available. Hotkey disabled.")
            return

        # Unhook all existing hotkeys first to avoid duplicates/conflicts
        try:
            keyboard.unhook_all()
        except Exception:
            pass

        hotkeys = self.config.get("hotkeys", {
            "extract": "ctrl+shift+alt+z",
            "replay": "ctrl+shift+alt+x",
            "detect": "ctrl+shift+alt+m"
        })

        def register():
            try:
                # Extract
                hk_extract = hotkeys.get("extract", "ctrl+shift+alt+z")
                keyboard.add_hotkey(hk_extract, lambda: self.trigger_hotkey(mode="extract"))
                
                # Replay
                hk_replay = hotkeys.get("replay", "ctrl+shift+alt+x")
                keyboard.add_hotkey(hk_replay, lambda: self.trigger_hotkey(mode="replay"))
                
                # Detect
                hk_detect = hotkeys.get("detect", "ctrl+shift+alt+m")
                keyboard.add_hotkey(hk_detect, lambda: self.trigger_hotkey(mode="detect"))
                
                print(f"[Hotkey] Registered: Extract={hk_extract}, Replay={hk_replay}, Detect={hk_detect}")
            except Exception as e:
                print(f"[Hotkey] Error registering keys: {e}")

        # Register immediately (keyboard lib handles its own threading/hooks)
        register()

    def trigger_hotkey(self, mode="extract"):
        """Called when hotkey is pressed (runs in background thread)"""
        print(f"[Hotkey] Triggered: mode={mode}")
        # Play acknowledgment beep immediately
        try:
            self.play_beep("success")
        except Exception:
            pass
            
        # Use QTimer to safely call GUI method from background thread
        if mode == "extract":
            QTimer.singleShot(0, self.run_capture_and_extract)
        elif mode == "replay":
            QTimer.singleShot(0, self.replay_audio)
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
            self.play_beep("error")

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
            self.play_beep("error")

    def display_image(self, pil_image):
        """Display PIL image in graphics view"""
        self.scene.clear()
        self.raw_boxes = []
        self.filtered_boxes = []
        self.merged_boxes = []
        self.manual_box_items = []  # Clear references
        self.pixmap_item = None  # Reset reference
        self.selection_overlay_item = None

        # Convert PIL to QPixmap
        im_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QImage(im_data, pil_image.size[0], pil_image.size[1], QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)
        self.pixmap_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        # Draw initial selection overlay
        self.update_selection_overlay()
        
        # Draw existing manual boxes from state
        self.draw_manual_boxes()
        
        self.fit_image_to_view()

    def clear_all_boxes(self):
        """Remove all box items from scene (except pixmap)"""
        if not self.pixmap_item:
            return  # No image loaded yet
            
        items_to_remove = []
        items_list = self.scene.items()
        
        for item in items_list:
            # Keep pixmap, overlay, and manual boxes
            is_manual = isinstance(item, ManualBoxItem) or item in self.manual_box_items
            if item != self.pixmap_item and item != self.selection_overlay_item and not is_manual:
                # Also check if it's the current drawing rect
                if item != self.current_rect_item:
                    items_to_remove.append(item)
        
        for item in items_to_remove:
            self.scene.removeItem(item)

    def preview_live_filtering(self):
        """Show raw boxes: Green (Keep) vs Red (Discard) based on adaptive filtering"""
        if not self.raw_boxes or not self.pixmap_item:
            return

        self.clear_all_boxes()
        
        img_w = self.pixmap_item.pixmap().width()
        img_h = self.pixmap_item.pixmap().height()
        
        # Use adaptive parameters only
        td_config = self.config.get("text_detection", {})
        min_height_ratio = td_config.get("min_height_ratio", 0.0)
        median_height_fraction = td_config.get("median_height_fraction", 1.0)
        min_width_ratio = td_config.get("min_width_ratio", 0.0)
        
        # Calculate median height for adaptive filtering
        heights = [y2 - y1 for x1, y1, x2, y2 in self.raw_boxes if x2 > x1 and y2 > y1]
        import numpy as np
        median_h = np.median(heights) if heights else 0
        
        # Prepare pens/brushes
        pen_keep = QPen(QColor(0, 255, 0), 2)  # Green solid
        brush_keep = QBrush(QColor(0, 255, 0, 50))  # Semi-transparent green
        
        pen_discard = QPen(QColor(255, 0, 0), 1)  # Red dashed
        pen_discard.setStyle(Qt.PenStyle.DashLine)
        brush_discard = QBrush(QColor(255, 0, 0, 30))  # Semi-transparent red
        
        # Filter logic using adaptive parameters only
        for (x1, y1, x2, y2) in self.raw_boxes:
            # Clamp
            x1 = max(0, min(x1, img_w)); x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h)); y2 = max(0, min(y2, img_h))
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0: continue
            
            # Apply adaptive filtering only
            keep = True
            if min_height_ratio > 0 and h < (img_h * min_height_ratio):
                keep = False
            if min_width_ratio > 0 and w < (img_w * min_width_ratio):
                keep = False
            if median_h > 0 and h < (median_h * median_height_fraction) and w < (median_h * 2):
                keep = False
            
            if keep:
                # Keep (Green)
                self.scene.addRect(float(x1), float(y1), float(w), float(h), pen_keep, brush_keep)
            else:
                # Discard (Red Ghost)
                self.scene.addRect(float(x1), float(y1), float(w), float(h), pen_discard, brush_discard)

    def preview_live_merging(self, show_tolerance_zones=True):
        """Calculate and show merges live using Python (fast enough for typical OCR)"""
        if not self.raw_boxes or not self.pixmap_item:
            return

        # 1. Local Filtering using adaptive parameters only
        td_config = self.config.get("text_detection", {})
        img_w = self.pixmap_item.pixmap().width() if self.pixmap_item else 2000
        img_h = self.pixmap_item.pixmap().height() if self.pixmap_item else 2000
        
        min_width_ratio = td_config.get("min_width_ratio", 0.0)
        min_height_ratio = td_config.get("min_height_ratio", 0.0)
        median_height_fraction = td_config.get("median_height_fraction", 1.0)
        
        filtered = filter_text_regions(self.raw_boxes, (img_h, img_w), 
                                      min_width_ratio=min_width_ratio,
                                      min_height_ratio=min_height_ratio,
                                      median_height_fraction=median_height_fraction)
        
        # 2. Local Sorting
        sort_config = self.config.get("text_sorting", {})
        direction = sort_config.get("direction") or "horizontal_ltr"
        group_tol = sort_config.get("group_tolerance", 0.5)
        
        sorted_regions = sort_text_regions_by_reading_order(
            filtered, direction=direction, group_tolerance=group_tol
        )
        
        # 3. Local Merging using adaptive ratios only
        merge_vertical_ratio = td_config.get("merge_vertical_ratio", 0.07)
        merge_horizontal_ratio = td_config.get("merge_horizontal_ratio", 0.37)
        merge_width_ratio_threshold = td_config.get("merge_width_ratio_threshold", 0.75)
        
        merged, is_merged_flags, original_groups = merge_close_text_boxes(
            sorted_regions,
            vertical_ratio=merge_vertical_ratio,
            horizontal_ratio=merge_horizontal_ratio,
            width_ratio_threshold=merge_width_ratio_threshold
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
        # Use adaptive ratios, estimate pixels for visualization (assume 20px text height)
        v_ratio = td_config.get("merge_vertical_ratio", 0.07)
        h_ratio = td_config.get("merge_horizontal_ratio", 0.37)
        # Estimate pixels for visualization (assume average text height of 20px)
        v_tol = int(v_ratio * 20)
        h_tol = int(h_ratio * 20)

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
        ratio = self.config["text_detection"].get("merge_width_ratio_threshold", 0.75)
        
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
        direction = sort_config.get("direction") or "horizontal_ltr"
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
            
        # Ensure manual boxes are always visible and on top
        self.draw_manual_boxes()

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
            # TTS was already streamed during extraction, no need to call it again
            self.status_label.setText(f"Done! Reading {len(text)} chars aloud...")
        else:
            self.status_label.setText("Detection complete (no text extracted)")
        
        # Update phoneme display from state
        if hasattr(state, 'last_phonemes') and state.last_phonemes:
            self.phoneme_output.setText(state.last_phonemes)
        else:
            self.phoneme_output.setText("")

    def on_worker_error(self, err: str):
        """Handle worker error"""
        self.status_label.setText(f"Error: {err}")
        self.btn_cancel.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_extract.setEnabled(True)
        self.progress_bar.setValue(0)
        # Play error beep
        self.play_beep("error")

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
        """Filter events to handle viewport resize and mouse tools"""
        if obj == self.view.viewport():
            if event.type() == QEvent.Type.Resize:
                QTimer.singleShot(0, self.fit_image_to_view)
                
            # MOUSE TOOLS LOGIC
            elif self.tool_mode != "none" and self.pixmap_item:
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    self.is_drawing = True
                    self.start_point = self.view.mapToScene(event.pos())
                    return True
                    
                elif event.type() == QEvent.Type.MouseMove and self.is_drawing:
                    current_point = self.view.mapToScene(event.pos())
                    rect = QRectF(self.start_point, current_point).normalized()
                    
                    if not self.current_rect_item:
                        if self.tool_mode == "add":
                            pen_color = QColor(0, 255, 0)
                        elif self.tool_mode == "sub":
                            pen_color = QColor(255, 0, 0)
                        else:  # manual
                            pen_color = QColor(0, 100, 255)
                        self.current_rect_item = self.scene.addRect(rect, QPen(pen_color, 2))
                        self.current_rect_item.setZValue(1000)
                    else:
                        self.current_rect_item.setRect(rect)
                    return True
                    
                elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
                    self.is_drawing = False
                    end_point = self.view.mapToScene(event.pos())
                    
                    # Calculate normalized rect
                    img_w = self.pixmap_item.pixmap().width()
                    img_h = self.pixmap_item.pixmap().height()
                    
                    rect_f = QRectF(self.start_point, end_point).normalized()
                    
                    # Convert to normalized 0-1
                    nx = max(0, rect_f.x() / img_w)
                    ny = max(0, rect_f.y() / img_h)
                    nw = min(1.0 - nx, rect_f.width() / img_w)
                    nh = min(1.0 - ny, rect_f.height() / img_h)
                    
                    if nw > 0.001 and nh > 0.001:  # Min size check
                        if self.tool_mode == "manual":
                            # Add manual box in NORMALIZED coordinates (x, y, w, h)
                            box_norm = (nx, ny, nw, nh)
                            
                            # Avoid duplicates
                            if box_norm not in state.manual_boxes:
                                state.manual_boxes.append(box_norm)
                                self.save_manual_boxes()
                                # Redraw manual boxes immediately
                                self.draw_manual_boxes()
                                self.run_detection_preview()
                        else:
                            # Add selection operation
                            op = {
                                "op": self.tool_mode,
                                "rect": (nx, ny, nw, nh)
                            }
                            state.selection_ops.append(op)
                            self.update_selection_overlay()
                            self.run_detection_preview()
                    
                    # Remove temp item
                    if self.current_rect_item:
                        self.scene.removeItem(self.current_rect_item)
                        self.current_rect_item = None
                    
                    return True

        return super().eventFilter(obj, event)

    def draw_manual_boxes(self):
        """Draw manual boxes from state"""
        # First remove existing manual box items to avoid duplicates
        for item in self.manual_box_items[:]:  # Copy list to avoid modification during iteration
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        self.manual_box_items.clear()
        
        if not self.pixmap_item:
            return
            
        img_w = self.pixmap_item.pixmap().width()
        img_h = self.pixmap_item.pixmap().height()

        for box_norm in state.manual_boxes:
            # Validate normalized coordinates (should be 0-1)
            nx, ny, nw, nh = box_norm
            # Clamp normalized values to valid range
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            nw = max(0.0, min(1.0 - nx, nw))  # Ensure width doesn't exceed bounds
            nh = max(0.0, min(1.0 - ny, nh))  # Ensure height doesn't exceed bounds
            
            # Convert normalized -> pixels for display
            # This ensures boxes scale correctly with different image sizes/aspect ratios
            x = nx * img_w
            y = ny * img_h
            w = nw * img_w
            h = nh * img_h
            
            # Only draw if box is valid
            if w > 0 and h > 0:
                item = ManualBoxItem(QRectF(float(x), float(y), float(w), float(h)), self.remove_manual_box)
                # Store normalized data on item for easier removal identification
                item.box_data = box_norm
                
                self.scene.addItem(item)
                self.manual_box_items.append(item)

    def remove_manual_box(self, item: ManualBoxItem):
        """Callback to remove a manual box"""
        # Retrieve normalized data attached to item
        if hasattr(item, 'box_data'):
            target = item.box_data
            if target in state.manual_boxes:
                state.manual_boxes.remove(target)
                self.save_manual_boxes()
        
        # Remove from scene
        if item in self.manual_box_items:
            self.manual_box_items.remove(item)
        self.scene.removeItem(item)
            
        # Re-run preview to update results
        if state.last_image:
            self.run_detection_preview()

    def apply_dark_theme(self):
        """Apply dark theme using built-in QSS (no external dependencies)"""
        self.setStyleSheet(DARK_THEME_STYLESHEET)

    def change_ui_language(self):
        """Handle UI language change and save globally"""
        self.ui_lang = self.lang_combo.currentData()
        save_app_settings({"ui_lang": self.ui_lang})
        self.retranslateUi()

    def retranslateUi(self):
        """Update all UI text based on selected language"""
        t = self.TRANSLATIONS[self.ui_lang]
        
        # Set Layout Direction
        if self.ui_lang == "ar":
            self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        else:
            self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        # Update Main Labels
        self.setWindowTitle(t["window_title"])
        
        # Update Model Info
        if hasattr(self, 'model_info'):
            self.model_info.setText(t["model_info"])
        
        # Update Groups
        if hasattr(self, 'profile_group'):
            self.profile_group.setTitle(t["profile_group"])
        if hasattr(self, 'selection_group'):
            self.selection_group.setTitle(t["selection_group"])
        if hasattr(self, 'hotkeys_group'):
            self.hotkeys_group.setTitle(t["hotkeys_group"])
        if hasattr(self, 'image_settings_group'):
            self.image_settings_group.setTitle(t["img_adj_group"])
        if hasattr(self, 'text_processing_group'):
            self.text_processing_group.setTitle(t["text_proc_group"])
        if hasattr(self, 'tts_group'):
            self.tts_group.setTitle(t["tts_group"])
        if hasattr(self, 'status_group'):
            self.status_group.setTitle(t["status_group"])
        if hasattr(self, 'output_group'):
            self.output_group.setTitle(t["output_group"])
        
        # Update Preview Label
        if hasattr(self, 'preview_label'):
            self.preview_label.setText(t["preview_label"])
        
        # Update Lock Notice
        if hasattr(self, 'lock_notice'):
            self.lock_notice.setText(t["lock_notice"])
        
        # Update Buttons
        if hasattr(self, 'btn_capture'):
            self.btn_capture.setText(t["btn_capture"])
        if hasattr(self, 'btn_extract'):
            self.btn_extract.setText(t["btn_extract"])
        if hasattr(self, 'btn_cancel'):
            self.btn_cancel.setText(t["btn_cancel"])
        if hasattr(self, 'status_label'):
            self.status_label.setText(t["status_ready"])
        
        # Update Selection Tools
        if hasattr(self, 'chk_rapid'):
            self.chk_rapid.setText(t["chk_rapid"])
            self.chk_rapid.setToolTip(t["chk_rapid_tooltip"])
        if hasattr(self, 'chk_gpu'):
            self.chk_gpu.setText(t["chk_gpu"])
            self.chk_gpu.setToolTip(t["chk_gpu_tooltip"])
        if hasattr(self, 'btn_tool_none'):
            self.btn_tool_none.setText(t["btn_tool_none"])
        if hasattr(self, 'btn_tool_add'):
            self.btn_tool_add.setText(t["btn_tool_add"])
        if hasattr(self, 'btn_tool_sub'):
            self.btn_tool_sub.setText(t["btn_tool_sub"])
        if hasattr(self, 'btn_tool_manual'):
            self.btn_tool_manual.setText(t["btn_tool_manual"])
        if hasattr(self, 'btn_sel_all'):
            self.btn_sel_all.setText(t["btn_sel_all"])
        if hasattr(self, 'btn_desel_all'):
            self.btn_desel_all.setText(t["btn_desel_all"])
        if hasattr(self, 'btn_clear_manual'):
            self.btn_clear_manual.setText(t["btn_clear_manual"])
        
        # Update Profile Button Tooltips
        if hasattr(self, 'btn_new_profile'):
            self.btn_new_profile.setToolTip(t["btn_new_profile_tooltip"])
        if hasattr(self, 'btn_rename_profile'):
            self.btn_rename_profile.setToolTip(t["btn_rename_profile_tooltip"])
        if hasattr(self, 'btn_delete_profile'):
            self.btn_delete_profile.setToolTip(t["btn_delete_profile_tooltip"])
        
        # Update Language Label and Globe Button
        if hasattr(self, 'lang_label'):
            self.lang_label.setText(t["ui_lang_label"])
        if hasattr(self, 'lang_globe_btn'):
            self.lang_globe_btn.setToolTip(t["ui_lang_label"])
        
        # Update Image Settings Labels
        if hasattr(self, 'label_max_dimension'):
            self.label_max_dimension.setText(t["label_max_dimension"])
        if hasattr(self, 'label_colors'):
            self.label_colors.setText(t["label_colors"])
        
        # Update preprocessing labels
        if hasattr(self, '_preprocessing_labels'):
            if "label_binary_threshold" in self._preprocessing_labels:
                self._preprocessing_labels["label_binary_threshold"].setText(t["label_binary_threshold"])
            if "label_contrast" in self._preprocessing_labels:
                self._preprocessing_labels["label_contrast"].setText(t["label_contrast"])
            if "label_brightness" in self._preprocessing_labels:
                self._preprocessing_labels["label_brightness"].setText(t["label_brightness"])
            if "label_dilation" in self._preprocessing_labels:
                self._preprocessing_labels["label_dilation"].setText(t["label_dilation"])
        
        if hasattr(self, 'chk_invert'):
            # Update combo box items
            current_index = self.chk_invert.currentIndex()
            self.chk_invert.clear()
            self.chk_invert.addItem(t["colors_normal"], False)
            self.chk_invert.addItem(t["colors_invert"], True)
            self.chk_invert.setCurrentIndex(current_index)
        
        # Update Text Processing Labels
        if hasattr(self, 'legend'):
            self.legend.setText(t["legend_text"])
        if hasattr(self, 'detection_header_label'):
            self.detection_header_label.setText(t["detection_header"])
        if hasattr(self, 'detection_info_label'):
            self.detection_info_label.setText(t["detection_info"])
        if hasattr(self, 'label_min_height_ratio'):
            self.label_min_height_ratio.setText(t["label_min_height_ratio"])
        if hasattr(self, 'label_noise_filter'):
            self.label_noise_filter.setText(t["label_noise_filter"])
        if hasattr(self, 'merging_header_label'):
            self.merging_header_label.setText(t["merging_header"])
        if hasattr(self, 'label_order'):
            self.label_order.setText(t["label_order"])
        if hasattr(self, 'order_combo'):
            # Update order combo items
            current_data = self.order_combo.currentData()
            self.order_combo.clear()
            self.order_combo.addItem(t["order_ltr"], "horizontal_ltr")
            self.order_combo.addItem(t["order_rtl"], "horizontal_rtl")
            self.order_combo.addItem(t["order_vertical_rtl"], "vertical_rtl")
            self.order_combo.addItem(t["order_vertical_ltr"], "vertical_ltr")
            idx = self.order_combo.findData(current_data)
            if idx >= 0:
                self.order_combo.setCurrentIndex(idx)
        # Update merge ratio labels
        if hasattr(self, 'label_v_ratio'):
            self.label_v_ratio.setText(t["label_v_ratio"])
            self.label_v_ratio.setToolTip(t["tooltip_v_ratio"])
        if hasattr(self, 'label_h_ratio'):
            self.label_h_ratio.setText(t["label_h_ratio"])
            self.label_h_ratio.setToolTip(t["tooltip_h_ratio"])
        if hasattr(self, 'label_width_ratio'):
            self.label_width_ratio.setText(t["label_width_ratio"])
            self.label_width_ratio.setToolTip(t["tooltip_width_ratio"])
        if hasattr(self, 'label_line_grouping'):
            self.label_line_grouping.setText(t["label_line_grouping"])
        
        # Update TTS Labels
        if hasattr(self, 'tts_info_label'):
            self.tts_info_label.setText(t["tts_info"])
        if hasattr(self, 'label_language_filter'):
            self.label_language_filter.setText(t["label_language"])
        if hasattr(self, 'label_gender_filter'):
            self.label_gender_filter.setText(t["label_gender"])
        if hasattr(self, 'label_voice'):
            self.label_voice.setText(t["label_voice"])
        if hasattr(self, 'label_speed'):
            self.label_speed.setText(t["label_speed"])
        if hasattr(self, 'label_volume'):
            self.label_volume.setText(t["label_volume"])
        if hasattr(self, 'btn_play_new'):
            self.btn_play_new.setText(t["btn_play"])
            self.btn_play_new.setToolTip(t["btn_play_tooltip"])
        if hasattr(self, 'btn_replay'):
            self.btn_replay.setText(t["btn_replay"])
            self.btn_replay.setToolTip(t["btn_replay_tooltip"])
        if hasattr(self, 'btn_stop_audio'):
            self.btn_stop_audio.setText(t["btn_stop"])
            self.btn_stop_audio.setToolTip(t["btn_stop_tooltip"])
        if hasattr(self, 'label_phonemes'):
            self.label_phonemes.setText(t["label_phonemes"])
        if hasattr(self, 'phoneme_output'):
            self.phoneme_output.setPlaceholderText(t["phonemes_placeholder"])
        
        # Update all reset button tooltips (buttons with 🔄 emoji)
        for widget in self.findChildren(QPushButton):
            if widget.text() == "🔄":
                widget.setToolTip(t["reset_tooltip"])

