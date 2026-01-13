"""PyQt6 GUI frontend for EyeHearYou"""
import sys
import time
from PyQt6.QtWidgets import QApplication, QSplashScreen, QProgressBar, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QPixmap, QColor, QFont

from src.frontend.window import OCRWindow
from src.backend.core.model_loader import preload_model
from src.backend.core.tts import preload_tts
from src.backend.core.detection import preload_rapidocr


class ModelLoaderThread(QThread):
    progress_update = Signal(str, int)
    finished_loading = Signal()

    def run(self):
        # 1. RapidOCR
        self.progress_update.emit("Initializing RapidOCR...", 10)
        try:
            preload_rapidocr(test=True)
        except Exception as e:
            print(f"Error loading RapidOCR: {e}")
        
        # 2. H2OVL Model
        self.progress_update.emit("Loading AI Model (H2OVL)...", 30)
        try:
            preload_model(test=True)
        except Exception as e:
            print(f"Error loading Model: {e}")
        
        # 3. TTS Model
        self.progress_update.emit("Loading TTS Engine (Kokoro)...", 60)
        try:
            preload_tts(test=True)
        except Exception as e:
            print(f"Error loading TTS: {e}")
        
        self.progress_update.emit("Starting UI...", 100)
        time.sleep(0.5)  # Short pause to see completion
        self.finished_loading.emit()


class LoadingScreen(QWidget):
    """Loading screen widget with progress bar"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeHearYou - Loading")
        self.setFixedSize(450, 250)
        self.setStyleSheet("background-color: #1e1e1e;")
        # Center window on screen
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(30, 40, 30, 30)
        
        # Title
        self.title_label = QLabel("EyeHearYou")
        font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #cccccc; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        layout.addSpacing(10)
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
                color: transparent; 
                height: 6px;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
        self.progress.setFixedHeight(8)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        
        layout.addStretch()

    def update_progress(self, msg, value):
        self.status_label.setText(msg)
        self.progress.setValue(value)
        QApplication.processEvents()  # Process events to update UI


def main():
    """Main entry point for the GUI application"""
    app = QApplication(sys.argv)
    
    # Create and show loading screen
    splash = LoadingScreen()
    # Center on screen
    screen = app.primaryScreen().geometry()
    splash_geometry = splash.geometry()
    splash.move(
        (screen.width() - splash_geometry.width()) // 2,
        (screen.height() - splash_geometry.height()) // 2
    )
    splash.show()
    app.processEvents()  # Show loading screen immediately
    
    # Create loader thread
    loader = ModelLoaderThread()
    
    # Define callback for when loading finishes
    def on_finished():
        # Initialize main window
        window = OCRWindow()
        window.show()
        splash.close()  # Close loading screen
        # Keep window reference to prevent garbage collection
        app.active_window = window
        
    loader.progress_update.connect(splash.update_progress)
    loader.finished_loading.connect(on_finished)
    
    # Start loading
    loader.start()
    
    sys.exit(app.exec())


__all__ = ['main', 'OCRWindow']


