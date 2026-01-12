"""PyQt6 GUI frontend for OCR Accessibility Tool"""
import sys
import threading
from PyQt6.QtWidgets import QApplication

from src.frontend.window import OCRWindow
from src.backend.core.model_loader import preload_model
from src.backend.core.tts import preload_tts
from src.backend.core.detection import preload_rapidocr


def main():
    """Main entry point for the GUI application"""
    app = QApplication(sys.argv)
    
    # Preload models in background thread to avoid blocking UI
    def load_models_async():
        preload_rapidocr(test=True)
        preload_model(test=True)
        preload_tts(test=True)
    
    model_thread = threading.Thread(target=load_models_async, daemon=True)
    model_thread.start()
    
    window = OCRWindow()
    window.show()
    sys.exit(app.exec())


__all__ = ['main', 'OCRWindow']


