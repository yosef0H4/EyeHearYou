"""PyQt6 GUI frontend for Visual Novel OCR"""
import sys
import threading
from PyQt6.QtWidgets import QApplication

from src.frontend.window import OCRWindow
from src.backend.core.model_loader import preload_model


def main():
    """Main entry point for the GUI application"""
    app = QApplication(sys.argv)
    
    # Preload model in background thread to avoid blocking UI
    def load_model_async():
        preload_model(test=True)
    
    model_thread = threading.Thread(target=load_model_async, daemon=True)
    model_thread.start()
    
    window = OCRWindow()
    window.show()
    sys.exit(app.exec())


__all__ = ['main', 'OCRWindow']


