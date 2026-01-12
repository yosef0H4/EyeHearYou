"""PyQt6 GUI frontend for Visual Novel OCR"""
import sys
from PyQt6.QtWidgets import QApplication

from src.frontend.window import OCRWindow


def main():
    """Main entry point for the GUI application"""
    app = QApplication(sys.argv)
    window = OCRWindow()
    window.show()
    sys.exit(app.exec())


__all__ = ['main', 'OCRWindow']


