"""Theme styling for the GUI"""
DARK_THEME_STYLESHEET = """
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
"""



