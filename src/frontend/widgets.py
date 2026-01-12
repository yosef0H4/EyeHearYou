"""Custom widgets for the OCR GUI"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont


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

