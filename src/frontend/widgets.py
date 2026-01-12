"""Custom widgets for the OCR GUI"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPixmap


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
    """Visualizer widget for Merge settings showing tolerance zones and width ratio"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        self.setMaximumSize(120, 120)
        self.v_tol = 30
        self.h_tol = 50
        self.ratio = 0.3  # Default ratio
        
    def update_values(self, v_tol: int, h_tol: int, ratio: float):
        """Update visualization values"""
        self.v_tol = v_tol
        self.h_tol = h_tol
        self.ratio = ratio
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Base box size (Reference Box)
        base_w = 60
        base_h = 30
        
        # Center position
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # --- 1. Draw Tolerance Zone (Yellow Dashed) ---
        # Scale factor for tolerance visualization to fit in widget
        scale = 0.3
        
        zone_w = base_w + (self.h_tol * 2 * scale)
        zone_h = base_h + (self.v_tol * 2 * scale)
        zone_x = center_x - zone_w / 2
        zone_y = center_y - zone_h / 2
        
        pen = QPen(QColor(255, 255, 0), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        brush = QBrush(QColor(255, 255, 0, 25))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(int(zone_x), int(zone_y), int(zone_w), int(zone_h))
        
        # --- 2. Draw Reference Box (Blue Outline) ---
        box_x = center_x - base_w / 2
        box_y = center_y - base_h / 2
        
        pen = QPen(QColor(0, 150, 255), 2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        # No brush set - outline only
        painter.drawRect(int(box_x), int(box_y), base_w, base_h)

        # --- 3. Draw Width Ratio Threshold (Filled Inner Bar) ---
        # This represents the MINIMUM width another box must have to merge
        ratio_w = base_w * self.ratio
        ratio_x = center_x - ratio_w / 2
        
        # Draw the threshold bar (Cyan/White filled)
        no_pen = QPen()
        no_pen.setStyle(Qt.PenStyle.NoPen)
        painter.setPen(no_pen)
        painter.setBrush(QBrush(QColor(0, 200, 255, 100))) # Semi-transparent Cyan
        painter.drawRect(int(ratio_x), int(box_y) + 4, int(ratio_w), base_h - 8)
        
        # Draw "Min" label if space permits
        if ratio_w > 20:
            painter.setPen(QColor(255, 255, 255))
            font = QFont()
            font.setPixelSize(9)
            painter.setFont(font)
            painter.drawText(QRect(int(ratio_x), int(box_y), int(ratio_w), base_h), 
                           Qt.AlignmentFlag.AlignCenter, "MIN")


class ResizeVizWidget(QWidget):
    """Visualizer for Image Dimension settings showing pixelation effect"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 60)
        self.setMaximumSize(120, 60)
        self.max_dim = 1080
        
        # Create a source pixmap with text to demonstrate quality
        self.source_pix = QPixmap(240, 120)
        self.source_pix.fill(QColor(0, 0, 0))
        
        p = QPainter(self.source_pix)
        p.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI", 24)
        font.setBold(True)
        p.setFont(font)
        p.drawText(self.source_pix.rect(), Qt.AlignmentFlag.AlignCenter, "Quality")
        p.end()
        
    def update_value(self, max_dim: int):
        """Update visualization value"""
        self.max_dim = max_dim
        self.update()

    def paintEvent(self, event):
        """Draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # False to show pixels
        
        # Black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Simulate scaling
        # We assume a "standard" source height of 1080p. 
        # If max_dim is < 1080, we simulate the quality loss.
        base_h = 1080.0
        scale_factor = min(1.0, self.max_dim / base_h)
        
        # 1. Scale down to target resolution
        target_w = max(1, int(self.source_pix.width() * scale_factor))
        target_h = max(1, int(self.source_pix.height() * scale_factor))
        
        tiny_pix = self.source_pix.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation  # Pixelated scaling
        )
        
        # 2. Scale back up to widget size (nearest neighbor) to show the pixels
        final_pix = tiny_pix.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        
        # Center drawing
        x = (self.width() - final_pix.width()) // 2
        y = (self.height() - final_pix.height()) // 2
        painter.drawPixmap(x, y, final_pix)

