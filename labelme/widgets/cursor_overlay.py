import math

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class CursorOverlayWidget(QtWidgets.QWidget):
    """Transparent overlay widget for drawing custom cursor (crosshair + point circle)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Make widget transparent and pass through mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self._cursor_pos = None  # Position in widget coordinates
        self._show_crosshair = False
        self._show_point_circle = False
        self._crosshair_color = QtGui.QColor(0, 255, 0, 128)
        self._point_circle_color = QtGui.QColor(0, 255, 0, 128)

        # Pre-render Gaussian gradient
        self._gradient_pixmap = self._create_gradient_pixmap()

    def _create_gradient_pixmap(self) -> QtGui.QPixmap:
        """Create pre-rendered Gaussian gradient for crosshair."""
        radius = 40
        sigma = 15
        max_alpha = 100
        size = radius * 2 + 1
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq <= radius * radius:
                    alpha = int(max_alpha * math.exp(-dist_sq / (2 * sigma * sigma)))
                    painter.setPen(QtGui.QColor(255, 255, 255, alpha))
                    painter.drawPoint(radius + dx, radius + dy)
        painter.end()
        return pixmap

    def setCursorPos(self, widget_pos):
        """Set cursor position in widget coordinates."""
        self._cursor_pos = widget_pos
        self.update()

    def setShowCrosshair(self, show: bool):
        """Enable/disable crosshair display."""
        self._show_crosshair = show
        self.update()

    def setShowPointCircle(self, show: bool):
        """Enable/disable point circle display."""
        self._show_point_circle = show
        self.update()

    def setCrosshairColor(self, color: QtGui.QColor):
        """Set crosshair line color."""
        self._crosshair_color = color
        self.update()

    def setPointCircleColor(self, color: QtGui.QColor):
        """Set point circle color."""
        self._point_circle_color = color
        self.update()

    def hideCursor(self):
        """Hide all cursor graphics."""
        self._show_crosshair = False
        self._show_point_circle = False
        self._cursor_pos = None
        self.update()

    def paintEvent(self, event):
        if self._cursor_pos is None:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Use widget coordinates directly
        cx = int(self._cursor_pos.x())
        cy = int(self._cursor_pos.y())

        if self._show_crosshair:
            # Draw Gaussian gradient
            radius = 40
            painter.drawPixmap(cx - radius, cy - radius, self._gradient_pixmap)

            # Draw crosshair lines
            line_color = QtGui.QColor(self._crosshair_color)
            line_color.setAlpha(128)
            pen = QtGui.QPen(line_color)
            pen.setWidth(1)
            painter.setPen(pen)
            line_len = 30
            painter.drawLine(cx - line_len, cy, cx + line_len, cy)
            painter.drawLine(cx, cy - line_len, cx, cy + line_len)

        if self._show_point_circle:
            # Draw point circle
            point_radius = 20
            pen = QtGui.QPen(self._point_circle_color)
            pen.setWidthF(5)  # Shape.PEN_WIDTH
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), point_radius, point_radius)
