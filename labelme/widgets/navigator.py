from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class NavigatorWidget(QtWidgets.QWidget):
    """Minimap navigator showing the full image with current viewport indicator."""

    # Signal emitted when user clicks on navigator to move viewport
    viewportChangeRequested = QtCore.pyqtSignal(float, float)  # center_x_ratio, center_y_ratio

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QtGui.QPixmap | None = None
        self._scaled_pixmap: QtGui.QPixmap | None = None
        self._viewport_rect: QtCore.QRectF | None = None  # In ratio (0-1)
        self._shapes: list = []  # List of shapes to draw
        self._img_size: tuple[int, int] = (0, 0)  # Original image size
        self._max_size = 300
        self.setMinimumSize(100, 100)
        self.setMaximumSize(self._max_size, self._max_size)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def setPixmap(self, pixmap: QtGui.QPixmap | None):
        """Set the source image pixmap."""
        self._pixmap = pixmap
        if pixmap and not pixmap.isNull():
            self._img_size = (pixmap.width(), pixmap.height())
        else:
            self._img_size = (0, 0)
        self._updateScaledPixmap()
        self.update()

    def setShapes(self, shapes: list):
        """Set the shapes to draw on the navigator."""
        self._shapes = shapes
        self.update()

    def setViewportRect(self, x_ratio: float, y_ratio: float, w_ratio: float, h_ratio: float):
        """Set the viewport rectangle in ratio coordinates (0-1)."""
        self._viewport_rect = QtCore.QRectF(x_ratio, y_ratio, w_ratio, h_ratio)
        self.update()

    def _updateScaledPixmap(self):
        """Update the scaled pixmap to fit widget size."""
        if self._pixmap is None or self._pixmap.isNull():
            self._scaled_pixmap = None
            return

        # Scale to fit within max size while maintaining aspect ratio
        self._scaled_pixmap = self._pixmap.scaled(
            self._max_size,
            self._max_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        # Resize widget to match scaled pixmap
        self.setFixedSize(self._scaled_pixmap.size())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QtGui.QColor(40, 40, 40))

        if self._scaled_pixmap is None or self._scaled_pixmap.isNull():
            # Draw placeholder text
            painter.setPen(QtGui.QColor(128, 128, 128))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Image")
            return

        # Draw scaled image
        painter.drawPixmap(0, 0, self._scaled_pixmap)

        # Draw shapes
        if self._shapes and self._img_size[0] > 0 and self._img_size[1] > 0:
            scale_x = self._scaled_pixmap.width() / self._img_size[0]
            scale_y = self._scaled_pixmap.height() / self._img_size[1]
            for shape in self._shapes:
                if not shape.points:
                    continue
                color = shape.line_color if hasattr(shape, 'line_color') else QtGui.QColor(0, 255, 0)
                # Make shapes more visible on navigator
                pen = QtGui.QPen(color)
                pen.setWidth(5)
                painter.setPen(pen)
                # Semi-transparent fill
                fill_color = QtGui.QColor(color)
                fill_color.setAlpha(60)
                painter.setBrush(fill_color)

                points = [QtCore.QPointF(p.x() * scale_x, p.y() * scale_y) for p in shape.points]
                if shape.shape_type == "rectangle" and len(points) == 2:
                    rect = QtCore.QRectF(points[0], points[1])
                    painter.drawRect(rect)
                elif shape.shape_type == "circle" and len(points) == 2:
                    center = points[0]
                    radius = ((points[1].x() - points[0].x())**2 + (points[1].y() - points[0].y())**2)**0.5
                    painter.drawEllipse(center, radius, radius)
                elif shape.shape_type == "point" and len(points) >= 1:
                    painter.drawEllipse(points[0], 2, 2)
                elif len(points) >= 2:
                    polygon = QtGui.QPolygonF(points)
                    if shape.shape_type in ["polygon"] or (hasattr(shape, '_closed') and shape._closed):
                        polygon.append(points[0])  # Close polygon
                    painter.drawPolyline(polygon)

        # Draw viewport rectangle
        if self._viewport_rect is not None:
            w = self._scaled_pixmap.width()
            h = self._scaled_pixmap.height()
            rect = QtCore.QRectF(
                self._viewport_rect.x() * w,
                self._viewport_rect.y() * h,
                self._viewport_rect.width() * w,
                self._viewport_rect.height() * h,
            )

            # Orange semi-transparent fill
            painter.setBrush(QtGui.QColor(255, 165, 0, 80))
            # Orange border
            pen = QtGui.QPen(QtGui.QColor(255, 165, 0, 200))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._emitViewportChange(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self._emitViewportChange(event.pos())

    def _emitViewportChange(self, pos):
        """Emit signal to move viewport center to clicked position."""
        if self._scaled_pixmap is None or self._scaled_pixmap.isNull():
            return

        # Convert click position to ratio
        x_ratio = pos.x() / self._scaled_pixmap.width()
        y_ratio = pos.y() / self._scaled_pixmap.height()

        # Clamp to valid range
        x_ratio = max(0.0, min(1.0, x_ratio))
        y_ratio = max(0.0, min(1.0, y_ratio))

        self.viewportChangeRequested.emit(x_ratio, y_ratio)
