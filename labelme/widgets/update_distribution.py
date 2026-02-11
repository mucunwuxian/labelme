from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class UpdateDistributionWidget(QtWidgets.QWidget):
    """Widget showing update distribution based on shape modification times."""

    # Signal emitted when user clicks to move viewport
    viewportChangeRequested = QtCore.pyqtSignal(float, float)  # center_x_ratio, center_y_ratio

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QtGui.QPixmap | None = None
        self._scaled_pixmap: QtGui.QPixmap | None = None
        self._viewport_rect: QtCore.QRectF | None = None  # In ratio (0-1)
        self._shapes: list = []  # List of shapes to draw
        self._img_size: tuple[int, int] = (0, 0)  # Original image size
        self._img_offset: tuple[int, int] = (0, 0)  # Offset for centered image
        self.setMinimumSize(100, 50)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

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
        """Set the shapes to draw on the distribution map."""
        self._shapes = shapes
        self.update()

    def setViewportRect(self, x_ratio: float, y_ratio: float, w_ratio: float, h_ratio: float):
        """Set the viewport rectangle in ratio coordinates (0-1)."""
        self._viewport_rect = QtCore.QRectF(x_ratio, y_ratio, w_ratio, h_ratio)
        self.update()

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self._updateScaledPixmap()

    def _updateScaledPixmap(self):
        """Update the scaled pixmap to fit widget size with centered padding."""
        if self._pixmap is None or self._pixmap.isNull():
            self._scaled_pixmap = None
            self._img_offset = (0, 0)
            return

        # Get available space
        available_w = self.width()
        available_h = self.height()

        # Scale to fit within available space while maintaining aspect ratio
        self._scaled_pixmap = self._pixmap.scaled(
            available_w,
            available_h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        # Calculate offset for centering
        offset_x = (available_w - self._scaled_pixmap.width()) // 2
        offset_y = (available_h - self._scaled_pixmap.height()) // 2
        self._img_offset = (offset_x, offset_y)

    def _get_shape_alpha(self, shape) -> int:
        """Calculate alpha value for a shape based on whether modified_at is set.

        - modified_at is set = alpha 0 (transparent, no overlay)
        - modified_at is not set = alpha 128 (semi-transparent red)
        """
        modified_at_str = getattr(shape, 'modified_at', None)
        if modified_at_str:
            return 0  # Has modification date = transparent
        else:
            return 128  # No modification date = show red overlay

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Fill background with gray
        painter.fillRect(self.rect(), QtGui.QColor(180, 180, 180))

        if self._scaled_pixmap is None or self._scaled_pixmap.isNull():
            # Draw placeholder text
            painter.setPen(QtGui.QColor(128, 128, 128))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Image")
            return

        offset_x, offset_y = self._img_offset
        w = self._scaled_pixmap.width()
        h = self._scaled_pixmap.height()

        # Draw scaled image at centered position (like navigator)
        painter.drawPixmap(offset_x, offset_y, self._scaled_pixmap)

        # Draw shapes with red overlay based on age
        # Newer shapes = less overlay (bright), older shapes = more overlay (red)
        # Small shapes (both dimensions <= 50) and points are drawn as 50x50 circles
        # All shapes are expanded by 20 pixels in all directions
        if self._shapes and self._img_size[0] > 0 and self._img_size[1] > 0:
            scale_x = w / self._img_size[0]
            scale_y = h / self._img_size[1]
            min_radius_img = 25  # Minimum radius in image coordinates (50x50 circle)
            expand_img = 30  # Expansion in image coordinates

            for shape in self._shapes:
                if not shape.points:
                    continue

                alpha = self._get_shape_alpha(shape)
                # Use shape's assigned color for overlay
                shape_color = shape.line_color if shape.line_color else QtGui.QColor(255, 0, 0)
                fill_color = QtGui.QColor(shape_color.red(), shape_color.green(), shape_color.blue(), alpha)
                painter.setBrush(fill_color)
                # Border with shape color and alpha 176
                border_color = QtGui.QColor(shape_color.red(), shape_color.green(), shape_color.blue(), 176)
                border_pen = QtGui.QPen(border_color)
                border_pen.setWidthF(0.5)
                painter.setPen(border_pen)

                # Calculate bounding box in image coordinates
                xs = [p.x() for p in shape.points]
                ys = [p.y() for p in shape.points]

                if shape.shape_type == "circle" and len(shape.points) == 2:
                    # Circle: center + edge point
                    cx, cy = shape.points[0].x(), shape.points[0].y()
                    ex, ey = shape.points[1].x(), shape.points[1].y()
                    radius_img = ((ex - cx)**2 + (ey - cy)**2)**0.5
                    bbox_w = bbox_h = radius_img * 2
                elif shape.shape_type == "point":
                    cx, cy = shape.points[0].x(), shape.points[0].y()
                    bbox_w = bbox_h = 0  # Point has no size
                else:
                    bbox_w = max(xs) - min(xs)
                    bbox_h = max(ys) - min(ys)
                    cx = (max(xs) + min(xs)) / 2
                    cy = (max(ys) + min(ys)) / 2

                # If both dimensions <= 50 or is a point, draw as circle (radius 25 + 20 expansion)
                if (bbox_w <= 50 and bbox_h <= 50) or shape.shape_type == "point":
                    # Draw circle with minimum radius + expansion
                    center_x = cx * scale_x + offset_x
                    center_y = cy * scale_y + offset_y
                    radius = (min_radius_img + expand_img) * min(scale_x, scale_y)
                    painter.drawEllipse(QtCore.QPointF(center_x, center_y), radius, radius)
                else:
                    # Draw expanded shape
                    expand_x = expand_img * scale_x
                    expand_y = expand_img * scale_y

                    if shape.shape_type == "rectangle" and len(shape.points) == 2:
                        # Expand rectangle by 20 pixels on each side
                        x1 = min(xs) * scale_x + offset_x - expand_x
                        y1 = min(ys) * scale_y + offset_y - expand_y
                        x2 = max(xs) * scale_x + offset_x + expand_x
                        y2 = max(ys) * scale_y + offset_y + expand_y
                        painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
                    elif shape.shape_type == "circle" and len(shape.points) == 2:
                        # Expand circle radius by 20 pixels
                        center_x = cx * scale_x + offset_x
                        center_y = cy * scale_y + offset_y
                        radius = (radius_img + expand_img) * min(scale_x, scale_y)
                        painter.drawEllipse(QtCore.QPointF(center_x, center_y), radius, radius)
                    elif len(shape.points) >= 3:
                        # Expand polygon using QPainterPathStroker
                        points = [
                            QtCore.QPointF(p.x() * scale_x + offset_x, p.y() * scale_y + offset_y)
                            for p in shape.points
                        ]
                        path = QtGui.QPainterPath()
                        path.addPolygon(QtGui.QPolygonF(points))
                        path.closeSubpath()

                        # Create expanded path using stroker
                        stroker = QtGui.QPainterPathStroker()
                        stroker.setWidth(expand_img * 2 * min(scale_x, scale_y))
                        stroker.setJoinStyle(QtCore.Qt.RoundJoin)
                        expanded_path = stroker.createStroke(path).united(path)
                        painter.drawPath(expanded_path)

        # Draw viewport rectangle
        if self._viewport_rect is not None:
            rect = QtCore.QRectF(
                self._viewport_rect.x() * w + offset_x,
                self._viewport_rect.y() * h + offset_y,
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

        # Convert click position to ratio (accounting for offset)
        offset_x, offset_y = self._img_offset
        x_ratio = (pos.x() - offset_x) / self._scaled_pixmap.width()
        y_ratio = (pos.y() - offset_y) / self._scaled_pixmap.height()

        # Clamp to valid range
        x_ratio = max(0.0, min(1.0, x_ratio))
        y_ratio = max(0.0, min(1.0, y_ratio))

        self.viewportChangeRequested.emit(x_ratio, y_ratio)
