from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from labelme.widgets._shape_render_snapshot import shape_render_snapshot


class NavigatorWidget(QtWidgets.QWidget):
    """Minimap navigator showing the full image with current viewport indicator."""

    # Signal emitted when user clicks on navigator to move viewport
    viewportChangeRequested = QtCore.pyqtSignal(
        float, float
    )  # center_x_ratio, center_y_ratio

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QtGui.QPixmap | None = None
        self._scaled_pixmap: QtGui.QPixmap | None = None
        self._viewport_rect: QtCore.QRectF | None = None  # In ratio (0-1)
        self._shapes: list = []  # List of shapes to draw
        self._img_size: tuple[int, int] = (0, 0)  # Original image size
        self._img_offset: tuple[int, int] = (0, 0)  # Offset for centered image
        # Cached scene layer (gray background + scaled image + shapes);
        # rebuilt when the shape render snapshot / geometry / image content /
        # DPR changes — compared every paint, no external invalidation hooks.
        # The orange viewport rect is painted directly every time so viewport
        # moves never rebuild. The background/image are baked in (instead of
        # a transparent shapes-only layer) so the cached path is an opaque
        # copy and stays bit-exact with direct painting; a transparent
        # intermediate layer would add 8-bit premultiplied quantization
        # (measured ±2 LSB on antialiased edges).
        self._scene_layer: QtGui.QPixmap | None = None
        self._scene_layer_key: tuple | None = None
        self._scene_layer_builds: int = 0  # instrumentation (tests/bench)
        self._scene_layer_enabled: bool = True  # False: paint directly (tests)
        self.setMinimumSize(100, 50)  # Allow small size
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
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
        """Set the shapes to draw on the navigator."""
        self._shapes = shapes
        self.update()

    def setViewportRect(
        self, x_ratio: float, y_ratio: float, w_ratio: float, h_ratio: float
    ):
        """Set the viewport rectangle in ratio coordinates (0-1)."""
        rect = QtCore.QRectF(x_ratio, y_ratio, w_ratio, h_ratio)
        if self._viewport_rect is not None and rect == self._viewport_rect:
            return
        self._viewport_rect = rect
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
        dpr = self.devicePixelRatioF()
        use_cache = (
            self._scene_layer_enabled
            # Fractional DPR (e.g. 1.5): the layer's pixel grid cannot align
            # with the widget's physical pixels when width*dpr is not an
            # integer, so the blit is not bit-exact — fall back to direct
            # painting. Integer DPR (1 / 2, incl. macOS Retina) uses the cache.
            and dpr == int(dpr)
            and self._shapes
            and self._img_size[0] > 0
            and self._img_size[1] > 0
        )
        if use_cache:
            # Cached scene layer (background + image + shapes)
            key = (
                shape_render_snapshot(self._shapes),
                self.width(),
                self.height(),
                self._scaled_pixmap.width(),
                self._scaled_pixmap.height(),
                offset_x,
                offset_y,
                self._img_size,
                self.devicePixelRatioF(),
                self._scaled_pixmap.cacheKey(),  # detects image content change
            )
            if self._scene_layer is None or key != self._scene_layer_key:
                self._scene_layer = self._buildSceneLayer()
                self._scene_layer_key = key
                self._scene_layer_builds += 1
            painter.drawPixmap(0, 0, self._scene_layer)
        else:
            # Draw scaled image at centered position
            painter.drawPixmap(offset_x, offset_y, self._scaled_pixmap)
            # Draw shapes
            if self._shapes and self._img_size[0] > 0 and self._img_size[1] > 0:
                self._paintShapes(painter)

        # Draw viewport rectangle
        if self._viewport_rect is not None:
            w = self._scaled_pixmap.width()
            h = self._scaled_pixmap.height()
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

    def _buildSceneLayer(self) -> QtGui.QPixmap:
        """Render background + image + shapes into a widget-sized pixmap.

        Uses the exact same paint sequence as the direct path so the cached
        result is bit-identical (the layer is opaque, so compositing it is a
        plain copy). Physical resolution follows the device pixel ratio so
        the cache does not blur on Retina displays.
        """
        assert self._scaled_pixmap is not None  # guarded by paintEvent
        dpr = self.devicePixelRatioF()
        layer = QtGui.QPixmap(
            max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr))
        )
        layer.setDevicePixelRatio(dpr)
        p = QtGui.QPainter(layer)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(180, 180, 180))
        offset_x, offset_y = self._img_offset
        p.drawPixmap(offset_x, offset_y, self._scaled_pixmap)
        self._paintShapes(p)
        p.end()
        return layer

    def _paintShapes(self, painter: QtGui.QPainter) -> None:
        """Draw all shapes (identical commands to the pre-cache implementation)."""
        assert self._scaled_pixmap is not None  # guarded by paintEvent
        offset_x, offset_y = self._img_offset
        scale_x = self._scaled_pixmap.width() / self._img_size[0]
        scale_y = self._scaled_pixmap.height() / self._img_size[1]
        for shape in self._shapes:
            if not shape.points:
                continue
            color = (
                shape.line_color
                if hasattr(shape, "line_color")
                else QtGui.QColor(0, 255, 0)
            )
            # Make shapes more visible on navigator
            pen = QtGui.QPen(color)
            pen.setWidth(5)
            painter.setPen(pen)
            # Semi-transparent fill
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(60)
            painter.setBrush(fill_color)

            points = [
                QtCore.QPointF(p.x() * scale_x + offset_x, p.y() * scale_y + offset_y)
                for p in shape.points
            ]
            if shape.shape_type == "rectangle" and len(points) == 2:
                rect = QtCore.QRectF(points[0], points[1])
                painter.drawRect(rect)
            elif shape.shape_type == "circle" and len(points) == 2:
                center = points[0]
                radius = (
                    (points[1].x() - points[0].x()) ** 2
                    + (points[1].y() - points[0].y()) ** 2
                ) ** 0.5
                painter.drawEllipse(center, radius, radius)
            elif shape.shape_type == "point" and len(points) >= 1:
                painter.drawEllipse(points[0], 2, 2)
            elif len(points) >= 2:
                polygon = QtGui.QPolygonF(points)
                if shape.shape_type in ["polygon"] or (
                    hasattr(shape, "_closed") and shape._closed
                ):
                    polygon.append(points[0])  # Close polygon
                painter.drawPolyline(polygon)

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
