import copy
from datetime import datetime

import numpy as np
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

import labelme.utils

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


class Shape:
    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Render handles as capsule (for edge midpoints)
    P_CAPSULE = 2

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    # Edge midpoint indices for rectangle (used for edge dragging)
    EDGE_TOP = 0
    EDGE_BOTTOM = 1
    EDGE_LEFT = 2
    EDGE_RIGHT = 3

    PEN_WIDTH = 5

    # Minimum rectangle size (in pixels) to show edge midpoints
    MIN_RECT_SIZE_FOR_EDGE_HANDLES = 20

    # Flag to hide vertex outline during vertex dragging (set by canvas)
    hide_vertex_outline = False

    # The following class variables influence the drawing of all shape objects.
    line_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 102)  # 60% transparency
    fill_color: QtGui.QColor = QtGui.QColor(0, 0, 0, 51)  # 80% transparency
    vertex_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 255)
    select_line_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)
    select_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 64)
    hvertex_fill_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 128)

    point_type = P_ROUND
    point_size = 8
    # Size for point shape_type objects only (not polygon/rectangle vertices).
    # Highlighted/selected point objects use point_object_size * 1.5.
    point_object_size = 24
    scale = 1.0

    _current_vertex_fill_color: QtGui.QColor

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.point_labels = []
        self.shape_type = shape_type
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.fill = False
        self.selected = False
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.mask = mask
        self.modified_at = datetime.now().isoformat()
        self._mw_preview = False  # Magic wand preview: skip vertex drawing

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (2, self.P_ROUND),  # Reduced from 8 to 2
            self.MOVE_VERTEX: (3, self.P_SQUARE),
        }
        self._highlightEdgeMidpoint = None  # For rectangle edge midpoint highlighting
        self._hide_edge_midpoint = False  # Hide edge midpoints during edge dragging (instance-level)
        self._is_creating = False  # True when shape is being created (for start vertex highlight)
        self._is_line_preview = False  # True only for the preview line (self.line in canvas)

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
        ]:
            raise ValueError(f"Unexpected shape_type: {value}")
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            if self.point_labels:
                self.point_labels.pop()
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def canRemovePoint(self) -> bool:
        if not self.canAddPoint():
            return False

        if self.shape_type == "polygon" and len(self.points) <= 3:
            return False

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            return False

        return True

    def removePoint(self, i: int):
        if not self.canRemovePoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter):
        if self.mask is None and not self.points:
            return

        color = self.line_color  # Always use line_color, selection is shown by fill
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(self.PEN_WIDTH)
        painter.setPen(pen)

        if self.mask is not None:
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            fill_color = (
                self.select_fill_color.getRgb()
                if self.selected
                else self.fill_color.getRgb()
            )
            image_to_draw[self.mask] = fill_color
            qimage = QtGui.QImage.fromData(labelme.utils.img_arr_to_data(image_to_draw))
            qimage = qimage.scaled(
                qimage.size() * self.scale,
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

            painter.drawImage(self._scale_point(point=self.points[0]), qimage)

            line_path = QtGui.QPainterPath()
            contours = skimage.measure.find_contours(np.pad(self.mask, pad_width=1))
            for contour in contours:
                contour += [self.points[0].y(), self.points[0].x()]
                line_path.moveTo(
                    self._scale_point(QtCore.QPointF(contour[0, 1], contour[0, 0]))
                )
                for point in contour[1:]:
                    line_path.lineTo(
                        self._scale_point(QtCore.QPointF(point[1], point[0]))
                    )
            painter.drawPath(line_path)

        if self.points:
            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()
            negative_vrtx_path = QtGui.QPainterPath()
            start_vrtx_path = QtGui.QPainterPath()

            if self.shape_type in ["rectangle", "mask"]:
                if len(self.points) == 2:
                    rectangle = QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                    line_path.addRect(rectangle)
                if self.shape_type == "rectangle":
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
                    # Draw edge midpoint handles for rectangle
                    midpoints = self.getRectEdgeMidpoints()
                    if midpoints:
                        self.drawEdgeMidpoint(
                            painter, midpoints["top"], is_horizontal=True,
                            highlighted=(self._highlightEdgeMidpoint == self.EDGE_TOP)
                        )
                        self.drawEdgeMidpoint(
                            painter, midpoints["bottom"], is_horizontal=True,
                            highlighted=(self._highlightEdgeMidpoint == self.EDGE_BOTTOM)
                        )
                        self.drawEdgeMidpoint(
                            painter, midpoints["left"], is_horizontal=False,
                            highlighted=(self._highlightEdgeMidpoint == self.EDGE_LEFT)
                        )
                        self.drawEdgeMidpoint(
                            painter, midpoints["right"], is_horizontal=False,
                            highlighted=(self._highlightEdgeMidpoint == self.EDGE_RIGHT)
                        )
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    radius = labelme.utils.distance(
                        self._scale_point(self.points[0] - self.points[1])
                    )
                    line_path.addEllipse(
                        self._scale_point(self.points[0]), radius, radius
                    )
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self._scale_point(self.points[0]))
                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":
                assert len(self.points) == len(self.point_labels)
                for i, point_label in enumerate(self.point_labels):
                    if point_label == 1:
                        self.drawVertex(vrtx_path, i)
                    else:
                        self.drawVertex(negative_vrtx_path, i)
            else:
                line_path.moveTo(self._scale_point(self.points[0]))
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    # For preview line, skip drawing point 0 (it's already drawn in self.current)
                    if self._is_line_preview and i == 0:
                        continue
                    # Draw start vertex separately when creating (for white color)
                    if self._is_creating and i == 0:
                        self.drawVertex(start_vrtx_path, i)
                    else:
                        self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self._scale_point(self.points[0]))

            painter.drawPath(line_path)
            # Skip vertex drawing for magic wand preview
            if not self._mw_preview:
                # Draw start vertex in white when creating
                if self._is_creating and start_vrtx_path.length() > 0:
                    painter.fillPath(start_vrtx_path, QtGui.QColor(255, 255, 255))
                    painter.drawPath(start_vrtx_path)
                if vrtx_path.length() > 0:
                    # Hide vertex only for selected shape during vertex dragging
                    # Non-selected shapes always show their vertices
                    if not self._is_line_preview and (not self.selected or not Shape.hide_vertex_outline):
                        painter.fillPath(vrtx_path, self._current_vertex_fill_color)
                        painter.drawPath(vrtx_path)
            if self.fill and self.shape_type not in [
                "line",
                "linestrip",
                "points",
                "mask",
            ]:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))
            painter.drawPath(negative_vrtx_path)


    def drawVertex(self, path, i):
        d = self.point_size
        shape = self.point_type
        point = self._scale_point(self.points[i])
        # Point shapes use larger size for better visibility
        if self.shape_type == "point":
            if self.selected or i == self._highlightIndex:
                d = self.point_object_size * 1.5  # selected/highlighted
                shape = self.P_SQUARE
            else:
                d = self.point_object_size
        elif i == self._highlightIndex:
            # Apply highlight size multiplier only for non-point shapes
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size  # type: ignore[assignment]
        # For polygon creation preview line: show mouse position as square
        # Use same size as vertex editing (MOVE_VERTEX: size multiplier 3)
        # Only applies to self.line (preview line), not self.current (the polygon)
        if self._is_line_preview and i == len(self.points) - 1 and i > 0:
            size, shape = self._highlightSettings[self.MOVE_VERTEX]
            d *= size
        # For point shapes, use fill_color (with fill opacity) for inner color
        # Line opacity is applied via line_color (pen) for the outline
        if self.shape_type == "point":
            if self.selected:
                self._current_vertex_fill_color = self.select_fill_color
            else:
                self._current_vertex_fill_color = self.fill_color
        elif self._is_creating and i == 0:
            # Draw starting vertex in white during polygon creation
            self._current_vertex_fill_color = QtGui.QColor(255, 255, 255)
        elif self._is_line_preview and i == len(self.points) - 1 and i > 0:
            # Preview square: 70% transparent fill
            self._current_vertex_fill_color = QtGui.QColor(255, 255, 255, 100)
        elif self._highlightIndex is not None:
            self._current_vertex_fill_color = self.hvertex_fill_color
        else:
            self._current_vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def getRectEdgeMidpoints(self):
        """Get the midpoints of rectangle edges.
        Returns dict with keys: 'top', 'bottom', 'left', 'right'
        """
        if self.shape_type != "rectangle" or len(self.points) != 2:
            return None
        p0, p1 = self.points[0], self.points[1]
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())
        return {
            "top": QtCore.QPointF((left + right) / 2, top),
            "bottom": QtCore.QPointF((left + right) / 2, bottom),
            "left": QtCore.QPointF(left, (top + bottom) / 2),
            "right": QtCore.QPointF(right, (top + bottom) / 2),
        }

    def drawEdgeMidpoint(self, painter, point, is_horizontal, highlighted=False):
        """Draw a capsule-shaped handle at edge midpoint."""
        # Hide edge midpoint during dragging (only for this shape)
        if self._hide_edge_midpoint:
            return
        # Check if rectangle is large enough to show edge handles
        if len(self.points) != 2:
            return
        p0, p1 = self.points[0], self.points[1]
        width = abs(p1.x() - p0.x()) * self.scale
        height = abs(p1.y() - p0.y()) * self.scale
        if width < self.MIN_RECT_SIZE_FOR_EDGE_HANDLES or height < self.MIN_RECT_SIZE_FOR_EDGE_HANDLES:
            return

        # Save painter state to restore later
        painter.save()

        scaled_point = self._scale_point(point)
        # Capsule dimensions
        if is_horizontal:
            w, h = 16, 6  # horizontal capsule for top/bottom edges
        else:
            w, h = 6, 16  # vertical capsule for left/right edges

        if highlighted:
            fill_color = self.hvertex_fill_color
            # Enlarge when highlighted
            if is_horizontal:
                w, h = 36, 18
            else:
                w, h = 18, 36
        else:
            fill_color = self.vertex_fill_color
        border_color = self.line_color  # Same as shape's line color

        painter.setBrush(fill_color)
        painter.setPen(QtGui.QPen(border_color, self.PEN_WIDTH))  # Use same line width as shape

        rect = QtCore.QRectF(
            scaled_point.x() - w / 2,
            scaled_point.y() - h / 2,
            w, h
        )
        if highlighted:
            # Draw rectangle with square corners when highlighted
            painter.drawRect(rect)
        else:
            # Draw capsule (rounded rectangle) when not highlighted
            painter.drawRoundedRect(rect, h / 2, h / 2)

        # Restore painter state
        painter.restore()

    def nearestEdgeMidpoint(self, point, epsilon):
        """Find the nearest edge midpoint for rectangle shapes.
        Returns edge index (EDGE_TOP, EDGE_BOTTOM, EDGE_LEFT, EDGE_RIGHT) or None.

        Uses an elongated hit area matching the capsule shape:
        wider along the edge direction, narrower perpendicular to it.
        """
        if self.shape_type != "rectangle" or len(self.points) != 2:
            return None

        # Check if rectangle is large enough
        p0, p1 = self.points[0], self.points[1]
        width = abs(p1.x() - p0.x()) * self.scale
        height = abs(p1.y() - p0.y()) * self.scale
        if width < self.MIN_RECT_SIZE_FOR_EDGE_HANDLES or height < self.MIN_RECT_SIZE_FOR_EDGE_HANDLES:
            return None

        midpoints = self.getRectEdgeMidpoints()
        if not midpoints:
            return None

        point_scaled = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        # Hit area: along edge = half of edge length (capped), perpendicular = epsilon * 2
        hit_along = epsilon * 4
        hit_perp = epsilon * 2

        edge_map = {
            "top": self.EDGE_TOP,
            "bottom": self.EDGE_BOTTOM,
            "left": self.EDGE_LEFT,
            "right": self.EDGE_RIGHT,
        }
        horiz_edges = {"top", "bottom"}

        min_distance = float("inf")
        min_edge = None

        for edge_name, midpoint in midpoints.items():
            mp_scaled = QtCore.QPointF(midpoint.x() * self.scale, midpoint.y() * self.scale)
            dx = abs(point_scaled.x() - mp_scaled.x())
            dy = abs(point_scaled.y() - mp_scaled.y())
            if edge_name in horiz_edges:
                if dx <= hit_along and dy <= hit_perp:
                    dist = dy
                else:
                    continue
            else:
                if dy <= hit_along and dx <= hit_perp:
                    dist = dx
                else:
                    continue
            if dist < min_distance:
                min_distance = dist
                min_edge = edge_map[edge_name]

        return min_edge

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        # For point shapes, use visual size for hit detection
        if self.shape_type == "point":
            visual_size = (
                self.point_object_size * 1.5
                if self.selected
                else self.point_object_size
            )
            epsilon = max(epsilon, visual_size / 2)
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i, p in enumerate(self.points):
            p = QtCore.QPointF(p.x() * self.scale, p.y() * self.scale)
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i in range(len(self.points)):
            start = self.points[i - 1]
            end = self.points[i]
            start = QtCore.QPointF(start.x() * self.scale, start.y() * self.scale)
            end = QtCore.QPointF(end.x() * self.scale, end.y() * self.scale)
            line = [start, end]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point) -> bool:
        if self.shape_type in ["line", "linestrip", "points"]:
            return False
        if self.shape_type == "point":
            # For point shapes, check if the click is within the point's visual radius
            # Use actual display size: 36 when selected, 24 otherwise
            if self.points:
                dist = labelme.utils.distance(self.points[0] - point)
                visual_size = 36 if self.selected else 24
                return dist <= visual_size / 2 / self.scale
            return False
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def makePath(self):
        if self.shape_type in ["rectangle", "mask"]:
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                raidus = labelme.utils.distance(self.points[0] - self.points[1])
                path.addEllipse(self.points[0], raidus, raidus)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def touch(self):
        """Update the modification timestamp."""
        self.modified_at = datetime.now().isoformat()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def moveEdgeBy(self, edge_index, offset):
        """Move a rectangle edge by offset.

        Args:
            edge_index (int): The edge index (EDGE_TOP, EDGE_BOTTOM, EDGE_LEFT, EDGE_RIGHT)
            offset (QPointF): The offset to move
        """
        if self.shape_type != "rectangle" or len(self.points) != 2:
            return

        p0, p1 = self.points[0], self.points[1]

        if edge_index == self.EDGE_TOP:
            # Move top edge (adjust y of the point with smaller y)
            if p0.y() < p1.y():
                self.points[0] = QtCore.QPointF(p0.x(), p0.y() + offset.y())
            else:
                self.points[1] = QtCore.QPointF(p1.x(), p1.y() + offset.y())
        elif edge_index == self.EDGE_BOTTOM:
            # Move bottom edge (adjust y of the point with larger y)
            if p0.y() > p1.y():
                self.points[0] = QtCore.QPointF(p0.x(), p0.y() + offset.y())
            else:
                self.points[1] = QtCore.QPointF(p1.x(), p1.y() + offset.y())
        elif edge_index == self.EDGE_LEFT:
            # Move left edge (adjust x of the point with smaller x)
            if p0.x() < p1.x():
                self.points[0] = QtCore.QPointF(p0.x() + offset.x(), p0.y())
            else:
                self.points[1] = QtCore.QPointF(p1.x() + offset.x(), p1.y())
        elif edge_index == self.EDGE_RIGHT:
            # Move right edge (adjust x of the point with larger x)
            if p0.x() > p1.x():
                self.points[0] = QtCore.QPointF(p0.x() + offset.x(), p0.y())
            else:
                self.points[1] = QtCore.QPointF(p1.x() + offset.x(), p1.y())

    def moveEdgeTo(self, edge_index, pos):
        """Move a rectangle edge to an absolute position.

        Args:
            edge_index (int): The edge index (EDGE_TOP, EDGE_BOTTOM, EDGE_LEFT, EDGE_RIGHT)
            pos (QPointF): The position to move the edge to
        """
        if self.shape_type != "rectangle" or len(self.points) != 2:
            return

        p0, p1 = self.points[0], self.points[1]

        if edge_index == self.EDGE_TOP:
            # Move top edge to pos.y()
            if p0.y() < p1.y():
                self.points[0] = QtCore.QPointF(p0.x(), pos.y())
            else:
                self.points[1] = QtCore.QPointF(p1.x(), pos.y())
        elif edge_index == self.EDGE_BOTTOM:
            # Move bottom edge to pos.y()
            if p0.y() > p1.y():
                self.points[0] = QtCore.QPointF(p0.x(), pos.y())
            else:
                self.points[1] = QtCore.QPointF(p1.x(), pos.y())
        elif edge_index == self.EDGE_LEFT:
            # Move left edge to pos.x()
            if p0.x() < p1.x():
                self.points[0] = QtCore.QPointF(pos.x(), p0.y())
            else:
                self.points[1] = QtCore.QPointF(pos.x(), p1.y())
        elif edge_index == self.EDGE_RIGHT:
            # Move right edge to pos.x()
            if p0.x() > p1.x():
                self.points[0] = QtCore.QPointF(pos.x(), p0.y())
            else:
                self.points[1] = QtCore.QPointF(pos.x(), p1.y())

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action
        self._highlightEdgeMidpoint = None  # Clear edge midpoint highlight

    def highlightEdgeMidpoint(self, edge_index):
        """Highlight an edge midpoint for rectangle shapes.

        Args:
            edge_index (int): The edge index (EDGE_TOP, EDGE_BOTTOM, EDGE_LEFT, EDGE_RIGHT)
        """
        self._highlightEdgeMidpoint = edge_index
        self._highlightIndex = None  # Clear vertex highlight

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None
        self._highlightEdgeMidpoint = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
