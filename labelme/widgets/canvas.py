from __future__ import annotations

import enum
import os
import ctypes
import ctypes.util
from typing import Literal

import imgviz
import numpy as np
import osam
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import Qt

import labelme.utils
from labelme._automation import OsamSession
from labelme._automation import polygon_from_mask
from labelme.shape import Shape

from .cursor_overlay import CursorOverlayWidget
from .download import download_ai_model

try:  # macOS cursor hide/unhide (PyObjC)
    from AppKit import NSCursor  # type: ignore

    _NSCURSOR_AVAILABLE = True
except Exception:
    _NSCURSOR_AVAILABLE = False

try:  # macOS CoreGraphics cursor hide/unhide (more forceful)
    import Quartz  # type: ignore

    _QUARTZ_AVAILABLE = True
except Exception:
    _QUARTZ_AVAILABLE = False

# Fallback for macOS CoreGraphics via ctypes (no Quartz module needed)
_CG_AVAILABLE = False
_CG = None
try:
    _cg_path = ctypes.util.find_library("ApplicationServices")
    if _cg_path:
        _CG = ctypes.cdll.LoadLibrary(_cg_path)
        _CG_AVAILABLE = True
except Exception:
    _CG_AVAILABLE = False

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

MOVE_SPEED = 5.0


class CanvasMode(enum.Enum):
    CREATE = enum.auto()
    EDIT = enum.auto()


class Canvas(QtWidgets.QWidget):
    pixmap: QtGui.QPixmap
    _pixmap_hash: int | None
    _cursor: QtCore.Qt.CursorShape
    shapes: list[Shape]
    shapesBackups: list[list[Shape]]
    movingShape: bool
    selectedShapes: list[Shape]
    selectedShapesCopy: list[Shape]
    current: Shape | None
    hShape: Shape | None
    prevhShape: Shape | None
    hVertex: int | None
    prevhVertex: int | None
    hEdge: int | None
    prevhEdge: int | None

    zoomRequest = QtCore.pyqtSignal(int, QPointF)
    pinchZoomRequest = QtCore.pyqtSignal(float, QPointF)
    scrollRequest = QtCore.pyqtSignal(int, int)
    newShape = QtCore.pyqtSignal()
    selectionChanged = QtCore.pyqtSignal(list)
    shapeMoved = QtCore.pyqtSignal()
    drawingPolygon = QtCore.pyqtSignal(bool)
    vertexSelected = QtCore.pyqtSignal(bool)
    mouseMoved = QtCore.pyqtSignal(QPointF)
    statusUpdated = QtCore.pyqtSignal(str)
    editModeChanged = QtCore.pyqtSignal(bool)  # True = edit mode, False = create mode

    mode: CanvasMode = CanvasMode.EDIT

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False
    _near_start_point = False  # True when creating polygon and near starting point

    prevPoint: QPointF
    prevMovePoint: QPointF
    offsets: tuple[QPointF, QPointF]

    _dragging_start_pos: QPointF
    _is_dragging: bool
    _is_dragging_enabled: bool

    _osam_session_model_name: str = "sam2:latest"
    _osam_session: OsamSession | None

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                f"Unexpected value for double_click event: {self.double_click}"
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
            },
        )
        super().__init__(*args, **kwargs)

        # State flags used by cursor handling during init/reset
        self._vertex_dragging = False  # True when dragging a vertex
        self._edge_midpoint_dragging = False  # True when dragging an edge midpoint
        self._edge_midpoint_drag_shape = None  # Shape whose edge midpoint is being dragged
        self._dragging_edge_index = None  # Which edge is being dragged (EDGE_TOP, etc.)
        self._snap_active = False  # True when parallel line snap is active
        self._snap_line_pos = None  # Position of detected parallel line (image coords)
        self._cursor_debug = os.environ.get("LABELME_CURSOR_DEBUG") == "1"
        self._custom_cursor_enabled = False
        self._right_click_edit_enabled = False
        self._parallel_line_dist_enabled = False
        self._text_bounding_enabled = False
        self._line_fit_enabled = False
        self._lf_snap_cache: tuple | None = None
        self._lf_snap_entered = False
        self._dark_pixel_magnet_enabled = False
        self._dpm_ghost_pos: QPointF | None = None  # real mouse pos during DPM snap
        self._text_bounding_snap_dots: list[tuple[float, float]] | None = None
        self._tb_boundary_cache: tuple | None = None  # cached boundary during drag
        self._tb_snap_entered = False  # True on first frame of snap (triggers cursor warp)
        self._pl_snap_cache: tuple | None = None  # cached parallel line snap during drag
        self._pl_snap_entered = False  # True on first frame of snap (triggers cursor warp)
        self._parallel_line_magnet_config: list[dict] = []
        self._text_bounding_magnet_config: list[dict] = []
        self._line_fit_magnet_config: list[dict] = []
        self._reference_medians: dict[str, float | None] = {}
        self._dark_pixel_magnet_config: list[dict] | None = None
        self._pending_draw_label: str | None = None
        self._dpm_reachable_cache: dict[int, np.ndarray] = {}
        self._dpm_reachable_hash: int | None = None
        self._ns_cursor_hidden = False
        self._os_cursor_hidden = False

        self.resetState()

        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.line._is_creating = True  # Mark line as creating preview
        self.line._is_line_preview = True  # Mark as the preview line (not the polygon itself)
        self.prevPoint = QPointF()
        self.prevMovePoint = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self._osam_session = None
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._dragging_start_pos = QPointF()
        self._is_dragging = False
        self._is_dragging_enabled = False
        self._context_menu_active = False
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

        # Enable pinch-to-zoom gesture (Mac trackpad support)
        self.grabGesture(Qt.PinchGesture)
        self.setAttribute(Qt.WA_AcceptTouchEvents)

        # Hover label delay
        self._hover_label_shape = None
        self._hover_label_ready = False
        self._hover_label_timer = QtCore.QTimer(self)
        self._hover_label_timer.setSingleShot(True)
        self._hover_label_timer.setInterval(500)  # 0.5 second delay
        self._hover_label_timer.timeout.connect(self._onHoverLabelTimeout)
        self._hover_label_last_pos = None  # Position when timer started
        self._mouse_pressed = False

        # Blank cursor for hiding during vertex/polygon operations
        blank_pixmap = QtGui.QPixmap(32, 32)
        blank_pixmap.fill(Qt.transparent)
        blank_pixmap.setDevicePixelRatio(self.devicePixelRatioF())
        self._blank_cursor = QtGui.QCursor(blank_pixmap, 0, 0)

        # Cursor overlay widget (separate layer for performance)
        self._cursor_overlay = CursorOverlayWidget(self)
        self._cursor_overlay.setGeometry(self.rect())
        self._cursor_overlay.show()
        self._cursor_overlay.raise_()

        # Scroll debounce
        self._last_scroll_time = 0.0

    def _updateCursorOverlay(self):
        """Update the cursor overlay widget based on current state."""
        if not self._custom_cursor_enabled:
            self._cursor_overlay.hideCursor()
            return

        if self.prevMovePoint is None or self.outOfPixmap(self.prevMovePoint):
            self._cursor_overlay.hideCursor()
            # Show arrow cursor in margin area (outside image but inside canvas)
            self.overrideCursor(CURSOR_DEFAULT)
            return

        # Determine if crosshair should show
        show_crosshair = (
            (self._vertex_dragging and self.prevMovePoint is not None)
            or (self.drawing() and self.current and self.prevMovePoint is not None)
            or (self.createMode in ["point", "polygon", "rectangle"] and self.drawing() and self.prevMovePoint is not None)
        ) and not self._near_start_point

        # Determine crosshair color
        if self.hShape is not None:
            crosshair_color = QtGui.QColor(self.hShape.line_color)
        elif self.current is not None:
            crosshair_color = QtGui.QColor(self.current.line_color)
        elif self.createMode in ["point", "polygon", "rectangle"]:
            crosshair_color = QtGui.QColor(Shape.line_color)
        else:
            crosshair_color = QtGui.QColor(128, 128, 128)

        # Determine if point circle should show
        show_point_circle = False
        point_circle_color = None
        point_circle_pos = self.prevMovePoint

        if self._vertex_dragging and self.hShape is not None and self.hShape.shape_type == "point":
            show_point_circle = True
            point_circle_color = QtGui.QColor(self.hShape.line_color)
            point_circle_pos = self.hShape.points[0]
        elif self.createMode == "point" and self.drawing():
            show_point_circle = True
            point_circle_color = QtGui.QColor(Shape.line_color)

        # Convert image coordinates to widget coordinates for overlay
        image_pos = point_circle_pos if show_point_circle else self.prevMovePoint
        offset = self.offsetToCenter() if self.pixmap else QPointF(0, 0)
        widget_pos = QPointF(
            (image_pos.x() + offset.x()) * self.scale,
            (image_pos.y() + offset.y()) * self.scale
        )
        self._cursor_overlay.setCursorPos(widget_pos)
        self._cursor_overlay.setShowCrosshair(show_crosshair)
        self._cursor_overlay.setShowPointCircle(show_point_circle)
        self._cursor_overlay.setCrosshairColor(crosshair_color)
        if point_circle_color:
            self._cursor_overlay.setPointCircleColor(point_circle_color)

    def setCustomCursorEnabled(self, enabled: bool):
        self._custom_cursor_enabled = enabled
        self._updateCursorOverlay()

    def setRightClickEditEnabled(self, enabled: bool):
        self._right_click_edit_enabled = enabled

    def setParallelLineDistEnabled(self, enabled: bool):
        self._parallel_line_dist_enabled = enabled

    def setTextBoundingEnabled(self, enabled: bool):
        self._text_bounding_enabled = enabled

    def setParallelLineMagnetConfig(self, config: list[dict]):
        self._parallel_line_magnet_config = config

    def setTextBoundingMagnetConfig(self, config: list[dict]):
        self._text_bounding_magnet_config = config

    def setLineFitEnabled(self, enabled: bool):
        self._line_fit_enabled = enabled

    def setLineFitMagnetConfig(self, config: list[dict]):
        self._line_fit_magnet_config = config

    def setReferenceMedians(self, medians: dict[str, float | None]):
        self._reference_medians = medians

    def setDarkPixelMagnetEnabled(self, enabled: bool):
        self._dark_pixel_magnet_enabled = enabled

    def setDarkPixelMagnetConfig(self, config: list[dict]):
        self._dark_pixel_magnet_config = config

    def setPendingDrawLabel(self, label: str | None):
        self._pending_draw_label = label

    def getPixelInfo(self, pos: QPointF):
        """Return (R, G, B, Gray) at image position, or None if out of bounds."""
        if self._img_arr_cache is None:
            return None
        x = int(round(pos.x()))
        y = int(round(pos.y()))
        h, w = self._img_arr_cache.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        if self._img_arr_cache.ndim == 3 and self._img_arr_cache.shape[2] >= 3:
            # BGRA order
            b, g, r = (
                int(self._img_arr_cache[y, x, 0]),
                int(self._img_arr_cache[y, x, 1]),
                int(self._img_arr_cache[y, x, 2]),
            )
        else:
            v = int(self._img_arr_cache[y, x])
            r, g, b = v, v, v
        gray = int(self._grayscale_cache[y, x]) if self._grayscale_cache is not None else int(0.299 * r + 0.587 * g + 0.114 * b)
        return r, g, b, gray

    def refreshCursorOverlay(self):
        """Public method to refresh the cursor overlay state."""
        self._updateCursorOverlay()

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
        ]:
            raise ValueError(f"Unsupported createMode: {value}")
        self._createMode = value

    def set_ai_model_name(self, model_name: str) -> None:
        self._osam_session_model_name = model_name

    def _get_osam_session(self) -> OsamSession:
        if (
            self._osam_session is None
            or self._osam_session.model_name != self._osam_session_model_name
        ):
            self._osam_session = OsamSession(model_name=self._osam_session_model_name)
        return self._osam_session

    def _update_shape_with_ai(
        self, points: list[QPointF], point_labels: list[int], shape: Shape
    ) -> None:
        image: np.ndarray = labelme.utils.img_qt_to_arr(img_qt=self.pixmap.toImage())
        response: osam.types.GenerateResponse = self._get_osam_session().run(
            image=imgviz.asrgb(image),
            image_id=str(self._pixmap_hash),
            points=np.array([[p.x(), p.y()] for p in points]),
            point_labels=np.array(point_labels),
        )
        _update_shape_with_ai_response(
            response=response,
            shape=shape,
            createMode=self.createMode,
        )

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)
        self.shapesRedoStack.clear()

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        redone = self.shapesBackups.pop()  # latest
        self.shapesRedoStack.append(redone)

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.sortShapesByArea()
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def redoShape(self):
        if not self.isShapeRedoable:
            return
        shapesRedo = self.shapesRedoStack.pop()
        self.shapes = shapesRedo
        self.sortShapesByArea()
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    @property
    def isShapeRedoable(self):
        return len(self.shapesRedoStack) > 0

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        if self._cursor_debug:
            self._log_cursor_state("enterEvent")
        if self._vertex_dragging:
            self.setCursor(self._blank_cursor)
            self._update_status()
            return
        self.overrideCursor(self._cursor)
        # Restore cursor overlay when mouse enters (e.g., after dialog closes)
        self._updateCursorOverlay()
        self._update_status()

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        if self._cursor_debug:
            self._log_cursor_state("leaveEvent:before")
        self.unHighlight()
        self.restoreCursor()
        self._cursor_overlay.hideCursor()
        if self._cursor_debug:
            self._log_cursor_state("leaveEvent:after")
        self._update_status()

    def focusOutEvent(self, a0: QtGui.QFocusEvent) -> None:
        if self._cursor_debug:
            self._log_cursor_state("focusOutEvent:before")
        self.restoreCursor()
        if self._cursor_debug:
            self._log_cursor_state("focusOutEvent:after")
        self._update_status()

    def focusInEvent(self, a0: QtGui.QFocusEvent) -> None:
        if self._cursor_debug:
            self._log_cursor_state("focusInEvent")
        # Restore cursor overlay when focus returns (e.g., after dialog closes)
        self._updateCursorOverlay()
        self._update_status()

    def isVisible(self, shape):  # type: ignore[override]
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == CanvasMode.CREATE

    def editing(self):
        return self.mode == CanvasMode.EDIT

    def setEditing(self, value=True):
        self.mode = CanvasMode.EDIT if value else CanvasMode.CREATE
        if self.mode == CanvasMode.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            # Keep highlight for selected point shapes during context menu
            if not (
                self._context_menu_active
                and self.hShape.shape_type == "point"
                and self.hShape in self.selectedShapes
            ):
                self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.prevhEdgeMidpoint = self.hEdgeMidpoint
        self.hShape = self.hVertex = self.hEdge = None
        self.hEdgeMidpoint = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def _update_status(self, extra_messages: list[str] | None = None) -> None:
        messages: list[str] = []
        if self.drawing():
            messages.append(self.tr("Creating %r") % self.createMode)
            messages.append(self._get_create_mode_message())
            if self.current:
                messages.append(self.tr("ESC to cancel"))
            if self.canCloseShape():
                messages.append(self.tr("Enter or Space to finalize"))
        else:
            assert self.editing()
            messages.append(self.tr("Editing shapes"))
        if extra_messages:
            messages.extend(extra_messages)
        self.statusUpdated.emit(" • ".join(messages))

    def _get_create_mode_message(self) -> str:
        assert self.drawing()
        isNew: bool = self.current is None
        if self.createMode == "ai_polygon":
            return self.tr(
                "Click points to include or Shift+Click to exclude for ai_polygon"
            )
        if self.createMode == "ai_mask":
            return self.tr(
                "Click points to include or Shift+Click to exclude for ai_mask"
            )
        if self.createMode == "line":
            if isNew:
                return self.tr("Click start point for line")
            else:
                return self.tr("Click end point for line")
        if self.createMode == "linestrip":
            if isNew:
                return self.tr("Click start point for linestrip")
            else:
                return self.tr(
                    "Click next point or finish by Ctrl/Cmd+Click for linestrip"
                )
        if self.createMode == "circle":
            if isNew:
                return self.tr("Click center point for circle")
            else:
                return self.tr("Click point on circumference for circle")
        if self.createMode == "rectangle":
            if isNew:
                return self.tr("Click first corner for rectangle")
            else:
                return self.tr("Click opposite corner for rectangle (Shift for square)")
        return self.tr("Click to add point")

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        """Update line with last point and current coordinates."""
        try:
            pos = self.transformPos(a0.localPos())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos

        is_shift_pressed = a0.modifiers() & Qt.ShiftModifier

        if self._vertex_dragging:
            if self._cursor_debug:
                self._log_cursor_state("mouseMoveEvent:vertex_dragging")
            # Always force blank cursor while dragging a vertex
            self._force_blank_cursor()
            # Dark pixel magnet: snap to dark pixel (polygon shapes only)
            if (
                not is_shift_pressed
                and self.hShape is not None
                and self.hShape.shape_type == "polygon"
            ):
                label = self.hShape.label
                snapped = self._snap_to_dark_pixel(pos, label)
                if snapped is not pos:
                    self._dpm_ghost_pos = pos  # ghost at real mouse position
                    pos = snapped
                else:
                    self._dpm_ghost_pos = None
            else:
                self._dpm_ghost_pos = None
            self.prevMovePoint = pos  # Update for crosshair drawing
            self.boundedMoveVertex(pos, is_shift_pressed=is_shift_pressed)
            self._updateCursorOverlay()
            self.repaint()
            self.movingShape = True
            return

        if self._is_dragging:
            self.overrideCursor(CURSOR_GRAB)
            delta: QPointF = pos - self._dragging_start_pos
            self.scrollRequest.emit(int(delta.x()), Qt.Horizontal)
            self.scrollRequest.emit(int(delta.y()), Qt.Vertical)
            return

        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode

            if self.current or self.createMode in ["point", "polygon", "rectangle"]:
                # Hide cursor when drawing (show crosshair instead)
                if self._custom_cursor_enabled:
                    self.overrideCursor(self._blank_cursor)
                else:
                    self.overrideCursor(CURSOR_DRAW)
            else:
                self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self._updateCursorOverlay()  # Update cursor overlay (no repaint needed)
                self._update_status()
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
                # Hide cursor when near start point
                if not self._near_start_point:
                    self._force_blank_cursor()
                self._near_start_point = True
            else:
                # Restore cursor when moving away from start point
                if self._near_start_point:
                    self._unhide_os_cursor()
                    self.restoreCursor()
                    self.overrideCursor(CURSOR_DRAW)
                self._near_start_point = False
            # Dark pixel magnet during drawing (polygon only)
            if (
                not is_shift_pressed
                and not self._near_start_point
                and self.createMode == "polygon"
            ):
                snapped = self._snap_to_dark_pixel(pos, self._pending_draw_label)
                if snapped is not pos:
                    self._dpm_ghost_pos = pos  # ghost at real mouse position
                    pos = snapped
                    self.prevMovePoint = pos
                else:
                    self._dpm_ghost_pos = None
            else:
                self._dpm_ghost_pos = None
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode == "rectangle":
                if is_shift_pressed:
                    self.prevMovePoint = pos = _snap_cursor_pos_for_square(  # override
                        pos=pos, opposite_vertex=self.current[0]
                    )
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self._updateCursorOverlay()
            self.repaint()
            self.current.highlightClear()
            self._update_status()
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & a0.buttons():
            if self.hEdgeMidpoint is not None and self.hShape is not None:
                # Moving rectangle edge midpoint
                if self._edge_midpoint_dragging:
                    self._force_blank_cursor()
                    self.prevMovePoint = pos  # Update for grid line drawing
                self.boundedMoveEdge(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint is not None:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        status_messages: list[str] = []
        for shape in self._shapes_hover_order:
            if not self.isVisible(shape):
                continue
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.hShape and self.hShape is not shape:
                    self.hShape.highlightClear()
                elif self.selectedVertex() and self.hShape:
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhEdgeMidpoint = self.hEdgeMidpoint
                self.hEdgeMidpoint = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                status_messages.append(self.tr("Click & drag to move point"))
                if shape.canRemovePoint():
                    status_messages.append(
                        self.tr("ALT + SHIFT + Click to delete point")
                    )
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.hShape and self.hShape is not shape:
                    self.hShape.highlightClear()
                elif self.selectedVertex() and self.hShape:
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhEdgeMidpoint = self.hEdgeMidpoint
                self.hEdgeMidpoint = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                status_messages.append(self.tr("ALT + Click to create point on shape"))
                break
            # Check for rectangle edge midpoint
            edge_midpoint = shape.nearestEdgeMidpoint(pos, self.epsilon)
            if edge_midpoint is not None:
                if self.hShape and self.hShape is not shape:
                    self.hShape.highlightClear()
                elif self.selectedVertex() and self.hShape:
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdgeMidpoint = self.hEdgeMidpoint = edge_midpoint
                shape.highlightEdgeMidpoint(edge_midpoint)
                self.overrideCursor(CURSOR_POINT)
                status_messages.append(self.tr("Click & drag to resize rectangle"))
                break
            elif shape.containsPoint(pos):
                if self.hShape and self.hShape is not shape:
                    self.hShape.highlightClear()
                elif self.selectedVertex() and self.hShape:
                    self.hShape.highlightClear()
                elif self.hEdgeMidpoint is not None and self.hShape:
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.prevhEdgeMidpoint = self.hEdgeMidpoint
                self.hEdgeMidpoint = None
                status_messages.extend(
                    [
                        self.tr("Click & drag to move shape"),
                        self.tr("Right-click & drag to copy shape"),
                    ]
                )
                self.overrideCursor(CURSOR_GRAB)
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
            # Restore to default cursor if not already default
            if self._cursor != CURSOR_DEFAULT:
                self._unhide_os_cursor()
                self._cursor = CURSOR_DEFAULT
                if QtWidgets.QApplication.overrideCursor() is not None:
                    QtWidgets.QApplication.restoreOverrideCursor()
                self.setCursor(CURSOR_DEFAULT)
        # Update hover label state - reset on any cursor movement
        self._hover_label_shape = self.hShape
        self._hover_label_ready = False
        self._hover_label_timer.stop()
        if self.hShape is not None:
            self._hover_label_last_pos = pos
            self._hover_label_timer.start()
        self.update()  # Repaint to hide label immediately

        self.vertexSelected.emit(self.hVertex is not None)
        self._update_status(extra_messages=status_messages)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.touch()  # Update modification timestamp
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.touch()  # Update modification timestamp
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self._mouse_pressed = True
        # Hide hover label on click
        self._hover_label_ready = False
        self._hover_label_timer.stop()
        if self._cursor_debug:
            self._log_cursor_state(f"mousePressEvent:{a0.button()}")

        pos: QPointF = self.transformPos(a0.localPos())

        is_shift_pressed = a0.modifiers() & Qt.ShiftModifier

        if a0.button() == Qt.LeftButton:
            if self.drawing():
                self._undone_points.clear()
                redo_action = getattr(self, "_redo_action", None)
                if redo_action is not None:
                    redo_action.setEnabled(False)
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(a0.modifiers()) == Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if a0.modifiers() & Qt.ControlModifier:
                            self.finalise()
                    if self.current is not None:
                        # Hide cursor immediately on click during polygon creation
                        self._force_blank_cursor()
                        self.repaint()
                elif not self.outOfPixmap(pos):
                    if self.createMode in ["ai_polygon", "ai_mask"]:
                        if not download_ai_model(
                            model_name=self._osam_session_model_name, parent=self
                        ):
                            return

                    # Create new shape.
                    self.current = Shape(
                        shape_type="points"
                        if self.createMode in ["ai_polygon", "ai_mask"]
                        else self.createMode
                    )
                    self.current._is_creating = True  # Mark as being created
                    # Dark pixel magnet on first click (polygon only)
                    if (
                        not is_shift_pressed
                        and self.createMode == "polygon"
                    ):
                        snapped = self._snap_to_dark_pixel(
                            pos, self._pending_draw_label
                        )
                        if snapped is not pos:
                            pos = snapped
                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    elif (
                        self.createMode in ["ai_polygon", "ai_mask"]
                        and a0.modifiers() & Qt.ControlModifier
                    ):
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
                        # Hide cursor immediately on first click of creation
                        self._force_blank_cursor()
            elif self.editing():
                if self.selectedEdge() and a0.modifiers() == Qt.AltModifier:
                    self.addPointToEdge()
                elif self.selectedVertex() and a0.modifiers() == (
                    Qt.AltModifier | Qt.ShiftModifier
                ):
                    self.removeSelectedPoint()

                # If no hover vertex is set, resolve the nearest vertex on click
                if self.hVertex is None:
                    for shape in self._shapes_hover_order:
                        if not self.isVisible(shape):
                            continue
                        index = shape.nearestVertex(pos, self.epsilon)
                        if index is not None:
                            if self.selectedVertex() and self.hShape:
                                self.hShape.highlightClear()
                            self.prevhVertex = self.hVertex = index
                            self.prevhShape = self.hShape = shape
                            self.prevhEdge = self.hEdge
                            self.hEdge = None
                            self.prevhEdgeMidpoint = self.hEdgeMidpoint
                            self.hEdgeMidpoint = None
                            shape.highlightVertex(index, shape.MOVE_VERTEX)
                            break

                group_mode = int(a0.modifiers()) == Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                # Start vertex dragging if a vertex is selected
                if self.hVertex is not None:
                    self._vertex_dragging = True
                    Shape.hide_vertex_outline = True  # Hide vertex outline during drag
                    self.prevMovePoint = pos  # Set immediately for crosshair
                    self._force_blank_cursor()
                    self._updateCursorOverlay()  # Show cursor overlay immediately
                # Start edge midpoint dragging if an edge midpoint is selected
                elif self.hEdgeMidpoint is not None:
                    self._edge_midpoint_dragging = True
                    self._dragging_edge_index = self.hEdgeMidpoint
                    self._edge_midpoint_drag_shape = self.hShape  # Track shape for flag restore
                    self.hShape._hide_edge_midpoint = True  # Hide edge midpoint during drag
                    self.prevMovePoint = pos  # Set immediately for grid line
                    self._force_blank_cursor()
                    self._updateCursorOverlay()  # Show cursor overlay immediately
                self.repaint()
        elif a0.button() == Qt.RightButton:
            if self.drawing() and not self._right_click_edit_enabled:
                # Show context menu during drawing (undo last point, etc.)
                menu = self.menus[0]
                undo_action = getattr(self, "_undo_action", None)
                redo_action = getattr(self, "_redo_action", None)
                orig_undo_text = None
                orig_redo_text = None
                if undo_action is not None and self.current:
                    orig_undo_text = undo_action.text()
                    undo_action.setText("元に戻す（最後の頂点を取り消し）")
                if redo_action is not None and self.current and self._undone_points:
                    orig_redo_text = redo_action.text()
                    redo_action.setText("やり直す（最後の頂点を再適用）")
                menu.exec_(self.mapToGlobal(a0.pos()))
                if undo_action is not None and orig_undo_text is not None:
                    undo_action.setText(orig_undo_text)
                if redo_action is not None and orig_redo_text is not None:
                    redo_action.setText(orig_redo_text)
            elif self.drawing():
                # Switch from create mode to edit mode
                self.current = None
                self.line.points = []
                self.line.point_labels = []
                self.drawingPolygon.emit(False)
                # Reset near-start-point flag
                self._near_start_point = False
                self.setEditing(True)
                self.editModeChanged.emit(True)  # Notify app.py
                # Restore cursor when exiting creation mode
                # Force show OS cursor regardless of flag
                self._os_cursor_hidden = True  # Ensure _unhide_os_cursor works
                self._unhide_os_cursor()
                # Clear all override cursors
                while QtWidgets.QApplication.overrideCursor() is not None:
                    QtWidgets.QApplication.restoreOverrideCursor()
                self._cursor = CURSOR_DEFAULT
                self._clear_parent_viewport_cursor()
                # Hide cursor overlay
                self._cursor_overlay.hideCursor()
                self.repaint()
                # Delay cursor restore to ensure it takes effect
                QtCore.QTimer.singleShot(50, lambda: self.setCursor(CURSOR_DEFAULT))
            elif self.editing():
                group_mode = int(a0.modifiers()) == Qt.ControlModifier
                if not self.selectedShapes or (
                    self.hShape is not None and self.hShape not in self.selectedShapes
                ):
                    self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                    self.repaint()
                # Highlight selected point shapes during right-click context menu
                for shape in self.selectedShapes:
                    if shape.shape_type == "point":
                        shape.highlightVertex(0, shape.MOVE_VERTEX)
                self._context_menu_active = True
                self.repaint()
                # Show context menu immediately on press
                menu = self.menus[0]
                menu.exec_(self.mapToGlobal(a0.pos()))
                # Clean up after menu closes
                self._context_menu_active = False
                for shape in self.selectedShapes:
                    if shape.shape_type == "point":
                        shape.highlightClear()
                self.repaint()
        elif a0.button() == Qt.MiddleButton and self._is_dragging_enabled:
            self.overrideCursor(CURSOR_GRAB)
            self._dragging_start_pos = pos
            self._is_dragging = True
        self._update_status()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self._mouse_pressed = False
        if self._cursor_debug:
            self._log_cursor_state(f"mouseReleaseEvent:{a0.button()}:before")

        if a0.button() == Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )
        elif a0.button() == Qt.MiddleButton:
            self._is_dragging = False
            self.restoreCursor()

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.hShape.touch()  # Update modification timestamp
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False
        self._dpm_ghost_pos = None  # Clear ghost cursor
        # End vertex dragging and restore cursor
        if self._vertex_dragging:
            if self.hShape:
                self.hShape.touch()  # Update modification timestamp
            self._vertex_dragging = False
            Shape.hide_vertex_outline = False  # Restore vertex outline
            # Restore all stacked cursors from drag
            while QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._cursor = CURSOR_DEFAULT
            self.unsetCursor()
            self._clear_parent_viewport_cursor()
            self._unhide_os_cursor()
            # Hide cursor overlay after vertex drag
            self._cursor_overlay.hideCursor()
            self.repaint()
            # After drag, if still over a vertex, show pointing hand immediately
            if self.hVertex is not None:
                self._force_point_cursor()
        # End edge midpoint dragging and restore cursor
        if self._edge_midpoint_dragging:
            if self.hShape:
                self.hShape.touch()  # Update modification timestamp
            # Restore edge midpoint on the shape that started the drag
            drag_shape = getattr(self, "_edge_midpoint_drag_shape", None)
            if drag_shape is not None:
                drag_shape._hide_edge_midpoint = False
                self._edge_midpoint_drag_shape = None
            elif self.hShape:
                self.hShape._hide_edge_midpoint = False
            self._edge_midpoint_dragging = False
            self._dragging_edge_index = None
            self._snap_active = False
            self._snap_line_pos = None
            self._text_bounding_snap_dots = None
            self._tb_boundary_cache = None
            self._tb_snap_entered = False
            self._pl_snap_cache = None
            self._pl_snap_entered = False
            # Restore all stacked cursors from drag
            while QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._cursor = CURSOR_DEFAULT
            self.unsetCursor()
            self._clear_parent_viewport_cursor()
            self._unhide_os_cursor()
            self.repaint()
            # After drag, if still over an edge midpoint, show pointing hand
            if self.hEdgeMidpoint is not None:
                self._force_point_cursor()
        if self._cursor_debug:
            self._log_cursor_state(f"mouseReleaseEvent:{a0.button()}:after")
        self._update_status()

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.sortShapesByArea()
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self) -> bool:
        if not self.drawing():
            return False
        if not self.current:
            return False
        if self.createMode in ["ai_polygon", "ai_mask"]:
            return True
        if self.createMode == "linestrip":
            return len(self.current) >= 2
        return len(self.current) >= 3

    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.double_click != "close":
            return

        if self.canCloseShape():
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.hVertex is not None:
            assert self.hShape is not None
            self.hShape.highlightVertex(i=self.hVertex, action=self.hShape.MOVE_VERTEX)
            self.setHiding()
            # Select the shape when clicking on its vertex
            if self.hShape not in self.selectedShapes:
                if multiple_selection_mode:
                    self.selectionChanged.emit(self.selectedShapes + [self.hShape])
                else:
                    self.selectionChanged.emit([self.hShape])
                self.hShapeIsSelected = False
            else:
                self.hShapeIsSelected = True
            self.calculateOffsets(point)
            return
        elif self.hEdgeMidpoint is not None:
            # For rectangle edge midpoint, select the shape (deselect others)
            assert self.hShape is not None
            self.hShape.highlightEdgeMidpoint(self.hEdgeMidpoint)
            self.setHiding()
            if self.hShape not in self.selectedShapes:
                if multiple_selection_mode:
                    self.selectionChanged.emit(self.selectedShapes + [self.hShape])
                else:
                    self.selectionChanged.emit([self.hShape])
                self.hShapeIsSelected = False
            else:
                self.hShapeIsSelected = True
            self.calculateOffsets(point)
            return
        else:
            shape: Shape
            # Sort by area ascending (smallest first), points always first
            # This ensures smaller objects are selected over larger ones
            for shape in self._shapes_hover_order:
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point: QPointF) -> None:
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    def boundedMoveVertex(self, pos: QPointF, is_shift_pressed: bool) -> None:
        if self.hVertex is None:
            logger.warning("hVertex is None, so cannot move vertex: pos=%r", pos)
            return
        assert self.hShape is not None

        if self.hVertex >= len(self.hShape.points):
            logger.warning("hVertex %d out of range (len=%d), resetting", self.hVertex, len(self.hShape.points))
            self.hVertex = None
            return

        point: QPointF = self.hShape[self.hVertex]

        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)

        if is_shift_pressed and self.hShape.shape_type == "rectangle":
            pos = _snap_cursor_pos_for_square(
                pos=pos, opposite_vertex=self.hShape[1 - self.hVertex]
            )

        self.hShape.moveVertexBy(i=self.hVertex, offset=pos - point)

    def boundedMoveEdge(self, pos: QPointF) -> None:
        """Move a rectangle edge to resize the rectangle."""
        if self.hEdgeMidpoint is None or self.hShape is None:
            return

        if self.outOfPixmap(pos):
            return

        snap_pos = pos
        self._snap_active = False
        self._snap_line_pos = None
        self._text_bounding_snap_dots = None

        # Shift held → cancel all snap adjustments
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        shift_held = bool(modifiers & Qt.ShiftModifier)

        if (
            shift_held
            or self._grayscale_cache is None
            or self.hShape.shape_type != "rectangle"
        ):
            self.hShape.moveEdgeTo(self.hEdgeMidpoint, snap_pos)
            return

        # Parallel line snap: try each rule in order
        if self._parallel_line_dist_enabled:
            for i, rule in enumerate(self._parallel_line_magnet_config):
                if self.hShape.label != rule.get("target_label"):
                    continue
                margin = self._reference_medians.get(f"pl:{i}")
                if margin is None:
                    continue
                result = self._detect_parallel_line_snap(
                    rule, margin, self.hShape, self.hEdgeMidpoint, pos,
                )
                if result is not None:
                    snap_pos, line_pos = result
                    self._snap_active = True
                    self._snap_line_pos = line_pos
                    if self._pl_snap_entered:
                        self._warp_cursor_to_image_pos(snap_pos)
                    break

        # Line fit snap: snap edge directly onto detected line peak
        if not self._snap_active and self._line_fit_enabled:
            for rule in self._line_fit_magnet_config:
                if self.hShape.label != rule.get("target_label"):
                    continue
                result = self._detect_line_fit_snap(
                    rule, self.hShape, self.hEdgeMidpoint, pos,
                )
                if result is not None:
                    snap_pos, line_pos = result
                    self._snap_active = True
                    self._snap_line_pos = line_pos
                    if self._lf_snap_entered:
                        self._warp_cursor_to_image_pos(snap_pos)
                    break

        # Text bounding snap: try each rule (only if no other snap active)
        if not self._snap_active and self._text_bounding_enabled:
            for i, rule in enumerate(self._text_bounding_magnet_config):
                if self.hShape.label != rule.get("target_label"):
                    continue
                result = self._detect_text_bounding_snap(
                    rule, i, self.hShape, self.hEdgeMidpoint, pos,
                )
                if result is not None:
                    snap_pos, dots = result
                    self._text_bounding_snap_dots = dots
                    # On first snap frame, warp cursor to snap position
                    if self._tb_snap_entered:
                        self._warp_cursor_to_image_pos(snap_pos)
                    break
        # (else: parallel line snap active or text bounding disabled — skip)

        self.hShape.moveEdgeTo(self.hEdgeMidpoint, snap_pos)

    # -- Line fit magnet defaults --
    _LF_DEFAULTS = {
        "luminance_threshold": 128,
        "resize_base": 2560,
        "snap_range_pixels": 5,
        "sample_points": 11,
        "consecutive_window": 8,
        "distance_tolerance": 1.0,
    }

    def _detect_line_fit_snap(
        self,
        rule: dict,
        shape,
        edge_index: int,
        cursor_pos: QPointF,
    ) -> tuple[QPointF, float] | None:
        """Detect a line near the edge and snap directly onto its luminance peak.

        Scans both perpendicular directions from the edge. Finds the darkest
        point (luminance peak) using parabolic sub-pixel interpolation.
        Returns (snapped_pos, peak_pos_for_guide) or None.
        """
        from labelme.shape import Shape

        d = self._LF_DEFAULTS
        snap_range_px = rule.get("snap_range_pixels", d["snap_range_pixels"])
        sample_points = rule.get("sample_points", d["sample_points"])
        consec_window = rule.get("consecutive_window", d["consecutive_window"])
        dist_tol = rule.get("distance_tolerance", d["distance_tolerance"])
        lum_thresh = rule.get("luminance_threshold", d["luminance_threshold"])
        resize_base = rule.get("resize_base", d["resize_base"])

        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        scale = max(img_w, img_h) / resize_base
        snap_window = snap_range_px * scale
        max_scan = max(int(snap_window * 3), 50)

        is_horiz = edge_index in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM)
        cursor_val = cursor_pos.y() if is_horiz else cursor_pos.x()

        # Cache: stay locked within snap zone
        cache = self._lf_snap_cache
        if cache is not None and cache[0] == edge_index:
            cached_snap_val = cache[1]
            cached_peak_pos = cache[2]
            if abs(cursor_val - cached_snap_val) <= snap_window:
                self._lf_snap_entered = False
                if is_horiz:
                    return (QPointF(cursor_pos.x(), cached_snap_val), cached_peak_pos)
                else:
                    return (QPointF(cached_snap_val, cursor_pos.y()), cached_peak_pos)
            else:
                self._lf_snap_cache = None
                self._lf_snap_entered = False

        p0, p1 = shape.points[0], shape.points[1]
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())

        if is_horiz:
            xs = np.linspace(left, right, sample_points + 2)[1:-1]
            iy = int(round(cursor_pos.y()))
            if iy < 0 or iy >= img_h:
                self._lf_snap_entered = False
                return None
            peak = self._find_line_peak_h(
                xs, iy, max_scan, grayscale, img_h, img_w,
                consec_window, dist_tol, lum_thresh,
            )
            if peak is not None:
                dist = abs(peak - cursor_pos.y())
                if dist <= snap_window:
                    self._lf_snap_cache = (edge_index, peak, peak)
                    self._lf_snap_entered = True
                    return (QPointF(cursor_pos.x(), peak), peak)
        else:
            ys = np.linspace(top, bottom, sample_points + 2)[1:-1]
            ix = int(round(cursor_pos.x()))
            if ix < 0 or ix >= img_w:
                self._lf_snap_entered = False
                return None
            peak = self._find_line_peak_v(
                ys, ix, max_scan, grayscale, img_h, img_w,
                consec_window, dist_tol, lum_thresh,
            )
            if peak is not None:
                dist = abs(peak - cursor_pos.x())
                if dist <= snap_window:
                    self._lf_snap_cache = (edge_index, peak, peak)
                    self._lf_snap_entered = True
                    return (QPointF(peak, cursor_pos.y()), peak)

        self._lf_snap_entered = False
        return None

    def _find_line_peak_h(self, xs, iy, max_scan, grayscale, img_h, img_w,
                          consec_window, dist_tol, lum_thresh):
        """Find horizontal line peak by scanning both vertical directions.

        For each sample x, scans up and down from iy to find the darkest
        pixel with luminance <= lum_thresh (absolute threshold).
        Uses parabolic interpolation for sub-pixel valley.
        Returns median peak y-position or None.
        """
        n = len(xs)
        w = min(consec_window, n)
        hits: list[tuple[bool, float]] = []

        for idx in range(n):
            col = int(round(xs[idx]))
            if col < 0 or col >= img_w:
                hits.append((False, -1.0))
                continue
            # Check center pixel
            center_lum = float(grayscale[iy, col])
            candidates = []
            if center_lum <= lum_thresh:
                candidates.append((center_lum, iy, 0))
            # Scan both directions independently
            for scan_dir in (+1, -1):
                dir_best_lum = 256.0
                dir_best_y = -1
                entered_dark = False
                for d in range(1, max_scan + 1):
                    sy = iy + scan_dir * d
                    if sy < 0 or sy >= img_h:
                        break
                    lum = float(grayscale[sy, col])
                    if lum <= lum_thresh:
                        entered_dark = True
                        if lum < dir_best_lum:
                            dir_best_lum = lum
                            dir_best_y = sy
                    # Stop after passing through dark region
                    if entered_dark and lum > lum_thresh:
                        break
                if dir_best_y >= 0:
                    candidates.append((dir_best_lum, dir_best_y, abs(dir_best_y - iy)))

            if candidates:
                # Pick nearest to cursor; tie-break by darkest
                candidates.sort(key=lambda c: (c[2], c[0]))
                best_y = candidates[0][1]
                # Parabolic sub-pixel interpolation around the valley
                y0 = best_y
                ym1 = max(0, y0 - 1)
                yp1 = min(img_h - 1, y0 + 1)
                v_m1 = float(grayscale[ym1, col])
                v_0 = float(grayscale[y0, col])
                v_p1 = float(grayscale[yp1, col])
                denom = v_m1 - 2.0 * v_0 + v_p1
                if abs(denom) > 1e-6:
                    offset = 0.5 * (v_m1 - v_p1) / denom
                    offset = max(-0.5, min(0.5, offset))
                else:
                    offset = 0.0
                refined = float(y0) + offset + 0.5  # +0.5 for pixel center
                hits.append((True, refined))
            else:
                hits.append((False, -1.0))

        # Consensus: cascade from strict to relaxed tolerance
        min_hits = max(w - 1, (w + 1) // 2)
        for tol in (dist_tol * 0.5, dist_tol):
            for start in range(max(n - w + 1, 1)):
                end = min(start + w, n)
                window = [hits[i][1] for i in range(start, end) if hits[i][0]]
                if len(window) < min_hits:
                    continue
                median = float(np.median(window))
                agree = [v for v in window if abs(v - median) <= tol]
                if len(agree) >= min_hits:
                    return float(np.median(agree))
        return None

    def _find_line_peak_v(self, ys, ix, max_scan, grayscale, img_h, img_w,
                          consec_window, dist_tol, lum_thresh):
        """Find vertical line peak by scanning both horizontal directions.

        Same logic as _find_line_peak_h but scanning left/right.
        Uses absolute luminance threshold.
        Returns median peak x-position or None.
        """
        n = len(ys)
        w = min(consec_window, n)
        hits: list[tuple[bool, float]] = []

        for idx in range(n):
            row = int(round(ys[idx]))
            if row < 0 or row >= img_h:
                hits.append((False, -1.0))
                continue
            # Check center pixel
            center_lum = float(grayscale[row, ix])
            candidates = []
            if center_lum <= lum_thresh:
                candidates.append((center_lum, ix, 0))
            # Scan both directions independently
            for scan_dir in (+1, -1):
                dir_best_lum = 256.0
                dir_best_x = -1
                entered_dark = False
                for d in range(1, max_scan + 1):
                    sx = ix + scan_dir * d
                    if sx < 0 or sx >= img_w:
                        break
                    lum = float(grayscale[row, sx])
                    if lum <= lum_thresh:
                        entered_dark = True
                        if lum < dir_best_lum:
                            dir_best_lum = lum
                            dir_best_x = sx
                    # Stop after passing through dark region
                    if entered_dark and lum > lum_thresh:
                        break
                if dir_best_x >= 0:
                    candidates.append((dir_best_lum, dir_best_x, abs(dir_best_x - ix)))

            if candidates:
                # Pick nearest to cursor; tie-break by darkest
                candidates.sort(key=lambda c: (c[2], c[0]))
                best_x = candidates[0][1]
                x0 = best_x
                xm1 = max(0, x0 - 1)
                xp1 = min(img_w - 1, x0 + 1)
                v_m1 = float(grayscale[row, xm1])
                v_0 = float(grayscale[row, x0])
                v_p1 = float(grayscale[row, xp1])
                denom = v_m1 - 2.0 * v_0 + v_p1
                if abs(denom) > 1e-6:
                    offset = 0.5 * (v_m1 - v_p1) / denom
                    offset = max(-0.5, min(0.5, offset))
                else:
                    offset = 0.0
                refined = float(x0) + offset + 0.5
                hits.append((True, refined))
            else:
                hits.append((False, -1.0))

        min_hits = max(w - 1, (w + 1) // 2)
        for tol in (dist_tol * 0.5, dist_tol):
            for start in range(max(n - w + 1, 1)):
                end = min(start + w, n)
                window = [hits[i][1] for i in range(start, end) if hits[i][0]]
                if len(window) < min_hits:
                    continue
                median = float(np.median(window))
                agree = [v for v in window if abs(v - median) <= tol]
                if len(agree) >= min_hits:
                    return float(np.median(agree))
        return None

    # -- Parallel line magnet detection defaults --
    _PL_DEFAULTS = {
        "luminance_threshold": 128,
        "sample_points": 11,
        "consecutive_window": 8,
        "distance_tolerance": 0.5,
        "snap_range_pixels": 5,
        "resize_base": 2560,
        "margin_pixels": 10,
    }
    # -- Text bounding snap detection defaults --
    _TB_DEFAULTS = {
        "luminance_threshold": 30,
        "min_agreement": 6,
        "snap_range_pixels": 3,
        "resize_base": 2560,
        "margin_pixels": 0.5,
    }
    # -- Dark pixel magnet defaults --
    _DPM_DEFAULTS = {
        "luminance_threshold": 128,
        "resize_base": 640,
        "snap_range_pixels": 20,
    }

    @staticmethod
    def _bilinear_lum(
        grayscale: np.ndarray, x: float, y: float, w: int, h: int
    ) -> float:
        """Bilinear interpolation of grayscale value at sub-pixel (x, y)."""
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        x0 = max(0, x0)
        y0 = max(0, y0)
        fx = x - x0
        fy = y - y0
        return float(
            grayscale[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + grayscale[y0, x1] * fx * (1.0 - fy)
            + grayscale[y1, x0] * (1.0 - fx) * fy
            + grayscale[y1, x1] * fx * fy
        )

    def _snap_to_dark_pixel(self, pos: QPointF, label: str | None) -> QPointF:
        """Snap pos to nearest dark pixel if a matching rule exists.

        Returns the same pos object if no snap needed, or a new QPointF if snapped.
        Caller can use ``snapped is not pos`` to detect whether snapping occurred.

        Performance strategy:
          - Reachable map (integer resolution, per threshold) is precomputed once
            per image and cached in ``_dpm_reachable_cache``.
          - Per frame: slice the cached map, find nearest reachable integer pixel,
            then refine with a small 0.1px grid (±2px around hit).
        """
        if (
            self._grayscale_cache is None
            or not self._dark_pixel_magnet_enabled
            or not self._dark_pixel_magnet_config
            or label is None
        ):
            return pos

        # Find matching rule
        rule = None
        for r in self._dark_pixel_magnet_config:
            if r.get("target_label") == label:
                rule = r
                break
        if rule is None:
            return pos

        d = self._DPM_DEFAULTS
        lum_thresh = rule.get("luminance_threshold", d["luminance_threshold"])
        resize_base = rule.get("resize_base", d["resize_base"])
        snap_range = rule.get("snap_range_pixels", d["snap_range_pixels"])

        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        img_scale = max(img_w, img_h) / resize_base
        actual_half = snap_range * img_scale / 2.0

        # Activation: centre-pixel bilinear check.
        cx, cy = pos.x(), pos.y()
        cxc = max(0.0, min(cx, float(img_w - 1)))
        cyc = max(0.0, min(cy, float(img_h - 1)))
        if self._bilinear_lum(grayscale, cxc, cyc, img_w, img_h) <= lum_thresh:
            return pos

        # --- Precomputed reachable map (cached per image + threshold) ---
        if self._dpm_reachable_hash != self._pixmap_hash:
            self._dpm_reachable_cache = {}
            self._dpm_reachable_hash = self._pixmap_hash

        if lum_thresh not in self._dpm_reachable_cache:
            binary = (grayscale <= lum_thresh).astype(np.float32)
            # K=3 box conv at integer level → ±1px reach (≈ ±0.9px at sub-pixel)
            K = 3
            KH = 1
            H, W = binary.shape
            padded = np.pad(binary, KH, mode="constant", constant_values=0.0)
            ii = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
            conv = (
                ii[K : H + K, K : W + K]
                - ii[:H, K : W + K]
                - ii[K : H + K, :W]
                + ii[:H, :W]
            )
            self._dpm_reachable_cache[lum_thresh] = conv > 0.0

        reachable_full = self._dpm_reachable_cache[lum_thresh]

        # Extract snap area from precomputed map
        ix_lo = max(0, int(cx - actual_half) - 1)
        ix_hi = min(img_w - 1, int(cx + actual_half) + 1)
        iy_lo = max(0, int(cy - actual_half) - 1)
        iy_hi = min(img_h - 1, int(cy + actual_half) + 1)

        patch = reachable_full[iy_lo : iy_hi + 1, ix_lo : ix_hi + 1]
        if not patch.any():
            return pos

        # Find nearest reachable integer pixel (only iterate reachable pixels)
        ry, rx = np.where(patch)
        rx_abs = rx.astype(np.float32) + ix_lo
        ry_abs = ry.astype(np.float32) + iy_lo
        dist_c = (rx_abs - cx) ** 2 + (ry_abs - cy) ** 2
        best_idx = int(np.argmin(dist_c))
        hit_x = float(rx_abs[best_idx])
        hit_y = float(ry_abs[best_idx])

        # --- Fine 0.1px grid around coarse hit (±2px) ---
        fine_r = 2.0
        step = 0.1
        fx_lo = max(0.0, hit_x - fine_r)
        fx_hi = min(float(img_w - 1), hit_x + fine_r)
        fy_lo = max(0.0, hit_y - fine_r)
        fy_hi = min(float(img_h - 1), hit_y + fine_r)

        fxs = np.arange(fx_lo, fx_hi + step * 0.5, step, dtype=np.float32)
        fys = np.arange(fy_lo, fy_hi + step * 0.5, step, dtype=np.float32)
        if len(fxs) < 2 or len(fys) < 2:
            return QPointF(hit_x + 0.3, hit_y + 0.5)

        fgy, fgx = np.meshgrid(fys, fxs, indexing="ij")

        # Vectorised bilinear interpolation (float32)
        x0 = fgx.astype(np.int32)
        y0 = fgy.astype(np.int32)
        x1 = np.minimum(x0 + 1, img_w - 1)
        y1 = np.minimum(y0 + 1, img_h - 1)
        fx = fgx - x0
        fy = fgy - y0
        lum = (
            grayscale[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + grayscale[y0, x1] * fx * (1.0 - fy)
            + grayscale[y1, x0] * (1.0 - fx) * fy
            + grayscale[y1, x1] * fx * fy
        )
        del x0, y0, x1, y1, fx, fy

        # Binary threshold + 17x17 box conv at 0.1px (±0.8px reach)
        binary_f = (lum <= lum_thresh).astype(np.float32)
        del lum

        K = 17
        KH = K // 2
        Hf, Wf = binary_f.shape
        padded = np.pad(binary_f, KH, mode="constant", constant_values=0.0)
        ii = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
        conv = (
            ii[K : Hf + K, K : Wf + K]
            - ii[:Hf, K : Wf + K]
            - ii[K : Hf + K, :Wf]
            + ii[:Hf, :Wf]
        )
        del padded, ii, binary_f

        reachable = conv > 0.0
        if not reachable.any():
            return QPointF(hit_x + 0.3, hit_y + 0.5)

        dist_sq = (fgy - cy) ** 2 + (fgx - cx) ** 2
        dist_sq[~reachable] = np.inf

        best = np.unravel_index(int(np.argmin(dist_sq)), dist_sq.shape)
        return QPointF(float(fgx[best]) + 0.3, float(fgy[best]) + 0.5)

    def _detect_parallel_line_snap(
        self,
        rule: dict,
        margin: float,
        shape,
        edge_index: int,
        cursor_pos: QPointF,
    ) -> tuple[QPointF, float] | None:
        """Detect a parallel line and snap the edge at margin distance from it.

        margin is pre-computed from resize_base/margin_pixels and image size.
        Returns (snapped_pos, line_pos_in_image_coords) or None.
        """
        from labelme.shape import Shape

        d = self._PL_DEFAULTS
        snap_range_px = rule.get("snap_range_pixels", d["snap_range_pixels"])
        sample_points = rule.get("sample_points", d["sample_points"])
        consec_window = rule.get("consecutive_window", d["consecutive_window"])
        dist_tol = rule.get("distance_tolerance", d["distance_tolerance"])
        lum_thresh = rule.get("luminance_threshold", d["luminance_threshold"])
        resize_base = rule.get("resize_base", d["resize_base"])

        M = margin
        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        scale = max(img_w, img_h) / resize_base
        snap_window = snap_range_px * scale
        max_scan = max(int(M * 3), 100)

        is_horiz = edge_index in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM)
        cursor_val = cursor_pos.y() if is_horiz else cursor_pos.x()

        # Cache logic: within snap zone → stay locked
        cache = self._pl_snap_cache
        if (
            cache is not None
            and cache[0] == edge_index
        ):
            cached_snap_val = cache[1]
            cached_line_pos = cache[2]
            if abs(cursor_val - cached_snap_val) <= snap_window:
                # Still within snap zone — stay locked
                self._pl_snap_entered = False
                if is_horiz:
                    return (QPointF(cursor_pos.x(), cached_snap_val), cached_line_pos)
                else:
                    return (QPointF(cached_snap_val, cursor_pos.y()), cached_line_pos)
            else:
                # Left snap zone — clear cache, rescan below
                self._pl_snap_cache = None
                self._pl_snap_entered = False

        p0, p1 = shape.points[0], shape.points[1]
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())

        lo = max(0.0, M - snap_window)
        hi = M + snap_window

        if is_horiz:
            if edge_index == Shape.EDGE_BOTTOM:
                shape_edge_val = bottom
                scan_dir = -1
            else:
                shape_edge_val = top
                scan_dir = 1
            xs = np.linspace(left, right, sample_points + 2)[1:-1]
            iy = int(round(shape_edge_val))
            if iy < 0 or iy >= img_h:
                self._pl_snap_entered = False
                return None

            result = self._detect_line_h(
                xs, iy, scan_dir, max_scan, grayscale, img_h, img_w,
                consec_window, dist_tol, lum_thresh,
            )
            if result is not None:
                near_edge, center = result
                near_edge_img = near_edge + 0.5
                center_img = center + 0.5
                dist = abs(near_edge_img - cursor_pos.y())
                if lo <= dist <= hi:
                    snapped_y = near_edge_img - scan_dir * M
                    self._pl_snap_cache = (edge_index, snapped_y, center_img)
                    self._pl_snap_entered = True
                    return (QPointF(cursor_pos.x(), snapped_y), center_img)
            self._pl_snap_entered = False
            return None

        elif edge_index in (Shape.EDGE_LEFT, Shape.EDGE_RIGHT):
            if edge_index == Shape.EDGE_LEFT:
                shape_edge_val = left
                scan_dir = 1
            else:
                shape_edge_val = right
                scan_dir = -1
            ys = np.linspace(top, bottom, sample_points + 2)[1:-1]
            ix = int(round(shape_edge_val))
            if ix < 0 or ix >= img_w:
                self._pl_snap_entered = False
                return None

            result = self._detect_line_v(
                ys, ix, scan_dir, max_scan, grayscale, img_h, img_w,
                consec_window, dist_tol, lum_thresh,
            )
            if result is not None:
                near_edge, center = result
                near_edge_img = near_edge + 0.5
                center_img = center + 0.5
                dist = abs(near_edge_img - cursor_pos.x())
                if lo <= dist <= hi:
                    snapped_x = near_edge_img - scan_dir * M
                    self._pl_snap_cache = (edge_index, snapped_x, center_img)
                    self._pl_snap_entered = True
                    return (QPointF(snapped_x, cursor_pos.y()), center_img)
            self._pl_snap_entered = False
            return None

        self._pl_snap_entered = False
        return None

    def _detect_line_h(self, xs, iy, scan_dir, max_scan, grayscale, img_h, img_w,
                       consec_window, dist_tol, lum_thresh):
        """Detect parallel line by scanning vertically from horizontal edge samples.

        Uses luminance difference from base (edge position) to detect lines.
        Scans all sample points upfront, then slides a window looking for
        sufficient agreement.  Allows up to 2 misses per window to tolerate
        noise, text crossing the line, etc.

        Returns (near_edge, center) tuple or None.
        near_edge: for snap calculation (margin measured from line edge).
        center: for guide line display (visual center of dark band).
        """
        n = len(xs)
        w = min(consec_window, n)
        tol = dist_tol
        thresh = lum_thresh
        # Each hit stores (found, near_edge, center)
        hits: list[tuple[bool, float, float]] = []

        for idx in range(n):
            x = xs[idx]
            col = int(round(x))
            if col < 0 or col >= img_w:
                hits.append((False, -1.0, -1.0))
                continue
            base_lum = int(grayscale[iy, col])
            found = False
            for d in range(1, max_scan + 1):
                sy = iy + scan_dir * d
                if sy < 0 or sy >= img_h:
                    break
                if abs(int(grayscale[sy, col]) - base_lum) > thresh:
                    # Sub-pixel near edge (closest to rectangle edge)
                    y_prev = iy + scan_dir * (d - 1)
                    v_prev = abs(float(grayscale[y_prev, col]) - base_lum)
                    v_hit = abs(float(grayscale[sy, col]) - base_lum)
                    dv = v_hit - v_prev
                    if dv != 0:
                        refined_near = y_prev + scan_dir * (thresh - v_prev) / dv
                    else:
                        refined_near = float(sy)
                    # Find band end for center calculation
                    sy_end = sy
                    for d2 in range(d + 1, max_scan + 1):
                        sy2 = iy + scan_dir * d2
                        if sy2 < 0 or sy2 >= img_h:
                            break
                        if abs(int(grayscale[sy2, col]) - base_lum) > thresh:
                            sy_end = sy2
                        else:
                            break
                    # Sub-pixel far edge
                    y_after = sy_end + scan_dir
                    if 0 <= y_after < img_h and abs(int(grayscale[y_after, col]) - base_lum) <= thresh:
                        v_last = abs(float(grayscale[sy_end, col]) - base_lum)
                        v_after = abs(float(grayscale[y_after, col]) - base_lum)
                        dv2 = v_last - v_after
                        if dv2 != 0:
                            refined_far = sy_end + scan_dir * (v_last - thresh) / dv2
                        else:
                            refined_far = float(sy_end)
                    else:
                        refined_far = float(sy_end)
                    hits.append((True, refined_near, (refined_near + refined_far) / 2.0))
                    found = True
                    break
            if not found:
                hits.append((False, -1.0, -1.0))

        # Slide window: cascade from strict to relaxed tolerance
        min_hits = max(w - 1, (w + 1) // 2)
        for cur_tol in (tol * 0.5, tol):
            for start in range(max(n - w + 1, 1)):
                end = min(start + w, n)
                window = [(hits[i][1], hits[i][2]) for i in range(start, end) if hits[i][0]]
                if len(window) < min_hits:
                    continue
                near_positions = [ne for ne, _ in window]
                median_near = float(np.median(near_positions))
                agree_idx = [j for j, ne in enumerate(near_positions) if abs(ne - median_near) <= cur_tol]
                if len(agree_idx) >= min_hits:
                    near_edge = float(np.median([near_positions[j] for j in agree_idx]))
                    center = float(np.median([window[j][1] for j in agree_idx]))
                    return (near_edge, center)

        return None

    def _detect_line_v(self, ys, ix, scan_dir, max_scan, grayscale, img_h, img_w,
                       consec_window, dist_tol, lum_thresh):
        """Detect parallel line by scanning horizontally from vertical edge samples.

        Uses luminance difference from base (edge position) to detect lines.
        Same tolerance logic as _detect_line_h (allows up to 2 misses per window).

        Returns (near_edge, center) tuple or None.
        """
        n = len(ys)
        w = min(consec_window, n)
        tol = dist_tol
        thresh = lum_thresh
        hits: list[tuple[bool, float, float]] = []

        for idx in range(n):
            y = ys[idx]
            row = int(round(y))
            if row < 0 or row >= img_h:
                hits.append((False, -1.0, -1.0))
                continue
            base_lum = int(grayscale[row, ix])
            found = False
            for d in range(1, max_scan + 1):
                sx = ix + scan_dir * d
                if sx < 0 or sx >= img_w:
                    break
                if abs(int(grayscale[row, sx]) - base_lum) > thresh:
                    # Sub-pixel near edge
                    x_prev = ix + scan_dir * (d - 1)
                    v_prev = abs(float(grayscale[row, x_prev]) - base_lum)
                    v_hit = abs(float(grayscale[row, sx]) - base_lum)
                    dv = v_hit - v_prev
                    if dv != 0:
                        refined_near = x_prev + scan_dir * (thresh - v_prev) / dv
                    else:
                        refined_near = float(sx)
                    # Find band end for center
                    sx_end = sx
                    for d2 in range(d + 1, max_scan + 1):
                        sx2 = ix + scan_dir * d2
                        if sx2 < 0 or sx2 >= img_w:
                            break
                        if abs(int(grayscale[row, sx2]) - base_lum) > thresh:
                            sx_end = sx2
                        else:
                            break
                    # Sub-pixel far edge
                    x_after = sx_end + scan_dir
                    if 0 <= x_after < img_w and abs(int(grayscale[row, x_after]) - base_lum) <= thresh:
                        v_last = abs(float(grayscale[row, sx_end]) - base_lum)
                        v_after = abs(float(grayscale[row, x_after]) - base_lum)
                        dv2 = v_last - v_after
                        if dv2 != 0:
                            refined_far = sx_end + scan_dir * (v_last - thresh) / dv2
                        else:
                            refined_far = float(sx_end)
                    else:
                        refined_far = float(sx_end)
                    hits.append((True, refined_near, (refined_near + refined_far) / 2.0))
                    found = True
                    break
            if not found:
                hits.append((False, -1.0, -1.0))

        # Slide window: cascade from strict to relaxed tolerance
        min_hits = max(w - 1, (w + 1) // 2)
        for cur_tol in (tol * 0.5, tol):
            for start in range(max(n - w + 1, 1)):
                end = min(start + w, n)
                window = [(hits[i][1], hits[i][2]) for i in range(start, end) if hits[i][0]]
                if len(window) < min_hits:
                    continue
                near_positions = [ne for ne, _ in window]
                median_near = float(np.median(near_positions))
                agree_idx = [j for j, ne in enumerate(near_positions) if abs(ne - median_near) <= cur_tol]
                if len(agree_idx) >= min_hits:
                    near_edge = float(np.median([near_positions[j] for j in agree_idx]))
                    center = float(np.median([window[j][1] for j in agree_idx]))
                    return (near_edge, center)

        return None

    # -- Text bounding snap detection --

    def _detect_text_bounding_snap(
        self,
        rule: dict,
        rule_index: int,
        shape,
        edge_index: int,
        cursor_pos: QPointF,
    ) -> tuple[QPointF, list[tuple[float, float]]] | None:
        """Detect luminance change boundary and snap edge.

        Scans every pixel along the edge (full coordinate scan).
        Caches boundary detection for stability during drag.
        Uses extreme value (min/max) to catch thin protrusions.
        Returns (snapped_pos, detected_dots) or None.
        """
        from labelme.shape import Shape

        d = self._TB_DEFAULTS
        snap_range_px = rule.get("snap_range_pixels", d["snap_range_pixels"])
        resize_base = rule.get("resize_base", d["resize_base"])
        margin_px = rule.get("margin_pixels", d["margin_pixels"])

        # Determine scan direction
        if edge_index in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM):
            scan_dir = -1 if edge_index == Shape.EDGE_BOTTOM else 1
            is_horiz = True
        elif edge_index in (Shape.EDGE_LEFT, Shape.EDGE_RIGHT):
            scan_dir = 1 if edge_index == Shape.EDGE_LEFT else -1
            is_horiz = False
        else:
            return None

        # Compute margin and snap zone from resize_base scale
        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        scale = max(img_w, img_h) / resize_base
        M = margin_px * scale
        snap_window = snap_range_px * scale

        cursor_val = cursor_pos.y() if is_horiz else cursor_pos.x()

        # Cache logic:
        #   - Within snap zone → keep cached boundary (locked)
        #   - Outside snap zone → rescan from current cursor position
        cache = self._tb_boundary_cache
        if (
            cache is not None
            and cache[0] == edge_index
            and cache[1] == rule_index
        ):
            cached_boundary = cache[2]
            cached_snap = (cached_boundary + 0.5) - scan_dir * M
            if abs(cursor_val - cached_snap) <= snap_window:
                # Still within snap zone — stay locked
                self._tb_snap_entered = False
                if is_horiz:
                    return (QPointF(cursor_pos.x(), cached_snap), cache[3])
                else:
                    return (QPointF(cached_snap, cursor_pos.y()), cache[3])
            else:
                # Left snap zone — clear cache, rescan below
                self._tb_boundary_cache = None
                self._tb_snap_entered = False

        # Live scan from current cursor position
        result = self._scan_text_boundary(
            rule, shape, edge_index, scan_dir, is_horiz, cursor_pos,
        )
        if result[0] is None:
            self._tb_snap_entered = False
            return None
        boundary_pos, dots = result

        boundary_img = boundary_pos + 0.5
        snap_offset = boundary_img - scan_dir * M
        dist = abs(cursor_val - snap_offset)
        if dist <= snap_window:
            # Enter snap zone — cache, lock, and signal cursor warp
            self._tb_boundary_cache = (
                edge_index, rule_index, boundary_pos, dots,
            )
            self._tb_snap_entered = True  # first frame of snap
            if is_horiz:
                return (QPointF(cursor_pos.x(), snap_offset), dots)
            else:
                return (QPointF(snap_offset, cursor_pos.y()), dots)
        self._tb_snap_entered = False
        return None

    def _scan_text_boundary(
        self,
        rule: dict,
        shape,
        edge_index: int,
        scan_dir: int,
        is_horiz: bool,
        cursor_pos: QPointF,
    ) -> tuple[float | None, list[tuple[float, float]]]:
        """Scan all pixels along edge to find text boundary.

        Scan origin is offset outward from cursor position by scan_offset pixels,
        ensuring base_lum is sampled from background area.
        Uses extreme value (min/max) to catch protrusions.
        Returns (boundary_pos, dots) or (None, []).
        """
        from labelme.shape import Shape

        d = self._TB_DEFAULTS
        lum_thresh = rule.get("luminance_threshold", d["luminance_threshold"])
        min_agreement = rule.get("min_agreement", d["min_agreement"])

        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape

        p0, p1 = shape.points[0], shape.points[1]
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())

        # Dynamic offset: 1/10 of the short side (constant for all edges)
        short_side = min(right - left, bottom - top)
        scan_offset = short_side / 10.0
        max_scan = int(max(right - left, bottom - top) + scan_offset)

        if is_horiz:
            cursor_val = cursor_pos.y()
            iy = int(round(cursor_val - scan_dir * scan_offset))
            xs = np.arange(int(np.ceil(left)), int(np.floor(right)) + 1)
            if len(xs) == 0:
                return None, []
            iy = max(0, min(iy, img_h - 1))
            dots = self._detect_boundary_h(
                xs, iy, scan_dir, max_scan, grayscale, img_h, img_w, lum_thresh,
            )
            if len(dots) < min_agreement:
                return None, []
            values = [dy for _, dy in dots]
            val_fn = lambda dx, dy: dy
        else:
            cursor_val = cursor_pos.x()
            ix = int(round(cursor_val - scan_dir * scan_offset))
            ys = np.arange(int(np.ceil(top)), int(np.floor(bottom)) + 1)
            if len(ys) == 0:
                return None, []
            ix = max(0, min(ix, img_w - 1))
            dots = self._detect_boundary_v(
                ys, ix, scan_dir, max_scan, grayscale, img_h, img_w, lum_thresh,
            )
            if len(dots) < min_agreement:
                return None, []
            values = [dx for dx, _ in dots]
            val_fn = lambda dx, dy: dx

        # Boundary = extreme of all dots (catches protrusions)
        all_vals = [val_fn(dx, dy) for dx, dy in dots]
        if scan_dir > 0:
            boundary_pos = float(min(all_vals))
        else:
            boundary_pos = float(max(all_vals))

        return boundary_pos, dots

    def _detect_boundary_h(self, xs, iy, scan_dir, max_scan,
                           grayscale, img_h, img_w, lum_threshold):
        """Scan vertically from horizontal edge, detecting luminance CHANGE.

        Vectorized numpy implementation: extracts a 2D slice of all scan
        columns at once, computes diffs, finds threshold crossings, and
        interpolates for sub-pixel precision.

        Returns list of (x_img, y_img) boundary points.
        """
        if iy < 0 or iy >= img_h:
            return []
        cols = np.array([int(round(x)) for x in xs], dtype=int)
        mask = (cols >= 0) & (cols < img_w)
        cols = cols[mask]
        if len(cols) == 0:
            return []

        # Base luminance: max in ±2px window around iy (5 pixels)
        iy_lo = max(0, iy - 2)
        iy_hi = min(img_h, iy + 3)
        base_lums = np.max(grayscale[iy_lo:iy_hi, cols].astype(float), axis=0)

        if scan_dir > 0:
            scan_ys = np.arange(iy + 1, min(iy + max_scan + 1, img_h))
        else:
            scan_ys = np.arange(iy - 1, max(iy - max_scan - 1, -1), -1)
        if len(scan_ys) == 0:
            return []

        # 2D slice: (num_scan_rows, num_cols)
        vals = grayscale[scan_ys[:, np.newaxis], cols[np.newaxis, :]].astype(float)
        diffs = np.abs(vals - base_lums[np.newaxis, :])

        crossed = diffs > lum_threshold
        first_idx = np.argmax(crossed, axis=0)
        has_hit = crossed[first_idx, np.arange(len(cols))]

        valid = np.where(has_hit)[0]
        if len(valid) == 0:
            return []

        idx = first_idx[valid]
        d_curr = diffs[idx, valid]
        prev_idx = np.maximum(idx - 1, 0)
        d_prev = np.where(idx > 0, diffs[prev_idx, valid], 0.0)
        denom = d_curr - d_prev
        frac = np.where(denom > 0, (lum_threshold - d_prev) / denom, 0.0)
        prev_y = np.where(idx > 0, scan_ys[prev_idx].astype(float), float(iy))
        refined_y = prev_y + scan_dir * frac

        return list(zip(cols[valid].astype(float).tolist(), refined_y.tolist()))

    def _detect_boundary_v(self, ys, ix, scan_dir, max_scan,
                           grayscale, img_h, img_w, lum_threshold):
        """Scan horizontally from vertical edge, detecting luminance CHANGE.

        Vectorized numpy implementation with sub-pixel interpolation.

        Returns list of (x_img, y_img) boundary points.
        """
        if ix < 0 or ix >= img_w:
            return []
        rows = np.array([int(round(y)) for y in ys], dtype=int)
        mask = (rows >= 0) & (rows < img_h)
        rows = rows[mask]
        if len(rows) == 0:
            return []

        # Base luminance: max in ±2px window around ix (5 pixels)
        ix_lo = max(0, ix - 2)
        ix_hi = min(img_w, ix + 3)
        base_lums = np.max(grayscale[rows[:, np.newaxis], np.arange(ix_lo, ix_hi)[np.newaxis, :]].astype(float), axis=1)

        if scan_dir > 0:
            scan_xs = np.arange(ix + 1, min(ix + max_scan + 1, img_w))
        else:
            scan_xs = np.arange(ix - 1, max(ix - max_scan - 1, -1), -1)
        if len(scan_xs) == 0:
            return []

        # 2D slice: (num_rows, num_scan_cols)
        vals = grayscale[rows[:, np.newaxis], scan_xs[np.newaxis, :]].astype(float)
        diffs = np.abs(vals - base_lums[:, np.newaxis])

        crossed = diffs > lum_threshold
        first_idx = np.argmax(crossed, axis=1)
        has_hit = crossed[np.arange(len(rows)), first_idx]

        valid = np.where(has_hit)[0]
        if len(valid) == 0:
            return []

        idx = first_idx[valid]
        d_curr = diffs[valid, idx]
        prev_idx = np.maximum(idx - 1, 0)
        d_prev = np.where(idx > 0, diffs[valid, prev_idx], 0.0)
        denom = d_curr - d_prev
        frac = np.where(denom > 0, (lum_threshold - d_prev) / denom, 0.0)
        prev_x = np.where(idx > 0, scan_xs[prev_idx].astype(float), float(ix))
        refined_x = prev_x + scan_dir * frac

        return list(zip(refined_x.tolist(), rows[valid].astype(float).tolist()))

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            for shape in self.selectedShapes:
                shape.highlightClear()
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.sortShapesByArea()
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.sortShapesByArea()
        self.storeShapes()
        self.update()

    def resizeEvent(self, event):
        """Handle resize events - resize cursor overlay."""
        super().resizeEvent(event)
        self._cursor_overlay.setGeometry(self.rect())

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        if not self.pixmap:
            return super().paintEvent(a0)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        p.scale(1 / self.scale, 1 / self.scale)

        # draw crosshair (not for point/polygon - they have custom crosshair only)
        if (
            self._crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint is not None
            and not self.outOfPixmap(self.prevMovePoint)
            and self.createMode not in ["point", "polygon"]
        ):
            p.setPen(QtGui.QColor(0, 0, 0, 128))
            p.drawLine(
                0,
                int(self.prevMovePoint.y() * self.scale),
                self.width() - 1,
                int(self.prevMovePoint.y() * self.scale),
            )
            p.drawLine(
                int(self.prevMovePoint.x() * self.scale),
                0,
                int(self.prevMovePoint.x() * self.scale),
                self.height() - 1,
            )

        Shape.scale = self.scale
        selected_shapes = []
        for shape in self._shapes_paint_order:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                if shape.selected:
                    selected_shapes.append(shape)
                else:
                    shape.paint(p)
        # Draw selected shapes last so they appear on top
        for shape in selected_shapes:
            shape.paint(p)
        if self.current:
            self.current.paint(p)
            # Don't paint preview line when near start point (hide cursor square)
            if not self._near_start_point:
                assert len(self.line.points) == len(self.line.point_labels)
                self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        # Draw grid line when dragging edge midpoint
        if self._edge_midpoint_dragging and self.hShape is not None and len(self.hShape.points) == 2:
            # Get shape's line color with alpha 128
            base_color = QtGui.QColor(self.hShape.line_color)
            line_color = QtGui.QColor(base_color)
            line_color.setAlpha(128)
            pen = QtGui.QPen(line_color)
            pen.setWidth(1)
            p.setPen(pen)

            # Get actual edge position from rectangle points
            p0, p1 = self.hShape.points[0], self.hShape.points[1]

            # EDGE_TOP=0, EDGE_BOTTOM=1 -> horizontal line at edge y
            # EDGE_LEFT=2, EDGE_RIGHT=3 -> vertical line at edge x
            # Note: painter still has offset transform, so just multiply by scale
            if self._dragging_edge_index == Shape.EDGE_TOP:
                edge_y = min(p0.y(), p1.y())
                cy = edge_y * self.scale
                p.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
            elif self._dragging_edge_index == Shape.EDGE_BOTTOM:
                edge_y = max(p0.y(), p1.y())
                cy = edge_y * self.scale
                p.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
            elif self._dragging_edge_index == Shape.EDGE_LEFT:
                edge_x = min(p0.x(), p1.x())
                cx = edge_x * self.scale
                p.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))
            elif self._dragging_edge_index == Shape.EDGE_RIGHT:
                edge_x = max(p0.x(), p1.x())
                cx = edge_x * self.scale
                p.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))

        # Draw red snap guide line when parallel line snap is active
        if self._snap_active and self._snap_line_pos is not None and self._edge_midpoint_dragging:
            snap_pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 80))
            snap_pen.setWidth(4)
            snap_pen.setStyle(Qt.DashLine)
            p.setPen(snap_pen)
            if self._dragging_edge_index in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM):
                # Horizontal line at detected parallel line center (already in image space)
                ly = self._snap_line_pos * self.scale
                p.drawLine(QPointF(0, ly), QPointF(self.width(), ly))
            elif self._dragging_edge_index in (Shape.EDGE_LEFT, Shape.EDGE_RIGHT):
                # Vertical line at detected parallel line center (already in image space)
                lx = self._snap_line_pos * self.scale
                p.drawLine(QPointF(lx, 0), QPointF(lx, self.height()))

        # Draw red semi-transparent dots when text bounding snap is active
        if self._text_bounding_snap_dots and self._edge_midpoint_dragging:
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))
            for dx, dy in self._text_bounding_snap_dots:
                sx = (dx + 0.5) * self.scale
                sy = (dy + 0.5) * self.scale
                p.drawEllipse(QPointF(sx, sy), 3.0, 3.0)

        # Draw ghost cursor for dark pixel magnet (real mouse position)
        if self._dpm_ghost_pos is not None:
            ghost_color = QtGui.QColor(128, 128, 128, 51)  # gray, 80% transparent
            ghost_pen = QtGui.QPen(ghost_color)
            ghost_pen.setWidth(1)
            p.setPen(ghost_pen)
            gx = self._dpm_ghost_pos.x() * self.scale
            gy = self._dpm_ghost_pos.y() * self.scale
            arm = 12
            p.drawLine(QPointF(gx - arm, gy), QPointF(gx + arm, gy))
            p.drawLine(QPointF(gx, gy - arm), QPointF(gx, gy + arm))

        # Draw hover label next to cursor
        # Hidden during: mouse button pressed (including vertex/edge dragging)
        if (
            self._hover_label_ready
            and self.hShape is not None
            and self.hShape.label
            and self.prevMovePoint is not None
            and not self._mouse_pressed
        ):
            self._drawHoverLabel(p, self.hShape)

        if not self.current or self.createMode not in [
            "polygon",
            "ai_polygon",
            "ai_mask",
        ]:
            p.end()
            return

        drawing_shape: Shape = self.current.copy()
        if self.createMode == "polygon":
            # Add preview point when near start
            if self._near_start_point and len(self.current.points) >= 2:
                drawing_shape.addPoint(self.line[1])
            # Only show fill when near start point (preview)
            drawing_shape.fill = self._near_start_point
            drawing_shape.selected = False  # Use fill_color, not select_fill_color
            drawing_shape._is_creating = True  # Flag for drawing start vertex in white
            # Apply current fill opacity from Shape class
            if self._near_start_point:
                # Use line color with current fill opacity for preview
                r, g, b, _ = drawing_shape.line_color.getRgb()
                fill_alpha = Shape.fill_color.alpha()
                drawing_shape.fill_color = QtGui.QColor(r, g, b, fill_alpha)
        elif self.createMode in ["ai_polygon", "ai_mask"]:
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            self._update_shape_with_ai(
                points=drawing_shape.points,
                point_labels=drawing_shape.point_labels,
                shape=drawing_shape,
            )
            drawing_shape.fill = self.fillDrawing()
            drawing_shape.selected = self.fillDrawing()
        else:
            drawing_shape.fill = self.fillDrawing()
            drawing_shape.selected = self.fillDrawing()
        drawing_shape.paint(p)
        p.end()

        # Ensure cursor overlay stays on top after canvas repaint
        self._cursor_overlay.raise_()

    def _onHoverLabelTimeout(self):
        """Called after hover delay; mark label as ready and repaint."""
        self._hover_label_ready = True
        self.update()

    def _drawHoverLabel(self, painter, shape):
        """Draw the shape's label next to the cursor (zoom-independent size)."""
        painter.save()

        # Cursor position in scaled coordinates
        cx = self.prevMovePoint.x() * self.scale
        cy = self.prevMovePoint.y() * self.scale
        offset_x = 20
        offset_y = -40

        # Fixed font size (zoom-independent)
        font = painter.font()
        font.setPointSize(20)
        painter.setFont(font)

        fm = QtGui.QFontMetrics(font)
        text_rect = fm.boundingRect(shape.label)
        padding = 8

        bg_rect = QtCore.QRectF(
            cx + offset_x,
            cy + offset_y - text_rect.height(),
            text_rect.width() + padding * 2,
            text_rect.height() + padding * 2,
        )

        r, g, b, _ = shape.line_color.getRgb()
        fill_alpha = Shape.fill_color.alpha()
        # 0.75x transparency = more opaque, capped at 255
        label_alpha = min(int(255 - (255 - fill_alpha) * 0.75), 255)
        painter.setBrush(QtGui.QColor(r, g, b, label_alpha))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(bg_rect, 3, 3)

        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(bg_rect, QtCore.Qt.AlignCenter, shape.label)

        painter.restore()

    def transformPos(self, point: QPointF) -> QPointF:
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def _warp_cursor_to_image_pos(self, image_pos: QPointF) -> None:
        """Move physical cursor to the given image coordinate position."""
        offset = self.offsetToCenter()
        widget_pos = QPointF(
            (image_pos.x() + offset.x()) * self.scale,
            (image_pos.y() + offset.y()) * self.scale,
        )
        global_pos = self.mapToGlobal(widget_pos.toPoint())
        QtGui.QCursor.setPos(global_pos)

    def enableDragging(self, enabled: bool):
        self._is_dragging_enabled = enabled

    def offsetToCenter(self) -> QPointF:
        s = self.scale
        area = super().size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def outOfPixmap(self, p: QPointF) -> bool:
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        if self.createMode in ["ai_polygon", "ai_mask"]:
            self._update_shape_with_ai(
                points=self.current.points,
                point_labels=self.current.point_labels,
                shape=self.current,
            )
        self.current.close()
        self.current._is_creating = False  # No longer creating

        self.shapes.append(self.current)
        self.sortShapesByArea()
        self.storeShapes()
        self.current = None
        self._near_start_point = False  # Reset for next shape
        self.setHiding(False)
        self.drawingPolygon.emit(False)  # Reset drawing state
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1: QPointF, p2: QPointF) -> QPointF:
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if not self.pixmap:
            return super().minimumSizeHint()

        min_size = self.scale * self.pixmap.size()
        if self._is_dragging_enabled:
            # When drag buffer should be enabled, add a bit of buffer around the image
            # This lets dragging the image around have a bit of give on the edges
            min_size = 1.167 * min_size
        return min_size

    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        import time

        mods: Qt.KeyboardModifiers = a0.modifiers()
        delta: QPoint = a0.angleDelta()
        if Qt.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoomRequest.emit(delta.y(), a0.posF())
        else:
            # scroll with debounce (45ms interval)
            current_time = time.time()
            if current_time - self._last_scroll_time < 0.045:
                a0.accept()
                return
            self._last_scroll_time = current_time
            self.scrollRequest.emit(delta.x(), Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), Qt.Vertical)
        a0.accept()

    def event(self, a0: QtCore.QEvent) -> bool:
        """Handle gesture events for pinch-to-zoom on Mac trackpad."""
        if a0.type() == QtCore.QEvent.Gesture:
            return self._gestureEvent(a0)
        return super().event(a0)

    def _gestureEvent(self, a0: QtGui.QGestureEvent) -> bool:
        """Process pinch gesture for zooming."""
        pinch: QtWidgets.QPinchGesture = a0.gesture(Qt.PinchGesture)
        if pinch:
            if pinch.state() == Qt.GestureUpdated:
                # scaleFactor is relative to previous state (>1 = zoom in, <1 = zoom out)
                scale_factor = pinch.scaleFactor()
                center = pinch.centerPoint()
                self.pinchZoomRequest.emit(scale_factor, center)
            return True
        return False

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        modifiers = a0.modifiers()
        key = a0.key()
        if self.drawing():
            if key == Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                # Reset near-start-point flag
                self._near_start_point = False
                # Restore cursor when canceling creation
                self._unhide_os_cursor()
                self.restoreCursor()
                self.update()
            elif (
                key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Space)
                and self.canCloseShape()
            ):
                self.finalise()
            elif modifiers == Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == Qt.Key_Up:
                self.moveByKeyboard(QPointF(0.0, -MOVE_SPEED))
            elif key == Qt.Key_Down:
                self.moveByKeyboard(QPointF(0.0, MOVE_SPEED))
            elif key == Qt.Key_Left:
                self.moveByKeyboard(QPointF(-MOVE_SPEED, 0.0))
            elif key == Qt.Key_Right:
                self.moveByKeyboard(QPointF(MOVE_SPEED, 0.0))
        self._update_status()

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        modifiers = a0.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        point = self.current.popPoint()
        self._undone_points.append(point)
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def redoLastPoint(self):
        if not self._undone_points:
            return
        if not self.current:
            return
        point = self._undone_points.pop()
        self.current.addPoint(point)
        self.line[0] = self.current[-1]
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        img_arr = labelme.utils.img_qt_to_arr(img_qt=self.pixmap.toImage())
        self._pixmap_hash = hash(img_arr.tobytes())
        # Cache image array (BGRA) for pixel info lookup
        self._img_arr_cache = img_arr.copy() if img_arr.ndim >= 2 else None
        # Build grayscale cache (OpenCV BT.601: Y = 0.299R + 0.587G + 0.114B)
        if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
            # Qt ARGB32 little-endian stores as BGRA
            self._grayscale_cache = np.dot(
                img_arr[:, :, :3].astype(np.float32),
                [0.114, 0.587, 0.299],
            ).astype(np.uint8)
        elif img_arr.ndim == 2:
            self._grayscale_cache = img_arr.astype(np.uint8)
        else:
            self._grayscale_cache = None
        if clear_shapes:
            self.shapes = []
            self._shapes_paint_order = []
            self._shapes_hover_order = []
        # Reset prevMovePoint to avoid out-of-bounds cursor position from previous image
        self.prevMovePoint = None
        self._cursor_overlay.hideCursor()
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.sortShapesByArea()
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def sortShapesByArea(self):
        """Rebuild cached sort orders for painting and hover detection."""
        # Paint order: largest first, points on top (drawn last)
        self._shapes_paint_order = sorted(
            self.shapes,
            key=lambda s: (
                1 if s.shape_type == "point" else 0,
                -(s.boundingRect().width() * s.boundingRect().height()),
            ),
        )
        # Hover order: selected first, then smallest first, points first
        self._shapes_hover_order = sorted(
            self.shapes,
            key=lambda s: (
                0 if s.selected else 1,
                0 if s.shape_type == "point" else 1,
                s.boundingRect().width() * s.boundingRect().height(),
            ),
        )

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        if self._cursor_debug:
            self._log_cursor_state(f"overrideCursor:request={cursor}")
        if self._vertex_dragging and cursor != self._blank_cursor:
            return
        if cursor == self._cursor:
            return
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)
        # Also set widget cursor to match (in case _force_blank_cursor set it)
        self.setCursor(cursor)
        if self._cursor_debug:
            self._log_cursor_state("overrideCursor:applied")

    def restoreCursor(self):
        if self._cursor_debug:
            self._log_cursor_state("restoreCursor:request")
        if self._vertex_dragging:
            return
        self._unhide_os_cursor()
        self._cursor = CURSOR_DEFAULT
        QtWidgets.QApplication.restoreOverrideCursor()
        self._clear_parent_viewport_cursor()
        # Also unset canvas widget cursor (set by _force_blank_cursor)
        self.unsetCursor()
        if self._cursor_debug:
            self._log_cursor_state("restoreCursor:applied")

    def _force_blank_cursor(self) -> None:
        """Force blank cursor immediately (click-time) for drag/creation."""
        if not self._custom_cursor_enabled:
            return
        if self._cursor_debug:
            self._log_cursor_state("force_blank:before")
        if not self._os_cursor_hidden:
            if _QUARTZ_AVAILABLE:
                try:
                    Quartz.CGDisplayHideCursor(Quartz.CGMainDisplayID())
                except Exception:
                    pass
            if _CG_AVAILABLE and _CG is not None:
                try:
                    _CG.CGDisplayHideCursor(_CG.CGMainDisplayID())
                except Exception:
                    pass
            if _NSCURSOR_AVAILABLE and not self._ns_cursor_hidden:
                NSCursor.hide()
                self._ns_cursor_hidden = True
            self._os_cursor_hidden = True
        if QtWidgets.QApplication.overrideCursor() is None:
            QtWidgets.QApplication.setOverrideCursor(self._blank_cursor)
        else:
            QtWidgets.QApplication.changeOverrideCursor(self._blank_cursor)
        # Also set widget cursor to avoid visual lag on some platforms
        self.setCursor(self._blank_cursor)
        self._set_parent_viewport_cursor(self._blank_cursor)
        # Nudge cursor to force platform redraw
        QtGui.QCursor.setPos(QtGui.QCursor.pos())
        self._cursor = self._blank_cursor
        if self._cursor_debug:
            self._log_cursor_state("force_blank:after")

    def _force_point_cursor(self) -> None:
        """Force pointing-hand cursor immediately."""
        self._unhide_os_cursor()
        cursor = QtGui.QCursor(CURSOR_POINT)
        if QtWidgets.QApplication.overrideCursor() is None:
            QtWidgets.QApplication.setOverrideCursor(cursor)
        else:
            QtWidgets.QApplication.changeOverrideCursor(cursor)
        self.setCursor(cursor)
        self._set_parent_viewport_cursor(cursor)
        self._cursor = CURSOR_POINT

    def _unhide_os_cursor(self) -> None:
        if self._os_cursor_hidden:
            if _QUARTZ_AVAILABLE:
                try:
                    Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
                except Exception:
                    pass
            if _CG_AVAILABLE and _CG is not None:
                try:
                    _CG.CGDisplayShowCursor(_CG.CGMainDisplayID())
                except Exception:
                    pass
            if _NSCURSOR_AVAILABLE and self._ns_cursor_hidden:
                NSCursor.unhide()
                self._ns_cursor_hidden = False
            self._os_cursor_hidden = False

    def _set_parent_viewport_cursor(self, cursor: QtGui.QCursor) -> None:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractScrollArea):
                parent.viewport().setCursor(cursor)
                break
            parent = parent.parent()

    def _clear_parent_viewport_cursor(self) -> None:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractScrollArea):
                parent.viewport().unsetCursor()
                break
            parent = parent.parent()

    def _log_cursor_state(self, tag: str) -> None:
        try:
            override = QtWidgets.QApplication.overrideCursor()
            override_shape = override.shape() if override is not None else None
        except Exception:
            override_shape = None
        cursor_shape = getattr(self, "_cursor", None)
        logger.info(
            "[cursor] {}: _cursor={!r} override={!r} dragging={!r}",
            tag,
            cursor_shape,
            override_shape,
            self._vertex_dragging,
        )

    def resetState(self):
        self.restoreCursor()
        self.pixmap = QtGui.QPixmap()
        self._pixmap_hash = None
        self._img_arr_cache: np.ndarray | None = None
        self._grayscale_cache: np.ndarray | None = None
        self.shapes = []
        self._shapes_paint_order = []
        self._shapes_hover_order = []
        self.shapesBackups = []
        self.shapesRedoStack = []
        self._undone_points = []
        self.movingShape = False
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.current = None
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.hEdgeMidpoint = None  # For rectangle edge midpoint hovering
        self.prevhEdgeMidpoint = None
        self._edge_midpoint_drag_shape = None
        self._text_bounding_snap_dots = None
        self._tb_boundary_cache = None
        self._pl_snap_cache = None
        self._lf_snap_cache = None
        self._pending_draw_label = None
        self._dpm_ghost_pos = None
        self._dpm_coarse_cache = None
        self.update()


def _update_shape_with_ai_response(
    response: osam.types.GenerateResponse,
    shape: Shape,
    createMode: Literal["ai_polygon", "ai_mask"],
) -> None:
    if createMode not in ["ai_polygon", "ai_mask"]:
        raise ValueError(
            f"createMode must be 'ai_polygon' or 'ai_mask', not {createMode}"
        )

    if not response.annotations:
        logger.warning("No annotations returned")
        return

    if createMode == "ai_mask":
        y1: int
        x1: int
        y2: int
        x2: int
        if response.annotations[0].bounding_box is None:
            y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
                [response.annotations[0].mask]
            )[0].astype(int)
        else:
            y1 = response.annotations[0].bounding_box.ymin
            x1 = response.annotations[0].bounding_box.xmin
            y2 = response.annotations[0].bounding_box.ymax
            x2 = response.annotations[0].bounding_box.xmax
        shape.setShapeRefined(
            shape_type="mask",
            points=[QPointF(x1, y1), QPointF(x2, y2)],
            point_labels=[1, 1],
            mask=response.annotations[0].mask[y1 : y2 + 1, x1 : x2 + 1],
        )
    elif createMode == "ai_polygon":
        points = polygon_from_mask.compute_polygon_from_mask(
            mask=response.annotations[0].mask
        )
        if len(points) < 2:
            return
        shape.setShapeRefined(
            shape_type="polygon",
            points=[QPointF(point[0], point[1]) for point in points],
            point_labels=[1] * len(points),
        )


def _snap_cursor_pos_for_square(pos: QPointF, opposite_vertex: QPointF) -> QPointF:
    pos_from_opposite: QPointF = pos - opposite_vertex
    square_size: float = min(abs(pos_from_opposite.x()), abs(pos_from_opposite.y()))
    return opposite_vertex + QPointF(
        np.sign(pos_from_opposite.x()) * square_size,
        np.sign(pos_from_opposite.y()) * square_size,
    )
