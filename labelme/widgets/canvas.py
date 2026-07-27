from __future__ import annotations

import enum
import math
import os
import ctypes
import ctypes.util
from typing import Literal

import cv2
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


class _MagicWandPanel(QtWidgets.QDialog):
    """Dialog for magic wand parameter adjustment."""

    paramsChanged = QtCore.pyqtSignal()
    rangeChanged = QtCore.pyqtSignal(int, int)
    accepted_signal = QtCore.pyqtSignal()
    rejected_signal = QtCore.pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self.setWindowTitle(self.tr("Magic Wand"))
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowContextHelpButtonHint
            | Qt.WindowStaysOnTopHint
        )

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(4)

        # --- Top area: grid + RGB±5 buttons ---
        top_layout = QtWidgets.QHBoxLayout()

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(2)

        self._r = self._make_spinbox(0, 255, 1)
        self._g = self._make_spinbox(0, 255, 1)
        self._b = self._make_spinbox(0, 255, 1)
        self._tol_r = self._make_spinbox(1, 255, 5)
        self._tol_g = self._make_spinbox(1, 255, 5)
        self._tol_b = self._make_spinbox(1, 255, 5)
        self._range_w = self._make_spinbox(1, 99999, 10)
        self._range_h = self._make_spinbox(1, 99999, 10)

        grid.addWidget(QtWidgets.QLabel("R"), 0, 0)
        grid.addWidget(self._r, 0, 1)
        grid.addWidget(QtWidgets.QLabel("±"), 0, 2)
        grid.addWidget(self._tol_r, 0, 3)

        grid.addWidget(QtWidgets.QLabel("G"), 1, 0)
        grid.addWidget(self._g, 1, 1)
        grid.addWidget(QtWidgets.QLabel("±"), 1, 2)
        grid.addWidget(self._tol_g, 1, 3)

        grid.addWidget(QtWidgets.QLabel("B"), 2, 0)
        grid.addWidget(self._b, 2, 1)
        grid.addWidget(QtWidgets.QLabel("±"), 2, 2)
        grid.addWidget(self._tol_b, 2, 3)

        grid.addWidget(QtWidgets.QLabel("W"), 3, 0)
        grid.addWidget(self._range_w, 3, 1)
        grid.addWidget(QtWidgets.QLabel("H"), 3, 2)
        grid.addWidget(self._range_h, 3, 3)

        top_layout.addLayout(grid)

        # RGB±5 buttons on the right
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.addStretch()
        self._btn_rgb_up = QtWidgets.QPushButton("RGB+5")
        self._btn_rgb_down = QtWidgets.QPushButton("RGB-5")
        self._btn_rgb_up.setFixedWidth(56)
        self._btn_rgb_down.setFixedWidth(56)
        self._btn_rgb_up.setFocusPolicy(Qt.NoFocus)
        self._btn_rgb_down.setFocusPolicy(Qt.NoFocus)
        self._btn_rgb_up.clicked.connect(self._rgb_plus)
        self._btn_rgb_down.clicked.connect(self._rgb_minus)
        btn_layout.addWidget(self._btn_rgb_up)
        btn_layout.addWidget(self._btn_rgb_down)
        btn_layout.addStretch()
        top_layout.addLayout(btn_layout)

        main_layout.addLayout(top_layout)

        # --- Bottom: Cancel / OK ---
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self._on_cancel)
        main_layout.addWidget(btn_box)

        self._ok_button = btn_box.button(QtWidgets.QDialogButtonBox.Ok)

        for sb in (self._r, self._g, self._b,
                    self._tol_r, self._tol_g, self._tol_b):
            sb.valueChanged.connect(self._on_value_changed)
        self._range_w.valueChanged.connect(self._on_range_changed)
        self._range_h.valueChanged.connect(self._on_range_changed)

        self.adjustSize()

    def focusDefault(self):
        """Set default focus to the OK button."""
        self._ok_button.setFocus()

    def _make_spinbox(self, lo, hi, step):
        sb = QtWidgets.QSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setFixedWidth(60)
        return sb

    def _rgb_plus(self):
        """Increase all three tolerances by 5."""
        for sb in (self._tol_r, self._tol_g, self._tol_b):
            sb.blockSignals(True)
            sb.setValue(min(255, sb.value() + 5))
            sb.blockSignals(False)
        self.paramsChanged.emit()

    def _rgb_minus(self):
        """Decrease all three tolerances by 5."""
        for sb in (self._tol_r, self._tol_g, self._tol_b):
            sb.blockSignals(True)
            sb.setValue(max(1, sb.value() - 5))
            sb.blockSignals(False)
        self.paramsChanged.emit()

    def _on_ok(self):
        self.accepted_signal.emit()

    def _on_cancel(self):
        self.rejected_signal.emit()

    def _on_value_changed(self):
        self.paramsChanged.emit()

    def _on_range_changed(self):
        self.rangeChanged.emit(self._range_w.value(), self._range_h.value())

    def setValues(self, r, g, b, tol_r, tol_g, tol_b):
        for sb in (self._r, self._g, self._b,
                    self._tol_r, self._tol_g, self._tol_b):
            sb.blockSignals(True)
        self._r.setValue(r)
        self._g.setValue(g)
        self._b.setValue(b)
        self._tol_r.setValue(tol_r)
        self._tol_g.setValue(tol_g)
        self._tol_b.setValue(tol_b)
        for sb in (self._r, self._g, self._b,
                    self._tol_r, self._tol_g, self._tol_b):
            sb.blockSignals(False)

    def setRange(self, w, h):
        self._range_w.blockSignals(True)
        self._range_h.blockSignals(True)
        self._range_w.setValue(w)
        self._range_h.setValue(h)
        self._range_w.blockSignals(False)
        self._range_h.blockSignals(False)

    def rgb(self):
        return [self._r.value(), self._g.value(), self._b.value()]

    def tolerances(self):
        return [self._tol_r.value(), self._tol_g.value(), self._tol_b.value()]

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down):
            focused = QtWidgets.QApplication.focusWidget()
            if isinstance(focused, QtWidgets.QSpinBox):
                # Spinbox focused: adjust that spinbox by ±5
                if key == Qt.Key_Up:
                    focused.setValue(
                        min(focused.maximum(), focused.value() + 5)
                    )
                else:
                    focused.setValue(
                        max(focused.minimum(), focused.value() - 5)
                    )
                return
            # No spinbox focused: adjust all ±R/±G/±B
            if key == Qt.Key_Up:
                self._rgb_plus()
            else:
                self._rgb_minus()
            return
        super().keyPressEvent(event)


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
        self._auto_fit_tolerance_sq = kwargs.pop("auto_fit_tolerance", 0.2) ** 2
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
                "magic_wand": False,
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
        self._auto_fit_enabled = False
        self._mw_tolerances: list[int] = [10, 10, 10]
        self._mw_click_pos: QPointF | None = None
        self._mw_contour = None
        self._mw_active: bool = False
        self._mw_base_rgb: list[int] = [0, 0, 0]
        self._mw_image = None
        self._mw_dialog: _MagicWandPanel | None = None
        self._mw_range_w: int | None = None
        self._mw_range_h: int | None = None
        # Accumulated clicks for multi-click union selection.
        # Each entry: {"pos": QPointF, "rgb": list[int],
        #              "tolerances": list[int], "range_w": int|None,
        #              "range_h": int|None, "frozen_mask": np.ndarray|None}
        # Only the last click is "working" (frozen_mask=None, live flood fill
        # controlled by the panel). All earlier clicks are frozen snapshots
        # and never recomputed.
        self._mw_clicks: list[dict] = []
        self._auto_fit_guides: list[tuple[int, float]] = []
        self._auto_fit_dots: list[tuple[float, float]] = []
        self._auto_fit_count: int = 0
        # edge → (source, abs coord); source is "line_fit" or "text_bounding".
        # Only line_fit entries get rendered as red dashed lines.
        self._auto_fit_snap_targets: dict[int, tuple[str, float]] = {}
        self._auto_fit_last_detect_pos: QPointF | None = None
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
            or (self.createMode in ["point", "polygon", "polygon3", "rectangle", "magic_wand"] and self.drawing() and self.prevMovePoint is not None)
        ) and not self._near_start_point

        # Determine crosshair color
        if self.hShape is not None:
            crosshair_color = QtGui.QColor(self.hShape.line_color)
        elif self.current is not None:
            crosshair_color = QtGui.QColor(self.current.line_color)
        elif self.createMode in ["point", "polygon", "polygon3", "rectangle", "magic_wand"]:
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

    def setAutoFitEnabled(self, enabled: bool):
        self._auto_fit_enabled = enabled

    def setReferenceMedians(self, medians: dict[str, float | None]):
        self._reference_medians = medians

    def setDarkPixelMagnetEnabled(self, enabled: bool):
        self._dark_pixel_magnet_enabled = enabled

    def setDarkPixelMagnetConfig(self, config: list[dict]):
        self._dark_pixel_magnet_config = config

    def setPendingDrawLabel(self, label: str | None):
        self._pending_draw_label = label

    def _ensureImgArrCache(self) -> None:
        """Pixel array (BGRA) plus an id for the image, built on first use.

        Used by the pixel readout and as the AI session's image id. The id
        is QPixmap.cacheKey() rather than a hash of every byte: hashing a 4K
        frame cost ~34 ms on each image while the key is free. A reloaded
        image gets a new key, so an AI embedding is recomputed instead of
        reused — correct, just occasionally slower.
        """
        if self._img_arr_ready:
            return
        self._img_arr_ready = True
        if self.pixmap is None or self.pixmap.isNull():
            self._pixmap_hash = None
            self._img_arr_cache = None
            return
        img_arr = labelme.utils.img_qt_to_arr(img_qt=self.pixmap.toImage())
        self._pixmap_hash = self.pixmap.cacheKey()
        self._img_arr_cache = img_arr.copy() if img_arr.ndim >= 2 else None

    def _ensureGrayscaleCache(self) -> None:
        """Luminance map for the magnets, built on first use.

        Only the magnets need it, so images are never converted unless one
        is actually active. cv2 converts a 4K frame in ~5 ms where the numpy
        dot product took ~400 ms; it rounds to nearest instead of truncating,
        so single levels can differ by 1 (the numpy version already varied by
        1 with memory layout, and the magnets threshold with consensus over
        many samples, so detection is unaffected).
        """
        if self._grayscale_ready:
            return
        self._grayscale_ready = True
        if self.pixmap is None or self.pixmap.isNull():
            self._grayscale_cache = None
            return
        self._ensureImgArrCache()
        img_arr = self._img_arr_cache
        if img_arr is None:
            self._grayscale_cache = None
        elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
            # Qt ARGB32 little-endian stores as BGRA
            self._grayscale_cache = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2GRAY)
        elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
            self._grayscale_cache = cv2.cvtColor(
                np.ascontiguousarray(img_arr), cv2.COLOR_BGR2GRAY
            )
        elif img_arr.ndim == 2:
            self._grayscale_cache = img_arr.astype(np.uint8)
        else:
            self._grayscale_cache = None

    def _ensureImageCaches(self) -> None:
        """Build every image cache (array, id and luminance)."""
        self._ensureImgArrCache()
        self._ensureGrayscaleCache()

    def warmImageCaches(self) -> None:
        """Build the image caches ahead of the first interaction.

        loadPixmap defers them so switching files stays fast, but the status
        bar's pixel readout and the magnets need them, so the cost would
        otherwise land on the user's first mouse move as a visible freeze
        (~600 ms on a 4K image). Callers schedule this right after a load,
        once the window has been painted.
        """
        if self.pixmap is None or self.pixmap.isNull():
            return
        self._ensureImgArrCache()
        if (
            self._parallel_line_dist_enabled
            or self._text_bounding_enabled
            or self._line_fit_enabled
            or self._auto_fit_enabled
            or self._dark_pixel_magnet_enabled
        ):
            self._ensureGrayscaleCache()

    def getPixelInfo(self, pos: QPointF):
        """Return (R, G, B, Gray) at image position, or None if out of bounds."""
        self._ensureImgArrCache()
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
            "polygon3",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
            "magic_wand",
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
        self._ensureImgArrCache()  # _pixmap_hash is used as the AI image id
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

    def _clearStaleHoverState(self):
        """Drop hover/highlight references to shapes no longer in self.shapes.

        Deletion, undo/redo and load all replace or shrink the shape list;
        a lingering hShape/prevhShape pointing at a removed shape is painted
        nowhere but can be resurrected by press/release handlers and then
        crashes shapes.index() or mutates a dead shape.
        """
        if self.hShape is not None and self.hShape not in self.shapes:
            self.hShape.highlightClear()
            self.hShape = None
            self.hVertex = None
            self.hEdge = None
            self.hEdgeMidpoint = None
        if self.prevhShape is not None and self.prevhShape not in self.shapes:
            self.prevhShape = None
            self.prevhVertex = None
            self.prevhEdge = None
            self.prevhEdgeMidpoint = None

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
        self._clearStaleHoverState()
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
        self._clearStaleHoverState()
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
        # Keep the editing state while a drag is in progress: dropping hShape
        # here loses the shape being moved, so mouseReleaseEvent can neither
        # store an undo snapshot nor emit shapeMoved and the edit is silently
        # left out of the undo history (and of the dirty state).
        dragging = (
            self._vertex_dragging
            or self._edge_midpoint_dragging
            or self.movingShape
            or bool(self.selectedShapesCopy)
            or self._mouse_pressed
        )
        if not dragging:
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

    def cancelDrawing(self):
        """Cancel any in-progress drawing and clear per-draw transient state.

        Shared by the ESC key and by mode switches so an unfinished shape
        (or its preview leftovers) never leaks into the next mode.
        """
        if self._mw_active:
            self._magic_wand_cancel()  # restores the cursor itself
        else:
            if self.current is not None:
                self.current = None
                self.drawingPolygon.emit(False)
            # Restore even when no shape is in progress: the cursor can be
            # hidden with current=None (e.g. hidden near the start point,
            # then the last vertex undone before switching modes).
            self._unhide_os_cursor()
            self.restoreCursor()
        # Transient draw state: clear even when no shape was in progress
        # (e.g. leftover preview line / DPM ghost after a bare mode switch).
        self.line.points = []
        self.line.point_labels = []
        self._undone_points = []
        self._dpm_ghost_pos = None
        self._near_start_point = False
        self.update()

    def setEditing(self, value=True):
        self.mode = CanvasMode.EDIT if value else CanvasMode.CREATE
        if self.mode == CanvasMode.EDIT:
            # CREATE -> EDIT
            if self._mw_active:
                self._mw_active = False
                self._mw_contour = None
                self._mw_image = None
                self._mw_clicks = []
                self._close_mw_panel()
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
            if self._mw_active and self.current:
                r, g, b = self._mw_base_rgb
                tr, tg, tb = self._mw_tolerances
                n_clicks = len(self._mw_clicks)
                messages.append(
                    self.tr("魔法の杖[%d]: RGB(%d,%d,%d) ±(%d,%d,%d)")
                    % (n_clicks, r, g, b, tr, tg, tb)
                )
                messages.append(self.tr("左クリックで追加 / 右クリックで戻す"))
                messages.append(self.tr("Enter確定 / ESCキャンセル"))
            else:
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
            prev_ghost_pos = self._dpm_ghost_pos
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
            moved_index = self.hVertex
            moved_shape = self.hShape
            old_point = None
            if (
                moved_shape is not None
                and moved_index is not None
                and moved_index < len(moved_shape.points)
            ):
                old_point = QPointF(moved_shape.points[moved_index])
            before_rect = self._shapeDeviceRect(moved_shape)
            self.boundedMoveVertex(pos, is_shift_pressed=is_shift_pressed)
            self._updateCursorOverlay()
            # update() (not repaint()): coalesce to one paint per frame on
            # 60-120Hz mouse-move streams; rendered frames are identical.
            # Only the dragged shape changes, so invalidate just the area it
            # can repaint differently instead of the whole canvas; combined
            # with the paint-time culling this stops every other shape from
            # being re-rendered on each drag frame.
            dirty = self._vertexMoveDeviceRect(moved_shape, moved_index, old_point)
            if dirty is None:
                after_rect = self._shapeDeviceRect(moved_shape)
                if before_rect is not None and after_rect is not None:
                    dirty = before_rect.united(after_rect)
            if dirty is not None:
                m = self._shapeDrawMargin(moved_shape)
                dirty = dirty.adjusted(-m, -m, m, m)
                # The dark-pixel-magnet ghost marker is drawn at the raw
                # cursor position, which can sit outside the shape bounds.
                ghost = self._ghostDeviceRect(self._dpm_ghost_pos)
                if ghost is not None:
                    dirty = dirty.united(ghost)
                ghost_prev = self._ghostDeviceRect(prev_ghost_pos)
                if ghost_prev is not None:
                    dirty = dirty.united(ghost_prev)
                self.update(dirty.toAlignedRect())
            else:
                self.update()
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
            elif self.createMode in ("magic_wand", "polygon3"):
                # polygon3 is a drawing mode, not a Shape type: it produces
                # ordinary polygons, so the preview line uses "polygon".
                self.line.shape_type = "polygon"
            else:
                self.line.shape_type = self.createMode

            if self.current or self.createMode in ["point", "polygon", "polygon3", "rectangle", "magic_wand"]:
                # Hide cursor when drawing (show crosshair instead)
                if self._custom_cursor_enabled:
                    self.overrideCursor(self._blank_cursor)
                else:
                    self.overrideCursor(CURSOR_DRAW)
            else:
                self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self._updateCursorOverlay()
                # Modes whose full-span crosshair is drawn by paintEvent need
                # a full-canvas repaint per move: the old cross lines used to
                # be erased as a side effect of the overlay's full-widget
                # update(), which is now region-limited. (Keeping this a full
                # update is deliberate — a stale-prone coordinate cache for
                # strip invalidation was tried and reverted per review.)
                if (
                    self._crosshair.get(
                        "polygon" if self._createMode == "polygon3"
                        else self._createMode, False
                    )
                    and self._createMode not in ("point", "polygon", "polygon3")
                ):
                    self.update()
                self._update_status()
                return
            # Magic wand preview: don't follow mouse
            if self._mw_active:
                self._updateCursorOverlay()
                return

            # Keep the start-vertex highlight until the queued paint has
            # consumed it.  Clear the previous move's transient state here,
            # then set it again below when the latest cursor position is
            # still close enough to the first vertex.
            previous_line_rect = self._shapeDeviceRect(self.line)
            previous_ghost_pos = self._dpm_ghost_pos
            was_near_start = self._near_start_point
            self.current.highlightClear()

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode in ("polygon", "polygon3")
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
                and self.createMode in ("polygon", "polygon3")
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
            if self.createMode in ["polygon", "polygon3", "linestrip"]:
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
            # Queue drawing rather than forcing a synchronous full-canvas
            # repaint for every mouse event.  Qt can now coalesce high-rate
            # mouse moves into one frame, and modes without a full-span
            # crosshair only invalidate the old/new preview footprint.
            full_span_crosshair = (
                self._crosshair.get(
                    "polygon" if self._createMode == "polygon3"
                    else self._createMode, False
                )
                and self._createMode not in ("point", "polygon", "polygon3")
            )
            if full_span_crosshair:
                self.update()
            else:
                dirty_rects = []
                line_margin = self._shapeDrawMargin(self.line)
                for rect in (
                    previous_line_rect,
                    self._shapeDeviceRect(self.line),
                ):
                    if rect is not None:
                        dirty_rects.append(
                            rect.adjusted(
                                -line_margin,
                                -line_margin,
                                line_margin,
                                line_margin,
                            )
                        )
                for ghost_pos in (
                    previous_ghost_pos,
                    self._dpm_ghost_pos,
                ):
                    ghost_rect = self._ghostDeviceRect(ghost_pos)
                    if ghost_rect is not None:
                        dirty_rects.append(ghost_rect)
                if was_near_start != self._near_start_point:
                    first = self.current[0]
                    offset = self.offsetToCenter()
                    cx = (first.x() + offset.x()) * self.scale
                    cy = (first.y() + offset.y()) * self.scale
                    highlight_margin = (
                        float(Shape.point_size) * 1.5
                        + float(Shape.PEN_WIDTH) / 2.0
                        + self._PAINT_MARGIN_SLACK
                    )
                    dirty_rects.append(
                        QtCore.QRectF(
                            cx - highlight_margin,
                            cy - highlight_margin,
                            2.0 * highlight_margin,
                            2.0 * highlight_margin,
                        )
                    )
                if dirty_rects:
                    dirty = dirty_rects[0]
                    for rect in dirty_rects[1:]:
                        dirty = dirty.united(rect)
                    self.update(dirty.toAlignedRect())
                else:
                    self.update()
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
                self.update()
                self.movingShape = True
            elif (
                self.selectedShapes
                and self.prevPoint is not None
                and not (a0.modifiers() & Qt.ControlModifier)
            ):
                # Ctrl/Cmd is the multi-select modifier: dragging with it held
                # is the user picking shapes, not moving them.
                self.overrideCursor(CURSOR_MOVE)
                before_rects = self._shapesDeviceRect(self.selectedShapes)
                self.boundedMoveShapes(self.selectedShapes, pos)
                if (
                    self._auto_fit_enabled
                    and self.hShape
                    and self.hShape.shape_type == "rectangle"
                    and not is_shift_pressed
                ):
                    # Re-detect when cursor moved ≥ tolerance from last detection
                    lp = self._auto_fit_last_detect_pos
                    if lp is None or (
                        (pos.x() - lp.x()) ** 2 + (pos.y() - lp.y()) ** 2
                        >= self._auto_fit_tolerance_sq
                    ):
                        self._autoFitDetectAndStore(self.hShape, merge=True)
                        self._auto_fit_last_detect_pos = QPointF(pos)
                    # Always re-apply cached snap targets
                    self._reapplyAutoFitSnaps(self.hShape)
                else:
                    self._auto_fit_snap_targets.clear()
                    self._auto_fit_last_detect_pos = None
                    self._autoFitClearGuides()
                # Only the dragged shapes move: repaint their old and new
                # areas instead of the whole canvas. Fall back to a full
                # update when auto-fit guides are on screen (they span the
                # canvas) or when a bound cannot be derived.
                after_rects = self._shapesDeviceRect(self.selectedShapes)
                if (
                    before_rects is not None
                    and after_rects is not None
                    and not self._auto_fit_guides
                    and self._snap_line_pos is None
                ):
                    self.update(before_rects.united(after_rects).toAlignedRect())
                else:
                    self.update()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        status_messages: list[str] = []
        prev_hover_state = (
            self.hShape, self.hVertex, self.hEdge, self.hEdgeMidpoint,
        )
        prev_label_visible = (
            self._hover_label_ready
            and self._hover_label_shape is not None
            and self._hover_label_last_pos is not None
        )
        prev_label_shape = self._hover_label_shape
        prev_label_pos = self._hover_label_last_pos
        # Iterate selected shapes first, then the rest (by hover order)
        _hover_iter = [
            s for s in self.selectedShapes if s in self._shapes_hover_order
        ] + [s for s in self._shapes_hover_order if s not in self.selectedShapes]
        # Widest reach of the hit tests below, in image units: the edge
        # midpoint capsule (epsilon * 4) and the point-shape handle are the
        # largest, and shapes are stored in image coordinates while epsilon
        # is in device pixels.
        hover_margin = (
            max(self.epsilon * 4.0, Shape.point_object_size * 1.5)
            / max(self.scale, 1e-9)
        )
        for shape in _hover_iter:
            if not self.isVisible(shape):
                continue
            # Cheap rejection first: everything below (vertex, edge, edge
            # midpoint and interior tests) only reports a hit within
            # hover_margin of the shape's own bounds, so shapes further away
            # need no per-vertex work at all. Bounds come straight from the
            # points, nothing is cached, so this cannot go stale.
            bounds = self._shapeImageBounds(shape)
            if bounds is not None:
                min_x, min_y, max_x, max_y = bounds
                if (
                    pos.x() < min_x - hover_margin
                    or pos.x() > max_x + hover_margin
                    or pos.y() < min_y - hover_margin
                    or pos.y() > max_y + hover_margin
                ):
                    continue
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            # nearestEdge is only consumed by the `elif index_edge ...
            # and shape.canAddPoint()` branch below, so skip the O(edges)
            # scan when a vertex already matched or the shape can never
            # take a new point (both functions are pure reads).
            index_edge = None
            if index is None and shape.canAddPoint():
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
        # Repaint only when something visible changed, and only where: plain
        # mouse motion that leaves the highlight (and the hover label) as it
        # was renders an identical frame, and re-rendering every shape for it
        # is what made hovering over a shape-heavy image stutter.
        hover_state = (self.hShape, self.hVertex, self.hEdge, self.hEdgeMidpoint)
        if hover_state != prev_hover_state or prev_label_visible:
            dirty: QtCore.QRectF | None = QtCore.QRectF()
            for sh in (prev_hover_state[0], self.hShape):
                if sh is None:
                    continue
                rect = self._shapeDeviceRect(sh)
                if rect is None:  # extent unknown (mask): repaint everything
                    dirty = None
                    break
                m = self._shapeDrawMargin(sh)
                dirty = dirty.united(rect.adjusted(-m, -m, m, m))
            if dirty is not None and prev_label_visible:
                label_rect = self._hoverLabelDeviceRect(
                    prev_label_shape, prev_label_pos
                )
                if label_rect is None:
                    dirty = None
                else:
                    dirty = dirty.united(label_rect)
            if dirty is None or dirty.isNull():
                self.update()
            else:
                self.update(dirty.toAlignedRect())

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

        # Magic wand: right-click undoes the most recent click
        if a0.button() == Qt.RightButton and self._mw_active:
            self._magic_wand_undo_click()
            return

        # Magic wand: left-click while active appends another click.
        # Must intercept before the generic "add point to existing shape"
        # branch, which would otherwise fall through with self.current set.
        if (
            a0.button() == Qt.LeftButton
            and self._mw_active
            and self.createMode == "magic_wand"
            and not self.outOfPixmap(pos)
        ):
            self._magic_wand_select(pos)
            return

        if a0.button() == Qt.LeftButton:
            if self.drawing():
                self._undone_points.clear()
                redo_action = getattr(self, "_redo_action", None)
                if redo_action is not None:
                    redo_action.setEnabled(False)
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon3":
                        # Fixed 3-vertex polygon: close as soon as the third
                        # point is placed (no need to click the start point).
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if len(self.current.points) >= 3:
                            self.current.close()
                            self.finalise()
                    elif self.createMode == "polygon":
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
                    if self.createMode == "magic_wand":
                        self._magic_wand_select(pos)
                        return

                    if self.createMode in ["ai_polygon", "ai_mask"]:
                        if not download_ai_model(
                            model_name=self._osam_session_model_name, parent=self
                        ):
                            return

                    # Create new shape.
                    if self.createMode in ["ai_polygon", "ai_mask"]:
                        new_shape_type = "points"
                    elif self.createMode == "polygon3":
                        new_shape_type = "polygon"  # saved as a normal polygon
                    else:
                        new_shape_type = self.createMode
                    self.current = Shape(shape_type=new_shape_type)
                    self.current._is_creating = True  # Mark as being created
                    # Dark pixel magnet on first click (polygon only)
                    if (
                        not is_shift_pressed
                        and self.createMode in ("polygon", "polygon3")
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

                group_mode = int(a0.modifiers()) == Qt.ControlModifier
                if group_mode:
                    # Multi-select click: pick shapes only. Grabbing a vertex
                    # or an edge here would start dragging geometry while the
                    # user is just building up a selection.
                    if self.hShape is not None:
                        self.hShape.highlightClear()
                    self.hVertex = None
                    self.hEdgeMidpoint = None

                # If no hover vertex/edge midpoint is set, resolve nearest vertex
                if not group_mode and self.hVertex is None and self.hEdgeMidpoint is None:
                    _click_iter = [
                        s for s in self.selectedShapes
                        if s in self._shapes_hover_order
                    ] + [
                        s for s in self._shapes_hover_order
                        if s not in self.selectedShapes
                    ]
                    for shape in _click_iter:
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
                self._repaintEditedShapes()
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
            had_guides = bool(self._auto_fit_guides) or self._snap_line_pos is not None
            self._auto_fit_snap_targets.clear()
            self._auto_fit_last_detect_pos = None
            self._autoFitClearGuides()
            if had_guides:
                self.repaint()  # guides spanned the canvas: clear them all
            else:
                self._repaintEditedShapes()

            # Defensive: hShape must be in shapes (cache coherence), but a
            # stale hover reference must not crash the app mid-annotation.
            if self.hShape in self.shapes:
                index = self.shapes.index(self.hShape)
                if (
                    self.shapesBackups
                    and len(self.shapesBackups[-1]) > index
                    and self.shapesBackups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.hShape.touch()  # Update modification timestamp
                    self.storeShapes()
                    self.shapeMoved.emit()
            else:
                logger.warning(
                    "movingShape released with stale hShape not in shapes; "
                    "ignoring (cache coherence bug?)"
                )
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
        prev_selected = list(self.selectedShapes)
        self.setHiding()
        self.selectionChanged.emit(shapes)
        # Selection only changes how the affected shapes are drawn (fill and
        # outline), so repaint just those instead of every shape.
        if self._hideBackround or self.hideBackround:
            self.update()  # hiding mode repaints everything anyway
            return
        changed = [
            sh
            for sh in set(prev_selected) | set(self.selectedShapes)
            if (sh in prev_selected) != (sh in self.selectedShapes)
        ]
        # Invalidate each shape's own area rather than their bounding box:
        # selecting a shape far from the previous one would otherwise repaint
        # everything in between.
        region = self._shapesDeviceRegion(changed) if changed else None
        if region is None:
            self.update()
        else:
            self.update(region)

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
        if multiple_selection_mode:
            # Ctrl/Cmd-clicking empty space while building a selection is a
            # miss, not "clear everything".
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

        if not shift_held:
            self._ensureGrayscaleCache()
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

    def _autoFitDetect(self, shape):
        """Detect auto-fit snap positions for all edges (preview only, no shape change).

        Returns list of (edge_index, snap_pos, source) and sets visual feedback
        variables. `source` is "line_fit" or "text_bounding".
        """
        self._auto_fit_guides = []
        self._auto_fit_dots = []
        self._auto_fit_count = 0

        if shape.shape_type != "rectangle" or len(shape.points) != 2:
            return []
        self._ensureGrayscaleCache()
        if self._grayscale_cache is None:
            return []
        pending = []  # [(edge_index, snap_pos, source), ...]

        for edge_index in (
            Shape.EDGE_TOP, Shape.EDGE_BOTTOM,
            Shape.EDGE_LEFT, Shape.EDGE_RIGHT,
        ):
            p0, p1 = shape.points[0], shape.points[1]
            if edge_index == Shape.EDGE_TOP:
                pos = QPointF(
                    (p0.x() + p1.x()) / 2, min(p0.y(), p1.y())
                )
            elif edge_index == Shape.EDGE_BOTTOM:
                pos = QPointF(
                    (p0.x() + p1.x()) / 2, max(p0.y(), p1.y())
                )
            elif edge_index == Shape.EDGE_LEFT:
                pos = QPointF(
                    min(p0.x(), p1.x()), (p0.y() + p1.y()) / 2
                )
            elif edge_index == Shape.EDGE_RIGHT:
                pos = QPointF(
                    max(p0.x(), p1.x()), (p0.y() + p1.y()) / 2
                )

            snap_found = False

            # 1. Line fit snap
            if self._line_fit_enabled and not snap_found:
                for rule in self._line_fit_magnet_config:
                    if shape.label != rule.get("target_label"):
                        continue
                    result = self._detect_line_fit_snap(
                        rule, shape, edge_index, pos
                    )
                    if result is not None:
                        pending.append((edge_index, result[0], "line_fit"))
                        self._auto_fit_guides.append(
                            (edge_index, result[1])
                        )
                        snap_found = True
                        break

            # 2. Text bounding snap
            if self._text_bounding_enabled and not snap_found:
                for i, rule in enumerate(
                    self._text_bounding_magnet_config
                ):
                    if shape.label != rule.get("target_label"):
                        continue
                    result = self._detect_text_bounding_snap(
                        rule, i, shape, edge_index, pos
                    )
                    if result is not None:
                        pending.append(
                            (edge_index, result[0], "text_bounding")
                        )
                        if result[1]:
                            self._auto_fit_dots.extend(result[1])
                        snap_found = True
                        break

        self._auto_fit_count = len(pending)
        return pending

    def _autoFitDetectAndStore(self, shape, merge=False):
        """Detect auto-fit snaps and store targets (does NOT modify shape).

        merge=False: clear all targets, store only newly detected ones.
        merge=True:  update detected edges, keep existing targets for
                     edges where detection failed (prevents partial dropout).
        """
        from labelme.shape import Shape

        # Clear edge-drag snap cache to prevent interference
        self._lf_snap_cache = None

        pending = self._autoFitDetect(shape)
        if not merge:
            self._auto_fit_snap_targets.clear()
        for edge_index, snap_pos, source in pending:
            if edge_index in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM):
                coord = snap_pos.y()
            else:
                coord = snap_pos.x()
            self._auto_fit_snap_targets[edge_index] = (source, coord)

        # Rebuild guides from line_fit-source targets only. text_bounding
        # entries are intentionally excluded so that the red dashed line
        # never appears unless the line-fit checkbox is on.
        self._auto_fit_guides = [
            (ei, coord)
            for ei, (src, coord) in self._auto_fit_snap_targets.items()
            if src == "line_fit"
        ]
        self._auto_fit_count = len(self._auto_fit_snap_targets)

    def _reapplyAutoFitSnaps(self, shape):
        """Re-apply cached snap targets after boundedMoveShapes."""
        from labelme.shape import Shape

        if not self._auto_fit_snap_targets:
            return
        for edge_idx, (_src, target) in self._auto_fit_snap_targets.items():
            if edge_idx in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM):
                shape.moveEdgeTo(edge_idx, QPointF(0, target))
            else:
                shape.moveEdgeTo(edge_idx, QPointF(target, 0))

    def _autoFitClearGuides(self):
        """Clear auto-fit visual feedback."""
        self._auto_fit_guides = []
        self._auto_fit_dots = []
        self._auto_fit_count = 0

    # -- Line fit magnet defaults --
    _LF_DEFAULTS = {
        "luminance_threshold": 128,
        "resize_base": 2560,
        "snap_range_pixels": 5,
        "sample_points": 21,
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
        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        scale = self._rule_scale(rule, d["resize_base"], img_w, img_h)
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

    @staticmethod
    def _subpix_y(y0, col, grayscale, img_h):
        """Parabolic sub-pixel interpolation for a valley at row y0."""
        ym = max(0, y0 - 1)
        yp = min(img_h - 1, y0 + 1)
        vm = float(grayscale[ym, col])
        v0 = float(grayscale[y0, col])
        vp = float(grayscale[yp, col])
        d = vm - 2.0 * v0 + vp
        off = 0.5 * (vm - vp) / d if abs(d) > 1e-6 else 0.0
        return float(y0) + max(-0.5, min(0.5, off)) + 0.5

    @staticmethod
    def _subpix_x(x0, row, grayscale, img_w):
        """Parabolic sub-pixel interpolation for a valley at col x0."""
        xm = max(0, x0 - 1)
        xp = min(img_w - 1, x0 + 1)
        vm = float(grayscale[row, xm])
        v0 = float(grayscale[row, x0])
        vp = float(grayscale[row, xp])
        d = vm - 2.0 * v0 + vp
        off = 0.5 * (vm - vp) / d if abs(d) > 1e-6 else 0.0
        return float(x0) + max(-0.5, min(0.5, off)) + 0.5

    def _find_line_peak_h(self, xs, iy, max_scan, grayscale, img_h, img_w,
                          consec_window, dist_tol, lum_thresh):
        """Find horizontal line peak by scanning both vertical directions.

        Scans ALL dark regions (not just the first) to collect every candidate
        valley per sample.  Consensus uses inner points with 60% hit rate
        and picks the line with the lowest mean valley luminance.
        """
        n = len(xs)

        # Per-sample: collect ALL dark valleys (sub-pixel y, luminance)
        per_sample: list[list[tuple[float, float]]] = []
        for idx in range(n):
            col = int(round(xs[idx]))
            if col < 0 or col >= img_w:
                per_sample.append([])
                continue
            valleys: list[tuple[float, float]] = []
            # Check center pixel darkness for scan initialization
            clum = float(grayscale[iy, col])
            center_dark = clum <= lum_thresh
            # Scan both directions — find ALL dark regions
            # If center pixel is dark, initialize scan as already in a dark
            # region so the center's dark region is handled as one contiguous
            # region instead of being split into a separate valley.
            for scan_dir in (+1, -1):
                if center_dark:
                    in_dark = True
                    best_lum = clum
                    best_y = iy
                else:
                    in_dark = False
                    best_lum = 256.0
                    best_y = -1
                for d in range(1, max_scan + 1):
                    sy = iy + scan_dir * d
                    if sy < 0 or sy >= img_h:
                        if in_dark and best_y >= 0:
                            valleys.append((
                                self._subpix_y(best_y, col, grayscale, img_h),
                                best_lum,
                            ))
                        break
                    lum = float(grayscale[sy, col])
                    if lum <= lum_thresh:
                        in_dark = True
                        if lum < best_lum:
                            best_lum = lum
                            best_y = sy
                    else:
                        if in_dark:
                            valleys.append((
                                self._subpix_y(best_y, col, grayscale, img_h),
                                best_lum,
                            ))
                            in_dark = False
                            best_lum = 256.0
                            best_y = -1
            per_sample.append(valleys)

        # Collect all unique valley positions as candidate targets
        all_positions: set[int] = set()
        for valleys in per_sample:
            for pos, _lum in valleys:
                all_positions.add(round(pos * 2))  # key in 0.5px units
        if not all_positions:
            return None

        # Consensus: inner points (skip first & last), 60% hit rate
        inner_start = min(1, n - 1)
        inner_end = max(n - 1, 1)
        n_inner = inner_end - inner_start
        min_hits = max((n_inner * 6 + 9) // 10, 1)  # ceil(n_inner * 0.6)

        all_candidates: list[tuple[float, float]] = []
        seen: set[int] = set()
        sorted_targets = sorted(all_positions)

        for tol in (dist_tol * 0.5, dist_tol):
            for tkey in sorted_targets:
                target = tkey / 2.0
                agree_pos: list[float] = []
                agree_lum: list[float] = []
                for i in range(inner_start, inner_end):
                    for pos, lum in per_sample[i]:
                        if abs(pos - target) <= tol:
                            agree_pos.append(pos)
                            agree_lum.append(lum)
                            break
                if len(agree_pos) >= min_hits:
                    peak = float(np.median(agree_pos))
                    key = round(peak * 2)
                    if key in seen:
                        continue
                    seen.add(key)
                    score = sum(agree_lum) / len(agree_lum)
                    all_candidates.append((peak, score))

        if not all_candidates:
            return None
        all_candidates.sort(key=lambda c: c[1])
        initial_peak = all_candidates[0][0]

        # Stage 2: refine ±3px around initial peak for deeper valley
        refine_r = 3
        iy2 = int(round(initial_peak))
        refined_per_sample: list[list[tuple[float, float]]] = []
        for idx in range(n):
            col = int(round(xs[idx]))
            if col < 0 or col >= img_w:
                refined_per_sample.append([])
                continue
            best_lum = 256.0
            best_y = -1
            for sy in range(max(0, iy2 - refine_r), min(img_h, iy2 + refine_r + 1)):
                lum = float(grayscale[sy, col])
                if lum <= lum_thresh and lum < best_lum:
                    best_lum = lum
                    best_y = sy
            if best_y >= 0:
                refined_per_sample.append([
                    (self._subpix_y(best_y, col, grayscale, img_h), best_lum)
                ])
            else:
                refined_per_sample.append([])

        # Refined consensus (inner points, 60% hit rate)
        ref_positions: set[int] = set()
        for valleys in refined_per_sample:
            for pos, _lum in valleys:
                ref_positions.add(round(pos * 2))
        refined_candidates: list[tuple[float, float]] = []
        seen_ref: set[int] = set()
        for tol in (dist_tol * 0.5, dist_tol):
            for tkey in sorted(ref_positions):
                target = tkey / 2.0
                agree_pos2: list[float] = []
                agree_lum2: list[float] = []
                for i in range(inner_start, inner_end):
                    for pos, lum in refined_per_sample[i]:
                        if abs(pos - target) <= tol:
                            agree_pos2.append(pos)
                            agree_lum2.append(lum)
                            break
                if len(agree_pos2) >= min_hits:
                    peak = float(np.median(agree_pos2))
                    key = round(peak * 2)
                    if key in seen_ref:
                        continue
                    seen_ref.add(key)
                    score = sum(agree_lum2) / len(agree_lum2)
                    refined_candidates.append((peak, score))
        if refined_candidates:
            refined_candidates.sort(key=lambda c: c[1])
            return refined_candidates[0][0]
        return initial_peak

    def _find_line_peak_v(self, ys, ix, max_scan, grayscale, img_h, img_w,
                          consec_window, dist_tol, lum_thresh):
        """Find vertical line peak by scanning both horizontal directions.

        Scans ALL dark regions (not just the first) to collect every candidate
        valley per sample.  Consensus uses inner points with 60% hit rate
        and picks the line with the lowest mean valley luminance.
        """
        n = len(ys)

        # Per-sample: collect ALL dark valleys (sub-pixel x, luminance)
        per_sample: list[list[tuple[float, float]]] = []
        for idx in range(n):
            row = int(round(ys[idx]))
            if row < 0 or row >= img_h:
                per_sample.append([])
                continue
            valleys: list[tuple[float, float]] = []
            # Check center pixel darkness for scan initialization
            clum = float(grayscale[row, ix])
            center_dark = clum <= lum_thresh
            # Scan both directions — find ALL dark regions
            # If center pixel is dark, initialize scan as already in a dark
            # region so the center's dark region is handled as one contiguous
            # region instead of being split into a separate valley.
            for scan_dir in (+1, -1):
                if center_dark:
                    in_dark = True
                    best_lum = clum
                    best_x = ix
                else:
                    in_dark = False
                    best_lum = 256.0
                    best_x = -1
                for d in range(1, max_scan + 1):
                    sx = ix + scan_dir * d
                    if sx < 0 or sx >= img_w:
                        if in_dark and best_x >= 0:
                            valleys.append((
                                self._subpix_x(best_x, row, grayscale, img_w),
                                best_lum,
                            ))
                        break
                    lum = float(grayscale[row, sx])
                    if lum <= lum_thresh:
                        in_dark = True
                        if lum < best_lum:
                            best_lum = lum
                            best_x = sx
                    else:
                        if in_dark:
                            valleys.append((
                                self._subpix_x(best_x, row, grayscale, img_w),
                                best_lum,
                            ))
                            in_dark = False
                            best_lum = 256.0
                            best_x = -1
            per_sample.append(valleys)

        # Collect all unique valley positions as candidate targets
        all_positions: set[int] = set()
        for valleys in per_sample:
            for pos, _lum in valleys:
                all_positions.add(round(pos * 2))  # key in 0.5px units
        if not all_positions:
            return None

        # Consensus: inner points (skip first & last), 60% hit rate
        inner_start = min(1, n - 1)
        inner_end = max(n - 1, 1)
        n_inner = inner_end - inner_start
        min_hits = max((n_inner * 6 + 9) // 10, 1)  # ceil(n_inner * 0.6)

        all_candidates: list[tuple[float, float]] = []
        seen: set[int] = set()
        sorted_targets = sorted(all_positions)

        for tol in (dist_tol * 0.5, dist_tol):
            for tkey in sorted_targets:
                target = tkey / 2.0
                agree_pos: list[float] = []
                agree_lum: list[float] = []
                for i in range(inner_start, inner_end):
                    for pos, lum in per_sample[i]:
                        if abs(pos - target) <= tol:
                            agree_pos.append(pos)
                            agree_lum.append(lum)
                            break
                if len(agree_pos) >= min_hits:
                    peak = float(np.median(agree_pos))
                    key = round(peak * 2)
                    if key in seen:
                        continue
                    seen.add(key)
                    score = sum(agree_lum) / len(agree_lum)
                    all_candidates.append((peak, score))

        if not all_candidates:
            return None
        all_candidates.sort(key=lambda c: c[1])
        initial_peak = all_candidates[0][0]

        # Stage 2: refine ±3px around initial peak for deeper valley
        refine_r = 3
        ix2 = int(round(initial_peak))
        refined_per_sample: list[list[tuple[float, float]]] = []
        for idx in range(n):
            row = int(round(ys[idx]))
            if row < 0 or row >= img_h:
                refined_per_sample.append([])
                continue
            best_lum = 256.0
            best_x = -1
            for sx in range(max(0, ix2 - refine_r), min(img_w, ix2 + refine_r + 1)):
                lum = float(grayscale[row, sx])
                if lum <= lum_thresh and lum < best_lum:
                    best_lum = lum
                    best_x = sx
            if best_x >= 0:
                refined_per_sample.append([
                    (self._subpix_x(best_x, row, grayscale, img_w), best_lum)
                ])
            else:
                refined_per_sample.append([])

        # Refined consensus (inner points, 60% hit rate)
        ref_positions: set[int] = set()
        for valleys in refined_per_sample:
            for pos, _lum in valleys:
                ref_positions.add(round(pos * 2))
        refined_candidates: list[tuple[float, float]] = []
        seen_ref: set[int] = set()
        for tol in (dist_tol * 0.5, dist_tol):
            for tkey in sorted(ref_positions):
                target = tkey / 2.0
                agree_pos2: list[float] = []
                agree_lum2: list[float] = []
                for i in range(inner_start, inner_end):
                    for pos, lum in refined_per_sample[i]:
                        if abs(pos - target) <= tol:
                            agree_pos2.append(pos)
                            agree_lum2.append(lum)
                            break
                if len(agree_pos2) >= min_hits:
                    peak = float(np.median(agree_pos2))
                    key = round(peak * 2)
                    if key in seen_ref:
                        continue
                    seen_ref.add(key)
                    score = sum(agree_lum2) / len(agree_lum2)
                    refined_candidates.append((peak, score))
        if refined_candidates:
            refined_candidates.sort(key=lambda c: c[1])
            return refined_candidates[0][0]
        return initial_peak

    @staticmethod
    def _rule_scale(rule: dict, default_resize_base, img_w: int, img_h: int) -> float:
        """Pixel scale for a magnet rule.

        Rules are normally written against a reference image whose longer
        side is ``resize_base``, so their *_pixels values scale with the
        actual image. ``absolute_pixels: true`` turns that off and makes
        them plain image pixels regardless of image size.
        """
        if rule.get("absolute_pixels"):
            return 1.0
        resize_base = rule.get("resize_base", default_resize_base)
        return max(img_w, img_h) / resize_base

    # -- Parallel line magnet detection defaults --
    # Measurement slack (degrees) added to max_tilt_degrees, see below.
    _PL_TILT_SLACK_DEG = 1.0
    _PL_DEFAULTS = {
        # Largest tilt (degrees) a detected line may have relative to the
        # rectangle edge. 0 keeps the original parallel-only behaviour.
        "max_tilt_degrees": 0.0,
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
        "scan_offset_pixels": 10,
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
            not self._dark_pixel_magnet_enabled
            or not self._dark_pixel_magnet_config
            or label is None
        ):
            return pos
        self._ensureGrayscaleCache()
        if (
            self._grayscale_cache is None
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
        snap_range = rule.get("snap_range_pixels", d["snap_range_pixels"])

        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        img_scale = self._rule_scale(rule, d["resize_base"], img_w, img_h)
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
            # "any dark pixel within the 3x3 neighbourhood" (±1px reach).
            # This used to be a 3x3 box convolution built from a float32
            # integral image, which allocated several full-size copies and
            # took ~0.5 s on a 4K frame; a 3x3 dilation is the same predicate
            # and runs in milliseconds.
            binary = (grayscale <= lum_thresh).astype(np.uint8)
            dilated = cv2.dilate(
                binary,
                np.ones((3, 3), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            self._dpm_reachable_cache[lum_thresh] = dilated > 0

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

        max_tilt = float(rule.get("max_tilt_degrees", d["max_tilt_degrees"]))
        # Pixel quantization makes the measured tilt drift by a few tenths of a
        # degree, so add a small slack: a line drawn at exactly max_tilt must
        # still be accepted.
        max_slope = (
            math.tan(math.radians(min(max_tilt + self._PL_TILT_SLACK_DEG, 89.0)))
            if max_tilt > 0
            else 0.0
        )

        M = margin
        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape
        scale = self._rule_scale(rule, d["resize_base"], img_w, img_h)
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
                consec_window, dist_tol, lum_thresh, max_slope,
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
                consec_window, dist_tol, lum_thresh, max_slope,
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

    @staticmethod
    def _fit_tilted_line(coords, values, tol, max_slope, anchor):
        """Robustly fit values = slope * coords + intercept (Theil-Sen).

        Used when the detected line may be slightly tilted with respect to the
        rectangle edge: a plain median of the per-sample distances would smear
        such a line out, while the fit keeps it sharp and evaluates the
        distance at ``anchor`` (the middle of the edge).

        Returns ``(inlier_indices, value_at_anchor)``, or None when the line is
        tilted by more than ``max_slope`` or too few samples agree with it.
        """
        n = len(coords)
        if n < 3:
            return None
        slopes = []
        for i in range(n - 1):
            ci = coords[i]
            vi = values[i]
            for j in range(i + 1, n):
                dc = coords[j] - ci
                if dc != 0.0:
                    slopes.append((values[j] - vi) / dc)
        if not slopes:
            return None
        slope = float(np.median(slopes))
        if abs(slope) > max_slope:
            return None
        intercept = float(
            np.median([values[i] - slope * coords[i] for i in range(n)])
        )
        inliers = [
            i
            for i in range(n)
            if abs(values[i] - (slope * coords[i] + intercept)) <= tol
        ]
        if not inliers:
            return None
        # Refit on the inliers only so a couple of outliers cannot bias the
        # value we report at the anchor.
        if len(inliers) < n:
            intercept = float(
                np.median([values[i] - slope * coords[i] for i in inliers])
            )
        return (inliers, slope * anchor + intercept)

    def _detect_line_h(self, xs, iy, scan_dir, max_scan, grayscale, img_h, img_w,
                       consec_window, dist_tol, lum_thresh, max_slope=0.0):
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
        anchor = float(np.mean(xs)) if len(xs) else 0.0
        for cur_tol in (tol * 0.5, tol):
            for start in range(max(n - w + 1, 1)):
                end = min(start + w, n)
                idxs = [i for i in range(start, end) if hits[i][0]]
                window = [(hits[i][1], hits[i][2]) for i in idxs]
                if len(window) < min_hits:
                    continue
                near_positions = [ne for ne, _ in window]
                if max_slope > 0.0:
                    # Allow a slightly tilted line: fit it and take the
                    # distance at the middle of the rectangle edge, so the
                    # axis-aligned edge still gets a sensible offset.
                    coords = [float(xs[i]) for i in idxs]
                    fit = self._fit_tilted_line(
                        coords, near_positions, cur_tol, max_slope, anchor
                    )
                    if fit is None:
                        continue
                    agree_idx, near_edge = fit
                    if len(agree_idx) < min_hits:
                        continue
                    centers = [window[j][1] for j in agree_idx]
                    center_fit = self._fit_tilted_line(
                        [coords[j] for j in agree_idx], centers,
                        cur_tol * 2.0, max_slope, anchor,
                    )
                    center = (
                        center_fit[1] if center_fit is not None
                        else float(np.median(centers))
                    )
                    return (float(near_edge), float(center))
                median_near = float(np.median(near_positions))
                agree_idx = [j for j, ne in enumerate(near_positions) if abs(ne - median_near) <= cur_tol]
                if len(agree_idx) >= min_hits:
                    near_edge = float(np.median([near_positions[j] for j in agree_idx]))
                    center = float(np.median([window[j][1] for j in agree_idx]))
                    return (near_edge, center)

        return None

    def _detect_line_v(self, ys, ix, scan_dir, max_scan, grayscale, img_h, img_w,
                       consec_window, dist_tol, lum_thresh, max_slope=0.0):
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
        anchor = float(np.mean(ys)) if len(ys) else 0.0
        for cur_tol in (tol * 0.5, tol):
            for start in range(max(n - w + 1, 1)):
                end = min(start + w, n)
                idxs = [i for i in range(start, end) if hits[i][0]]
                window = [(hits[i][1], hits[i][2]) for i in idxs]
                if len(window) < min_hits:
                    continue
                near_positions = [ne for ne, _ in window]
                if max_slope > 0.0:
                    # Allow a slightly tilted line: fit it and take the
                    # distance at the middle of the rectangle edge, so the
                    # axis-aligned edge still gets a sensible offset.
                    coords = [float(ys[i]) for i in idxs]
                    fit = self._fit_tilted_line(
                        coords, near_positions, cur_tol, max_slope, anchor
                    )
                    if fit is None:
                        continue
                    agree_idx, near_edge = fit
                    if len(agree_idx) < min_hits:
                        continue
                    centers = [window[j][1] for j in agree_idx]
                    center_fit = self._fit_tilted_line(
                        [coords[j] for j in agree_idx], centers,
                        cur_tol * 2.0, max_slope, anchor,
                    )
                    center = (
                        center_fit[1] if center_fit is not None
                        else float(np.median(centers))
                    )
                    return (float(near_edge), float(center))
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
        scale = self._rule_scale(rule, d["resize_base"], img_w, img_h)
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
        scan_offset_px = rule.get("scan_offset_pixels", d["scan_offset_pixels"])

        grayscale = self._grayscale_cache
        img_h, img_w = grayscale.shape

        p0, p1 = shape.points[0], shape.points[1]
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())

        # Fixed offset at resize_base scale (independent of rectangle size)
        scale = self._rule_scale(rule, d["resize_base"], img_w, img_h)
        scan_offset = scan_offset_px * scale
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
            self._clearStaleHoverState()
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
        self._clearStaleHoverState()
        self.storeShapes()
        self.update()

    def resizeEvent(self, event):
        """Handle resize events - resize cursor overlay."""
        super().resizeEvent(event)
        self._cursor_overlay.setGeometry(self.rect())

    # Slack (widget px) for antialiasing and float -> int rounding.
    _PAINT_MARGIN_SLACK = 8.0

    @staticmethod
    def _shapeDrawMargin(shape) -> float:
        """How far outside its own points a shape can put pixels.

        drawVertex centres a handle of side/diameter ``d`` on each vertex,
        so it reaches d / 2 beyond the point, plus half the outline pen.
        ``d`` is point_object_size * 1.5 for point shapes, point_size * 3
        for the MOVE_VERTEX highlight and the moving preview endpoint, and
        point_size otherwise.
        """
        if shape is not None and shape.shape_type == "point":
            handle = float(Shape.point_object_size) * 1.5
        elif shape is not None and (
            shape._highlightIndex is not None or shape._is_line_preview
        ):
            handle = float(Shape.point_size) * 3.0
        else:
            handle = float(Shape.point_size)
        return (
            handle / 2.0
            + float(Shape.PEN_WIDTH) / 2.0
            + Canvas._PAINT_MARGIN_SLACK
        )

    @staticmethod
    def _shapeImageBounds(shape):
        """(min_x, min_y, max_x, max_y) of a shape in image coordinates.

        Equal to shape.boundingRect() for every shape type — makePath draws
        a rect for rectangle/mask, an ellipse around the centre for circle
        and a polyline through the points otherwise — but without building a
        QPainterPath. Returns None when the shape has no points.
        """
        points = shape.points
        if not points:
            return None
        if shape.shape_type == "circle" and len(points) == 2:
            cx, cy = points[0].x(), points[0].y()
            dx, dy = points[1].x() - cx, points[1].y() - cy
            r = (dx * dx + dy * dy) ** 0.5
            return cx - r, cy - r, cx + r, cy + r
        min_x = max_x = points[0].x()
        min_y = max_y = points[0].y()
        for pt in points:
            x, y = pt.x(), pt.y()
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
        return min_x, min_y, max_x, max_y

    def _shapeDeviceRect(self, shape) -> QtCore.QRectF | None:
        """Widget-coordinate bounds of a shape's points, or None if unknown.

        Returns None for shapes whose painted extent is not derivable from
        the points (mask), so callers fall back to a full update.
        """
        if shape is None:
            return None
        points = shape.points
        if not points or shape.shape_type == "mask":
            return None
        if shape.shape_type == "circle" and len(points) == 2:
            cx, cy = points[0].x(), points[0].y()
            dx, dy = points[1].x() - cx, points[1].y() - cy
            r = (dx * dx + dy * dy) ** 0.5
            min_x, max_x, min_y, max_y = cx - r, cx + r, cy - r, cy + r
        else:
            xs = [pt.x() for pt in points]
            ys = [pt.y() for pt in points]
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        offset = self.offsetToCenter()
        s = self.scale
        return QtCore.QRectF(
            QtCore.QPointF((min_x + offset.x()) * s, (min_y + offset.y()) * s),
            QtCore.QPointF((max_x + offset.x()) * s, (max_y + offset.y()) * s),
        )

    # Half-extent (widget px) of the magnet ghost marker: 12 px arms + pen.
    _GHOST_MARKER_RADIUS = 16.0

    def _ghostDeviceRect(self, ghost_pos) -> QtCore.QRectF | None:
        """Widget-coordinate bounds of the dark-pixel-magnet ghost marker."""
        if ghost_pos is None:
            return None
        offset = self.offsetToCenter()
        s = self.scale
        cx = (ghost_pos.x() + offset.x()) * s
        cy = (ghost_pos.y() + offset.y()) * s
        r = self._GHOST_MARKER_RADIUS
        return QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)

    def _repaintEditedShapes(self) -> None:
        """Repaint the shapes a click can have changed, not the whole canvas.

        A press/release in edit mode only alters the selection, the hovered
        shape's highlight and any drag guides. Hiding background shapes is
        the one thing that changes every shape, so that case (and unknown
        bounds) still repaints everything.
        """
        if self.hideBackround or self._hideBackround:
            self.repaint()
            return
        if self._auto_fit_guides or self._snap_line_pos is not None:
            self.repaint()  # guides span the canvas
            return
        shapes = list(self.selectedShapes)
        if self.hShape is not None and self.hShape not in shapes:
            shapes.append(self.hShape)
        if self.prevhShape is not None and self.prevhShape not in shapes:
            shapes.append(self.prevhShape)
        region = self._shapesDeviceRegion(shapes) if shapes else None
        if region is None:
            self.repaint()
        else:
            self.update(region)

    def _shapesDeviceRegion(self, shapes):
        """Region covering the given shapes, one rect each.

        Returns None when any bound is unknown (mask shapes) or the region
        would be empty, so callers fall back to a full repaint.
        """
        region = QtGui.QRegion()
        for shape in shapes:
            rect = self._shapeDeviceRect(shape)
            if rect is None:
                return None
            m = self._shapeDrawMargin(shape)
            region = region + QtGui.QRegion(
                rect.adjusted(-m, -m, m, m).toAlignedRect()
            )
        return None if region.isEmpty() else region

    def _shapesDeviceRect(self, shapes) -> QtCore.QRectF | None:
        """Union of the given shapes' painted bounds in widget coordinates.

        Returns None when any bound is unknown (mask shapes) so callers fall
        back to a full repaint, and for an empty sequence.
        """
        union = QtCore.QRectF()
        for shape in shapes:
            rect = self._shapeDeviceRect(shape)
            if rect is None:
                return None
            m = self._shapeDrawMargin(shape)
            union = union.united(rect.adjusted(-m, -m, m, m))
        return None if union.isNull() else union

    def _vertexMoveDeviceRect(
        self, shape, index, old_point
    ) -> QtCore.QRectF | None:
        """Area a single-vertex move can repaint, in widget coordinates.

        moveVertexBy only rewrites points[index], so for multi-point shapes
        the outline change is confined to the two edges touching that vertex.
        The filled interior changes only between the old and new boundary
        paths, which lie inside the convex hull of the moved vertex (old and
        new) and its two neighbours — so that hull's bounds is a safe
        superset. Returns None when the whole shape can move (rectangle,
        circle) or its extent is unknown (mask), so the caller falls back to
        the full-shape bounds.
        """
        if shape is None or index is None or old_point is None:
            return None
        points = shape.points
        if index >= len(points):
            return None
        if shape.shape_type in ("rectangle", "circle", "mask"):
            return None
        candidates = [old_point, points[index]]
        n = len(points)
        if n > 1:
            if shape.isClosed():
                candidates.append(points[(index - 1) % n])
                candidates.append(points[(index + 1) % n])
            else:
                if index > 0:
                    candidates.append(points[index - 1])
                if index + 1 < n:
                    candidates.append(points[index + 1])
        xs = [pt.x() for pt in candidates]
        ys = [pt.y() for pt in candidates]
        offset = self.offsetToCenter()
        s = self.scale
        return QtCore.QRectF(
            QtCore.QPointF((min(xs) + offset.x()) * s, (min(ys) + offset.y()) * s),
            QtCore.QPointF((max(xs) + offset.x()) * s, (max(ys) + offset.y()) * s),
        )

    def _shapeIntersectsRect(
        self, shape, cull_rect: QtCore.QRectF, cull_region=None
    ) -> bool:
        """Whether the shape can paint inside cull_rect (widget coords).

        Shapes draw at point * scale while the painter adds scale * offset,
        so a point maps to (point + offset) * scale. Bounds come straight
        from the points (no QPainterPath build). Mask shapes paint an image
        whose extent is not derivable from the points, so they are never
        culled.
        """
        points = shape.points
        if not points or shape.shape_type == "mask":
            return True
        margin = self._shapeDrawMargin(shape)
        if shape.shape_type == "circle" and len(points) == 2:
            cx, cy = points[0].x(), points[0].y()
            dx, dy = points[1].x() - cx, points[1].y() - cy
            r = (dx * dx + dy * dy) ** 0.5
            min_x, max_x = cx - r, cx + r
            min_y, max_y = cy - r, cy + r
        else:
            min_x = max_x = points[0].x()
            min_y = max_y = points[0].y()
            for pt in points:
                x, y = pt.x(), pt.y()
                if x < min_x:
                    min_x = x
                elif x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                elif y > max_y:
                    max_y = y
        offset = self.offsetToCenter()
        s = self.scale
        left = (min_x + offset.x()) * s - margin
        top = (min_y + offset.y()) * s - margin
        right = (max_x + offset.x()) * s + margin
        bottom = (max_y + offset.y()) * s + margin
        if (
            right < cull_rect.left()
            or left > cull_rect.right()
            or bottom < cull_rect.top()
            or top > cull_rect.bottom()
        ):
            return False
        if cull_region is not None:
            return cull_region.intersects(
                QtCore.QRectF(left, top, right - left, bottom - top).toAlignedRect()
            )
        return True

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
            self._crosshair.get(
                "polygon" if self._createMode == "polygon3" else self._createMode,
                False,
            )
            and self.drawing()
            and self.prevMovePoint is not None
            and not self.outOfPixmap(self.prevMovePoint)
            and self.createMode not in ["point", "polygon", "polygon3"]
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
        # Qt clips painting to the event region, so shapes that fall entirely
        # outside it contribute no pixels — skipping them is invisible but
        # avoids re-rendering every off-screen shape on each drag frame.
        cull_rect = QtCore.QRectF(a0.rect())
        # When the update was requested as a region (e.g. two selected shapes
        # far apart), a0.rect() is only its bounding box; test against the
        # region itself so the gap between them is not repainted.
        cull_region = a0.region()
        if cull_region.rectCount() < 2:
            cull_region = None
        selected_shapes = []
        for shape in self._shapes_paint_order:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                if shape.selected:
                    selected_shapes.append(shape)
                elif self._shapeIntersectsRect(shape, cull_rect, cull_region):
                    shape.paint(p)
        # Draw selected shapes last so they appear on top
        for shape in selected_shapes:
            if self._shapeIntersectsRect(shape, cull_rect, cull_region):
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            # Don't paint preview line for magic wand or near start point
            if not self._mw_active and not self._near_start_point:
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

        # Draw auto-fit preview guides and count indicator
        if self._auto_fit_guides:
            snap_pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 80))
            snap_pen.setWidth(4)
            snap_pen.setStyle(Qt.DashLine)
            p.setPen(snap_pen)
            for edge_idx, line_pos in self._auto_fit_guides:
                if edge_idx in (Shape.EDGE_TOP, Shape.EDGE_BOTTOM):
                    ly = line_pos * self.scale
                    p.drawLine(QPointF(0, ly), QPointF(self.width(), ly))
                elif edge_idx in (Shape.EDGE_LEFT, Shape.EDGE_RIGHT):
                    lx = line_pos * self.scale
                    p.drawLine(QPointF(lx, 0), QPointF(lx, self.height()))
        if self._auto_fit_dots:
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))
            for dx, dy in self._auto_fit_dots:
                sx = (dx + 0.5) * self.scale
                sy = (dy + 0.5) * self.scale
                p.drawEllipse(QPointF(sx, sy), 3.0, 3.0)
        # Draw auto-fit count label (e.g. "3 lines fit")
        if self._auto_fit_count > 0 and self.prevPoint is not None:
            count = min(self._auto_fit_count, 4)
            label = f"{count} line{'s' if count > 1 else ''} fit"
            cursor_x = self.prevPoint.x() * self.scale
            cursor_y = self.prevPoint.y() * self.scale
            font = p.font()
            font.setPointSize(14)
            font.setBold(count == 4)
            p.setFont(font)
            fm = QtGui.QFontMetrics(font)
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            # Position: left of cursor
            tx = cursor_x - tw - 16
            ty = cursor_y + th // 2 - fm.descent()
            # White background
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 128)))
            p.drawRoundedRect(
                QtCore.QRectF(tx - 4, ty - th + fm.descent() - 2, tw + 8, th + 4),
                4, 4,
            )
            p.setPen(QtGui.QColor(255, 0, 0, 200))
            p.setBrush(Qt.NoBrush)
            p.drawText(QPointF(tx, ty), label)

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

        # Draw dots at every magic wand click position
        # (same green as shape/cursor); latest click is drawn slightly larger
        if self._mw_active and self._mw_clicks:
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, 255)))
            last_index = len(self._mw_clicks) - 1
            for i, click in enumerate(self._mw_clicks):
                cpos = click["pos"]
                dot_x = cpos.x() * self.scale
                dot_y = cpos.y() * self.scale
                r = 5.0 if i == last_index else 3.5
                p.drawEllipse(QPointF(dot_x, dot_y), r, r)

        if not self.current or self.createMode not in [
            "polygon",
            "ai_polygon",
            "ai_mask",
        ]:
            p.end()
            return

        drawing_shape: Shape = self.current.copy()
        if self.createMode in ("polygon", "polygon3"):
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

    def _hoverLabelDeviceRect(self, shape, cursor_pos) -> QtCore.QRectF | None:
        """Widget-coordinate bounds of the hover label drawn by
        _drawHoverLabel, generously padded (it is only used to erase)."""
        if shape is None or cursor_pos is None or not shape.label:
            return None
        cx = cursor_pos.x() * self.scale
        cy = cursor_pos.y() * self.scale
        font = self.font()
        font.setPointSize(20)
        fm = QtGui.QFontMetrics(font)
        text_rect = fm.boundingRect(shape.label)
        padding = 8
        pad = 12.0  # slack for the rounded border and antialiasing
        return QtCore.QRectF(
            cx + 20 - pad,
            cy - 40 - text_rect.height() - pad,
            text_rect.width() + padding * 2 + 2 * pad,
            text_rect.height() + padding * 2 + 2 * pad,
        )

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

    def _magic_wand_select(self, pos):
        """Start interactive magic wand selection or append a click."""
        # Subsequent click while active → union with existing selection
        if self._mw_active and self._mw_image is not None:
            self._magic_wand_add_click(pos)
            return

        image = labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        # img_qt_to_arr returns BGRA (Qt native) — convert to RGB
        image = image[:, :, ::-1].copy()
        h, w = image.shape[:2]
        ix, iy = int(pos.x()), int(pos.y())
        if not (0 <= ix < w and 0 <= iy < h):
            return
        self._mw_image = image
        self._mw_click_pos = pos
        self._mw_tolerances = [10, 10, 10]
        self._mw_base_rgb = image[iy, ix].tolist()
        self._mw_range_w = None
        self._mw_range_h = None
        self._mw_clicks = [{
            "pos": pos,
            "rgb": list(self._mw_base_rgb),
            "tolerances": list(self._mw_tolerances),
            "range_w": None,
            "range_h": None,
            "frozen_mask": None,
        }]
        self._mw_active = True
        # Create floating panel first so _magic_wand_update can set range
        if self._mw_dialog is not None:
            self._mw_dialog.close()
        self._mw_dialog = _MagicWandPanel(self)
        self._mw_dialog.setValues(
            self._mw_base_rgb[0], self._mw_base_rgb[1],
            self._mw_base_rgb[2],
            self._mw_tolerances[0], self._mw_tolerances[1], self._mw_tolerances[2],
        )
        self._mw_dialog.paramsChanged.connect(self._mw_on_param_changed)
        self._mw_dialog.rangeChanged.connect(self._mw_on_range_changed)
        self._mw_dialog.accepted_signal.connect(self._magic_wand_finalize)
        self._mw_dialog.rejected_signal.connect(self._magic_wand_cancel)
        self._magic_wand_update()
        self._mw_dialog.show()
        self._mw_dialog.focusDefault()
        self._position_mw_panel()

    def _magic_wand_add_click(self, pos):
        """Append a click to the current magic wand selection."""
        image = self._mw_image
        if image is None:
            return
        h, w = image.shape[:2]
        ix, iy = int(pos.x()), int(pos.y())
        if not (0 <= ix < w and 0 <= iy < h):
            return
        # Freeze the current working click before starting a new one
        self._freeze_last_click()
        new_rgb = image[iy, ix].tolist()
        new_click = {
            "pos": pos,
            "rgb": list(new_rgb),
            "tolerances": list(self._mw_tolerances),
            "range_w": None,
            "range_h": None,
            "frozen_mask": None,
        }
        self._mw_clicks.append(new_click)
        self._mw_click_pos = pos
        self._mw_base_rgb = list(new_rgb)
        self._mw_range_w = None
        self._mw_range_h = None
        if self._mw_dialog is not None:
            self._mw_dialog.setValues(
                self._mw_base_rgb[0], self._mw_base_rgb[1],
                self._mw_base_rgb[2],
                self._mw_tolerances[0], self._mw_tolerances[1],
                self._mw_tolerances[2],
            )
        self._magic_wand_update()
        self._position_mw_panel()
        self._update_status()

    def _magic_wand_undo_click(self):
        """Remove the most recent click. Cancel if this was the last one."""
        if not self._mw_active:
            return
        if len(self._mw_clicks) <= 1:
            self._magic_wand_cancel()
            return
        self._mw_clicks.pop()
        last = self._mw_clicks[-1]
        # Un-freeze the now-last click so the panel can edit it again
        last["frozen_mask"] = None
        self._mw_click_pos = last["pos"]
        self._mw_base_rgb = list(last["rgb"])
        self._mw_tolerances = list(last["tolerances"])
        self._mw_range_w = last["range_w"]
        self._mw_range_h = last["range_h"]
        if self._mw_dialog is not None:
            self._mw_dialog.setValues(
                self._mw_base_rgb[0], self._mw_base_rgb[1],
                self._mw_base_rgb[2],
                self._mw_tolerances[0], self._mw_tolerances[1],
                self._mw_tolerances[2],
            )
        self._magic_wand_update()
        self._position_mw_panel()
        self._update_status()

    def _compute_click_mask(self, click: dict) -> np.ndarray | None:
        """Flood fill a single click and return its binary mask (uint8 0/1)."""
        image = self._mw_image
        if image is None:
            return None
        h, w = image.shape[:2]
        cpos = click["pos"]
        cix, ciy = int(cpos.x()), int(cpos.y())
        if not (0 <= cix < w and 0 <= ciy < h):
            return None
        c_tol = click["tolerances"]
        img_work = image.copy()
        img_work[ciy, cix] = click["rgb"]
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        lo_diff = (int(c_tol[0]), int(c_tol[1]), int(c_tol[2]))
        up_diff = lo_diff
        flags = (
            8
            | cv2.FLOODFILL_MASK_ONLY
            | cv2.FLOODFILL_FIXED_RANGE
            | (255 << 8)
        )
        cv2.floodFill(
            img_work, flood_mask, (cix, ciy), 0, lo_diff, up_diff, flags,
        )
        filled = flood_mask[1:-1, 1:-1] > 0
        if click["range_w"] is not None or click["range_h"] is not None:
            rw = click["range_w"] if click["range_w"] is not None else w
            rh = click["range_h"] if click["range_h"] is not None else h
            rect_mask = np.zeros((h, w), dtype=bool)
            y0 = max(0, ciy - rh)
            y1 = min(h, ciy + rh + 1)
            x0 = max(0, cix - rw)
            x1 = min(w, cix + rw + 1)
            rect_mask[y0:y1, x0:x1] = True
            filled = filled & rect_mask
        return filled.astype(np.uint8)

    def _freeze_last_click(self) -> None:
        """Commit the working (last) click's mask as a frozen snapshot."""
        if not self._mw_clicks:
            return
        last = self._mw_clicks[-1]
        if last.get("frozen_mask") is not None:
            return
        # Sync panel state back to the working click before freezing
        last["rgb"] = list(self._mw_base_rgb)
        last["tolerances"] = list(self._mw_tolerances)
        last["range_w"] = self._mw_range_w
        last["range_h"] = self._mw_range_h
        mask = self._compute_click_mask(last)
        if mask is None:
            image = self._mw_image
            if image is None:
                return
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
        last["frozen_mask"] = mask

    def _magic_wand_update(self):
        """Update preview: frozen masks + live flood fill for working click.

        All clicks except the most recent hold a frozen mask that is never
        recomputed. Panel parameter changes only affect the working click.
        The resulting contour is the connected component that contains the
        working click; if none contains it, the largest is used.
        """
        if not self._mw_clicks:
            return
        image = self._mw_image
        if image is None:
            return
        h, w = image.shape[:2]

        # Mirror panel state back onto the working click
        last = self._mw_clicks[-1]
        if last.get("frozen_mask") is None:
            last["rgb"] = list(self._mw_base_rgb)
            last["tolerances"] = list(self._mw_tolerances)
            last["range_w"] = self._mw_range_w
            last["range_h"] = self._mw_range_h

        # Accumulate: frozen snapshots + live flood fill for the working click
        accum = np.zeros((h, w), dtype=np.uint8)
        for click in self._mw_clicks:
            fm = click.get("frozen_mask")
            if fm is not None:
                accum |= fm
            else:
                live = self._compute_click_mask(click)
                if live is not None:
                    accum |= live

        if not np.any(accum):
            self._mw_contour = None
            self.current = None
            self.update()
            return

        # Contour extraction with 0.2px sub-pixel precision
        # Upscale mask 5x, extract contour, scale back to get 1/5 = 0.2px steps
        scale = 5
        mask_up = cv2.resize(
            accum, (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        contours_up, _ = cv2.findContours(
            mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours_up:
            self._mw_contour = None
            self.current = None
            self.update()
            return

        # Prefer the component containing the latest click; fall back to largest
        latest_pos = self._mw_clicks[-1]["pos"]
        lx_up = float(latest_pos.x()) * scale
        ly_up = float(latest_pos.y()) * scale
        chosen = None
        for c in contours_up:
            if cv2.pointPolygonTest(c, (lx_up, ly_up), False) >= 0:
                if chosen is None or cv2.contourArea(c) > cv2.contourArea(chosen):
                    chosen = c
        if chosen is None:
            chosen = max(contours_up, key=cv2.contourArea)
        contour_up = chosen
        # Scale contour back to original coordinates (float)
        contour = (contour_up.astype(np.float64) / scale).astype(np.float32)
        self._mw_contour = contour

        # Compute bounding box half-widths from latest click
        ix_last = int(latest_pos.x())
        iy_last = int(latest_pos.y())
        bx, by, bw, bh = cv2.boundingRect(contour)
        half_w = max(abs(bx - ix_last), abs(bx + bw - ix_last))
        half_h = max(abs(by - iy_last), abs(by + bh - iy_last))
        if self._mw_dialog is not None:
            self._mw_dialog.setRange(half_w, half_h)

        # Preview polygon (vertex-free)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, peri * 0.005, True)
        points = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(points) < 3:
            self._mw_contour = None
            self.current = None
            self.update()
            return

        shape = Shape(shape_type="polygon")
        shape._mw_preview = True
        shape._is_creating = True
        shape.fill = True
        # Use line color with fill opacity (same as polygon close preview)
        r, g, b, _ = shape.line_color.getRgb()
        fill_alpha = Shape.fill_color.alpha()
        shape.fill_color = QtGui.QColor(r, g, b, fill_alpha)
        for x, y in points:
            shape.addPoint(QPointF(x, y))
        shape.close()
        self.current = shape
        self.update()

    def _mw_on_param_changed(self):
        """Called when floating panel RGB/tolerance spinbox values change."""
        if self._mw_dialog is None:
            return
        self._mw_base_rgb = self._mw_dialog.rgb()
        self._mw_tolerances = self._mw_dialog.tolerances()
        # RGB/tolerance change → no range constraint, show natural extent
        self._mw_range_w = None
        self._mw_range_h = None
        self._magic_wand_update()
        self._update_status()

    def _mw_on_range_changed(self, rw, rh):
        """Called when the user manually changes the W/H range spinboxes."""
        self._mw_range_w = rw
        self._mw_range_h = rh
        self._magic_wand_update()
        self._update_status()

    def _position_mw_panel(self):
        """Position the magic wand dialog slightly above the click point."""
        if self._mw_dialog is None or self._mw_click_pos is None:
            return
        offset = self.offsetToCenter()
        cx = (self._mw_click_pos.x() + offset.x()) * self.scale
        cy = (self._mw_click_pos.y() + offset.y()) * self.scale
        panel_w = self._mw_dialog.width()
        panel_h = self._mw_dialog.height()
        local_x = int(cx - panel_w / 2)
        local_y = int(cy - panel_h - 80)
        local_x = max(0, min(local_x, self.width() - panel_w))
        local_y = max(0, min(local_y, self.height() - panel_h))
        global_pos = self.mapToGlobal(QPoint(local_x, local_y))
        self._mw_dialog.move(global_pos)

    def _close_mw_panel(self):
        """Close and destroy the magic wand panel."""
        if self._mw_dialog is not None:
            self._mw_dialog.close()
            self._mw_dialog.deleteLater()
            self._mw_dialog = None

    def _magic_wand_cancel(self):
        """Cancel magic wand selection."""
        self._mw_active = False
        self._mw_contour = None
        self._mw_image = None
        self._mw_clicks = []
        self._close_mw_panel()
        self.current = None
        self.drawingPolygon.emit(False)
        self._unhide_os_cursor()
        self.restoreCursor()
        self.update()
        self._update_status()

    def _magic_wand_finalize(self):
        """Show vertex count dialog and finalize magic wand polygon."""
        if self._mw_contour is None:
            return
        peri = cv2.arcLength(self._mw_contour, True)
        # Recommended: approxPolyDP with moderate epsilon
        approx = cv2.approxPolyDP(self._mw_contour, peri * 0.005, True)
        recommended = len(approx)

        count = self._show_vertex_count_dialog(recommended)
        if count is None:
            return  # cancelled — stay in preview

        points = self._approx_contour_to_n_vertices(self._mw_contour, count)
        if len(points) < 3:
            return

        shape = Shape(shape_type="polygon")
        shape._is_creating = True
        for x, y in points:
            shape.addPoint(QPointF(x, y))
        self.current = shape
        self._mw_active = False
        self._mw_contour = None
        self._mw_image = None
        self._mw_clicks = []
        self._close_mw_panel()
        self.finalise()

    def _update_vertex_preview(self, count):
        """Update the polygon preview with visible vertices for given count."""
        if self._mw_contour is None:
            return
        points = self._approx_contour_to_n_vertices(self._mw_contour, count)
        if len(points) < 3:
            return
        shape = Shape(shape_type="polygon")
        shape._is_creating = True
        shape._mw_preview = False  # show vertices
        shape.fill = True
        r, g, b, _ = shape.line_color.getRgb()
        fill_alpha = Shape.fill_color.alpha()
        shape.fill_color = QtGui.QColor(r, g, b, fill_alpha)
        for x, y in points:
            shape.addPoint(QPointF(x, y))
        shape.close()
        self.current = shape
        self.update()

    def _show_vertex_count_dialog(self, recommended):
        """Show a dialog to input vertex count with recommended value."""
        # Show initial vertex preview
        self._update_vertex_preview(recommended)

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("頂点数")
        dialog.setWindowFlags(
            dialog.windowFlags() | Qt.WindowStaysOnTopHint
        )
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(
            f"頂点数を指定 (推奨: {recommended}):"
        )
        layout.addWidget(label)
        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(3, 9999)
        spinbox.setSingleStep(5)
        spinbox.setValue(recommended)
        spinbox.valueChanged.connect(self._update_vertex_preview)
        layout.addWidget(spinbox)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return spinbox.value()
        return None

    def _approx_contour_to_n_vertices(self, contour, target_n):
        """Binary search epsilon to approximate contour to target vertex count."""
        peri = cv2.arcLength(contour, True)
        lo, hi = 0.0, peri * 0.5
        best_pts = cv2.approxPolyDP(contour, peri * 0.02, True)
        for _ in range(50):
            mid = (lo + hi) / 2
            approx = cv2.approxPolyDP(contour, mid, True)
            n = len(approx)
            best_pts = approx
            if n == target_n:
                break
            elif n > target_n:
                lo = mid
            else:
                hi = mid
        return [(float(p[0][0]), float(p[0][1])) for p in best_pts]

    def finalise(self):
        assert self.current
        # A near-start highlight is transient drawing state.  With queued
        # paints it deliberately survives the last mouse move, so clear it
        # before the shape becomes part of the document.
        self.current.highlightClear()
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
            # Magic wand interactive mode — keys handled by dialog
            if self._mw_active and self.current:
                return
            if key == Qt.Key_Escape:
                self.cancelDrawing()
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
        # Keep the cached paint/hover orders coherent with self.shapes —
        # otherwise the popped shape lives on as a ghost that is painted
        # and hoverable but crashes on interaction (shapes.index ValueError,
        # e.g. create point -> cancel the label dialog -> drag the ghost).
        self.sortShapesByArea()
        self._clearStaleHoverState()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "polygon3", "linestrip"]:
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
        # The image array / hash / grayscale caches are only needed by the
        # magnets, the pixel readout and the AI session. Building them here
        # cost ~1 s per 4K image on every file switch even when unused, so
        # they are built on first access instead (see _ensureImageCaches).
        self._img_arr_ready = False
        self._grayscale_ready = False
        self._pixmap_hash = None
        self._img_arr_cache = None
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
        self.hEdgeMidpoint = None
        self._clearStaleHoverState()  # also drops stale prevh* references
        self.update()

    def sortShapesByArea(self):
        """Rebuild cached sort orders for painting and hover detection."""
        # boundingRect() builds a QPainterPath per call; the two sort keys
        # used to invoke it 4x per shape. Compute each area exactly once,
        # straight from the points (identical values, no path build).
        areas: dict[int, float] = {}
        for s in self.shapes:
            bounds = self._shapeImageBounds(s)
            if bounds is None:
                areas[id(s)] = 0.0
            else:
                min_x, min_y, max_x, max_y = bounds
                areas[id(s)] = (max_x - min_x) * (max_y - min_y)
        # Paint order: largest first, points on top (drawn last)
        self._shapes_paint_order = sorted(
            self.shapes,
            key=lambda s: (
                1 if s.shape_type == "point" else 0,
                -areas[id(s)],
            ),
        )
        # Hover order: selected first, then smallest first, points first
        self._shapes_hover_order = sorted(
            self.shapes,
            key=lambda s: (
                0 if s.selected else 1,
                0 if s.shape_type == "point" else 1,
                areas[id(s)],
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
        self._close_mw_panel()
        self._mw_active = False
        self._mw_contour = None
        self._mw_image = None
        self.restoreCursor()
        self.pixmap = QtGui.QPixmap()
        self._pixmap_hash = None
        self._img_arr_cache: np.ndarray | None = None
        self._grayscale_cache: np.ndarray | None = None
        self._img_arr_ready = True  # empty pixmap: nothing to build
        self._grayscale_ready = True
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
