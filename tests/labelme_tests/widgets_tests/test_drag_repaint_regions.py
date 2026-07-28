"""Drags repaint regions, not the whole canvas. Everything they draw on top of
the shapes - the edge guide, the snap guide, the auto-fit dots and counter,
the magnet ghost - has to be inside the invalidated region, at both ends of
the gesture.

The magnet detectors are stubbed: what is under test is the repainting, not
the detection (which has its own tests).
"""

import numpy as np
import pytest
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from labelme.shape import Shape
from labelme.widgets.canvas import Canvas


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def canvas(app):
    widget = Canvas(
        epsilon=10.0,
        double_click="close",
        num_backups=10,
        crosshair={
            k: False
            for k in [
                "polygon", "rectangle", "circle", "line", "point",
                "linestrip", "ai_polygon", "ai_mask", "magic_wand",
            ]
        },
    )
    widget.resize(500, 420)
    pixmap = QtGui.QPixmap(500, 420)
    pixmap.fill(QtGui.QColor(215, 215, 215))
    widget.loadPixmap(pixmap)
    widget.setEditing(True)
    widget.show()
    yield widget
    widget.close()


def _polygon(cx, cy, label="p", r=28, n=8):
    shape = Shape(shape_type="polygon", label=label)
    for i in range(n):
        a = 2 * np.pi * i / n
        shape.addPoint(QtCore.QPointF(cx + r * np.cos(a), cy + r * np.sin(a)))
    shape.close()
    return shape


def _rectangle(x0, y0, x1, y1, label="notes"):
    shape = Shape(shape_type="rectangle", label=label)
    shape.addPoint(QtCore.QPointF(x0, y0))
    shape.addPoint(QtCore.QPointF(x1, y1))
    shape.close()
    return shape


def _scene(canvas):
    """Polygons well clear of the rectangle, so its edge handles win the hover."""
    shapes = [_polygon(50 + 60 * i, 350, f"l{i % 3}") for i in range(7)]
    shapes += [_polygon(440, 60 + 60 * i, f"m{i % 3}") for i in range(4)]
    rect = _rectangle(150, 150, 330, 250)
    shapes.append(rect)
    canvas.loadShapes(shapes)
    canvas.repaint()
    return shapes, rect


def _grab(canvas):
    image = canvas.grab().toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine())
    return arr[:, : image.width() * 4].copy().reshape(
        image.height(), image.width(), 4
    )


class _Recorder:
    """Collects the regions a block of code asked Qt to repaint."""

    def __init__(self, canvas):
        self.canvas = canvas

    def __enter__(self):
        self.mask = np.zeros((self.canvas.height(), self.canvas.width()), bool)
        self.full = False
        self.requests = 0
        self._update, self._repaint = self.canvas.update, self.canvas.repaint

        def record(*args):
            self.requests += 1
            if not args:
                self.full = True
                return
            arg = args[0]
            if isinstance(arg, QtGui.QRegion):
                rects = list(arg.rects())
            elif isinstance(arg, (QtCore.QRect, QtCore.QRectF)):
                rects = [QtCore.QRectF(arg).toAlignedRect()]
            else:
                self.full = True
                return
            for rect in rects:
                x0, y0 = max(0, rect.left()), max(0, rect.top())
                x1 = min(self.mask.shape[1], rect.right() + 1)
                y1 = min(self.mask.shape[0], rect.bottom() + 1)
                if x1 > x0 and y1 > y0:
                    self.mask[y0:y1, x0:x1] = True

        def wrap(original):
            def wrapper(*args):
                record(*args)
                return original(*args)
            return wrapper

        self.canvas.update = wrap(self._update)
        self.canvas.repaint = wrap(self._repaint)
        return self

    def __exit__(self, *exc):
        self.canvas.update, self.canvas.repaint = self._update, self._repaint

    def uncovered(self, before, after):
        diff = np.any(before != after, axis=2)
        if self.full:
            return 0
        return int((diff & ~self.mask).sum())


def _pos(canvas, x, y):
    offset = canvas.offsetToCenter()
    return QtCore.QPointF(
        (x + offset.x()) * canvas.scale, (y + offset.y()) * canvas.scale
    )


def _send(canvas, kind, x, y, button=Qt.NoButton, buttons=Qt.NoButton):
    event = QtGui.QMouseEvent(
        kind, _pos(canvas, x, y), button, buttons, Qt.NoModifier
    )
    {
        QtCore.QEvent.MouseMove: canvas.mouseMoveEvent,
        QtCore.QEvent.MouseButtonPress: canvas.mousePressEvent,
        QtCore.QEvent.MouseButtonRelease: canvas.mouseReleaseEvent,
    }[kind](event)


def _covered(canvas, action):
    """Measure exactly one interaction, with the setup already painted."""
    canvas.repaint()
    QtWidgets.QApplication.processEvents()
    before = _grab(canvas)
    with _Recorder(canvas) as recorder:
        action()
        QtWidgets.QApplication.processEvents()
    after = _grab(canvas)
    return recorder, recorder.uncovered(before, after)


def test_edge_drag_with_snap_guide_repaints_its_bands(canvas):
    """The red dashed snap guide spans the canvas; erasing it must not need a
    full repaint, and must not leave the line behind."""
    _shapes, rect = _scene(canvas)
    mid_x = (rect.points[0].x() + rect.points[1].x()) / 2
    top_y = min(rect.points[0].y(), rect.points[1].y())

    _send(canvas, QtCore.QEvent.MouseMove, mid_x, top_y)
    assert canvas.hEdgeMidpoint is not None
    assert canvas.hShape is rect
    _send(canvas, QtCore.QEvent.MouseButtonPress, mid_x, top_y,
          Qt.LeftButton, Qt.LeftButton)
    assert canvas._edge_midpoint_dragging
    assert canvas._dragging_edge_index == canvas.hEdgeMidpoint
    QtWidgets.QApplication.processEvents()

    # stub the detector: the guide is what is under test, not the detection
    original_move_edge = canvas.boundedMoveEdge

    def move_edge_with_snap(pos):
        result = original_move_edge(pos)
        canvas._snap_active = True
        canvas._snap_line_pos = 90.0
        return result

    canvas.boundedMoveEdge = move_edge_with_snap

    for step, target in enumerate((130.0, 120.0, 110.0)):
        recorder, uncovered = _covered(
            canvas,
            lambda target=target: _send(
                canvas, QtCore.QEvent.MouseMove, mid_x, target,
                buttons=Qt.LeftButton),
        )
        assert canvas._snap_active, "the stub must keep the guide showing"
        assert uncovered == 0, f"frame {step}: {uncovered} px outside the region"
        assert not recorder.full, f"frame {step} repainted everything"

    recorder, uncovered = _covered(
        canvas,
        lambda: _send(canvas, QtCore.QEvent.MouseButtonRelease, mid_x, 110.0,
                      Qt.LeftButton),
    )
    assert uncovered == 0, f"release left {uncovered} px stale"
    assert not recorder.full, "release repainted everything"
    assert not canvas._snap_active


def test_shape_move_with_auto_fit_markers_repaints_them(canvas):
    """The auto-fit dots and the "N lines fit" label follow the drag."""
    shapes, _rect = _scene(canvas)
    mover = shapes[7]
    centre = QtCore.QPointF(
        sum(p.x() for p in mover.points) / len(mover.points),
        sum(p.y() for p in mover.points) / len(mover.points),
    )
    canvas.selectShapes([mover])
    QtWidgets.QApplication.processEvents()
    _send(canvas, QtCore.QEvent.MouseMove, centre.x(), centre.y())
    assert canvas.hShape is mover
    _send(canvas, QtCore.QEvent.MouseButtonPress, centre.x(), centre.y(),
          Qt.LeftButton, Qt.LeftButton)
    # one move first: the frame under test must not carry the press repaint
    _send(canvas, QtCore.QEvent.MouseMove, centre.x() + 4, centre.y(),
          buttons=Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    original_move_shapes = canvas.boundedMoveShapes
    offset = {"value": 0}

    def move_shapes_with_markers(shapes, pos):
        result = original_move_shapes(shapes, pos)
        # text-only auto fit: no guide lines, but dots and a counter
        canvas._auto_fit_guides = []
        canvas._auto_fit_dots = [
            (60 + offset["value"] + 30 * i, 300 + offset["value"])
            for i in range(3)
        ]
        canvas._auto_fit_count = 3
        return result

    canvas.boundedMoveShapes = move_shapes_with_markers

    for step, delta in enumerate((12, 26, 44)):
        offset["value"] = delta
        recorder, uncovered = _covered(
            canvas,
            lambda delta=delta: _send(
                canvas, QtCore.QEvent.MouseMove,
                centre.x() + delta, centre.y() + delta // 2,
                buttons=Qt.LeftButton),
        )
        assert uncovered == 0, f"frame {step}: {uncovered} px outside the region"
        assert not recorder.full, f"frame {step} repainted everything"

    recorder, uncovered = _covered(
        canvas,
        lambda: _send(canvas, QtCore.QEvent.MouseButtonRelease,
                      centre.x() + 44, centre.y() + 22, Qt.LeftButton),
    )
    assert uncovered == 0, f"release left {uncovered} px stale"
    assert not canvas._auto_fit_dots and canvas._auto_fit_count == 0


def test_drag_bounds_cache_follows_a_changing_draw_margin(canvas):
    """The cached bounds carry no margin, so a highlight appearing mid-drag
    still gets its bigger handle repainted."""
    shapes, _rect = _scene(canvas)
    neighbour = shapes[2]
    target = shapes[7]
    vertex = min(target.points, key=lambda p: p.y())
    _send(canvas, QtCore.QEvent.MouseMove, vertex.x(), vertex.y())
    _send(canvas, QtCore.QEvent.MouseButtonPress, vertex.x(), vertex.y(),
          Qt.LeftButton, Qt.LeftButton)
    assert canvas._vertex_dragging
    _send(canvas, QtCore.QEvent.MouseMove, vertex.x() + 5, vertex.y(),
          buttons=Qt.LeftButton)
    canvas.repaint()
    QtWidgets.QApplication.processEvents()
    assert neighbour in canvas._drag_bounds_cache

    cull = QtCore.QRectF(canvas.rect())
    plain = canvas._shapeIntersectsRect(neighbour, cull)
    neighbour.highlightVertex(0, neighbour.MOVE_VERTEX)
    highlighted = canvas._shapeIntersectsRect(neighbour, cull)
    canvas._drag_bounds_cache = None
    uncached = canvas._shapeIntersectsRect(neighbour, cull)
    assert plain == highlighted == uncached

    _send(canvas, QtCore.QEvent.MouseButtonRelease, vertex.x() + 5, vertex.y(),
          Qt.LeftButton)
