"""Changing the selection must repaint the shapes that lost it.

Emitting selectionChanged on its own left the previous selection drawn with
its fill: the canvas only invalidated the newly selected shape, so the old
colour stayed on screen and the highlights appeared to pile up.
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
    widget.resize(600, 500)
    pixmap = QtGui.QPixmap(600, 500)
    pixmap.fill(QtGui.QColor(225, 225, 225))
    widget.loadPixmap(pixmap)
    widget.setEditing(True)
    # stand in for MainWindow.shapeSelectionChanged
    def apply_selection(shapes):
        for shape in widget.selectedShapes:
            shape.selected = False
        widget.selectedShapes = list(shapes)
        for shape in widget.selectedShapes:
            shape.selected = True
    widget.selectionChanged.connect(apply_selection)
    return widget


def _square(cx, cy, label):
    shape = Shape(shape_type="polygon", label=label)
    for dx, dy in ((-45, -45), (45, -45), (45, 45), (-45, 45)):
        shape.addPoint(QtCore.QPointF(cx + dx, cy + dy))
    shape.close()
    return shape


def _record(canvas, action):
    region = QtGui.QRegion()
    full = []
    original_update, original_repaint = canvas.update, canvas.repaint

    def wrap(original):
        def recorder(*args):
            nonlocal region
            if not args:
                full.append(True)
            elif isinstance(args[0], QtGui.QRegion):
                region = region.united(args[0])
            else:
                region = region.united(
                    QtGui.QRegion(QtCore.QRectF(args[0]).toAlignedRect())
                )
            return original(*args)

        return recorder

    canvas.update, canvas.repaint = wrap(original_update), wrap(original_repaint)
    try:
        action()
    finally:
        canvas.update, canvas.repaint = original_update, original_repaint
    return region, bool(full)


def _shape_device_points(canvas, shape):
    offset = canvas.offsetToCenter()
    return [
        QtCore.QPoint(
            int((p.x() + offset.x()) * canvas.scale),
            int((p.y() + offset.y()) * canvas.scale),
        )
        for p in shape.points
    ]


def test_deselected_shape_is_repainted(canvas):
    far_left = _square(80, 80, "a")
    far_right = _square(500, 400, "b")
    canvas.loadShapes([far_left, far_right])

    canvas.selectShapes([far_left])
    assert far_left.selected

    region, full = _record(canvas, lambda: canvas.selectShapes([far_right]))
    assert not far_left.selected and far_right.selected
    for point in _shape_device_points(canvas, far_left):
        assert full or region.contains(point), (
            f"the deselected shape at {point} was not repainted: "
            f"{list(region.rects())}"
        )
    for point in _shape_device_points(canvas, far_right):
        assert full or region.contains(point)


def test_click_selection_repaints_the_previous_selection(canvas):
    first = _square(80, 80, "a")
    second = _square(480, 380, "b")
    canvas.loadShapes([first, second])

    def click(shape):
        centre = QtCore.QPointF(
            sum(p.x() for p in shape.points) / len(shape.points),
            sum(p.y() for p in shape.points) / len(shape.points),
        )
        offset = canvas.offsetToCenter()
        pos = QtCore.QPointF(
            (centre.x() + offset.x()) * canvas.scale,
            (centre.y() + offset.y()) * canvas.scale,
        )
        canvas.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton,
            Qt.NoModifier))
        canvas.mousePressEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseReleaseEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    click(first)
    assert canvas.selectedShapes == [first]

    region, full = _record(canvas, lambda: click(second))
    assert canvas.selectedShapes == [second]
    assert not first.selected
    for point in _shape_device_points(canvas, first):
        assert full or region.contains(point), (
            f"the shape that lost the selection was not repainted at {point}"
        )


def test_selection_flag_drift_is_repainted(canvas):
    shape = _square(200, 200, "a")
    other = _square(420, 300, "b")
    canvas.loadShapes([shape, other])
    canvas.selectShapes([shape])

    # same selection list, but the flag drifted: the repaired flag changes
    # what is drawn, so it has to be invalidated too
    shape.selected = False
    region, full = _record(canvas, lambda: canvas.selectShapes([shape]))
    assert shape.selected
    for point in _shape_device_points(canvas, shape):
        assert full or region.contains(point)


def test_hide_background_still_repaints_everything(canvas):
    a = _square(100, 100, "a")
    b = _square(400, 350, "b")
    canvas.loadShapes([a, b])
    canvas.selectShapes([a])
    canvas.hideBackroundShapes(True)

    _, full = _record(canvas, lambda: canvas.selectShapes([b]))
    assert full, "hiding the background changes every shape"
    canvas.hideBackroundShapes(False)


def test_selection_signal_is_emitted_once(canvas):
    a = _square(100, 100, "a")
    b = _square(400, 350, "b")
    canvas.loadShapes([a, b])
    emissions = []
    canvas.selectionChanged.connect(lambda shapes: emissions.append(list(shapes)))
    canvas.selectShapes([a])
    canvas.selectShapes([b])
    canvas.deSelectShape()
    assert len(emissions) == 3
    assert emissions[-1] == []
    assert not a.selected and not b.selected
