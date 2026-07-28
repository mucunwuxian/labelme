"""Alt+Shift-clicking a vertex removes it. That is a complete edit on its own:

  * the vertex is gone and the shape is repainted where it used to draw
  * no hover index survives, so the next mouse move must not drag whatever
    vertex slid into the removed index
  * a shape that cannot lose a vertex is left completely untouched
"""

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


def _canvas(app):
    canvas = Canvas(
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
    canvas.resize(400, 400)
    pixmap = QtGui.QPixmap(400, 400)
    pixmap.fill(QtGui.QColor(220, 220, 220))
    canvas.loadPixmap(pixmap)
    canvas.setEditing(True)
    return canvas


def _polygon(points, label="p"):
    shape = Shape(shape_type="polygon", label=label)
    for x, y in points:
        shape.addPoint(QtCore.QPointF(x, y))
    shape.close()
    return shape


def _press(canvas, point, modifiers=Qt.NoModifier):
    offset = canvas.offsetToCenter()
    pos = QtCore.QPointF(
        (point.x() + offset.x()) * canvas.scale,
        (point.y() + offset.y()) * canvas.scale,
    )
    canvas.mousePressEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton,
            modifiers,
        )
    )


def _move(canvas, point, buttons=Qt.NoButton):
    offset = canvas.offsetToCenter()
    pos = QtCore.QPointF(
        (point.x() + offset.x()) * canvas.scale,
        (point.y() + offset.y()) * canvas.scale,
    )
    canvas.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, pos, Qt.NoButton, buttons, Qt.NoModifier
        )
    )


def _release(canvas, point):
    offset = canvas.offsetToCenter()
    pos = QtCore.QPointF(
        (point.x() + offset.x()) * canvas.scale,
        (point.y() + offset.y()) * canvas.scale,
    )
    canvas.mouseReleaseEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier,
        )
    )


def _record_updates(canvas, action):
    """Run action and return (union of invalidated regions, saw_full_repaint)."""
    region = QtGui.QRegion()
    full = []
    original_update = canvas.update
    original_repaint = canvas.repaint

    def wrap(original):
        def recorder(*args):
            nonlocal region
            if not args:
                full.append(True)
            elif isinstance(args[0], QtGui.QRegion):
                region = region.united(args[0])
            else:
                region = region.united(QtGui.QRegion(QtCore.QRectF(
                    args[0]).toAlignedRect()))
            return original(*args)

        return recorder

    canvas.update = wrap(original_update)
    canvas.repaint = wrap(original_repaint)
    try:
        action()
    finally:
        canvas.update = original_update
        canvas.repaint = original_repaint
    return region, bool(full)


def _handle_outline(canvas, image_point, scale=1.0):
    """Widget-coordinate points around a vertex handle's edge."""
    offset = canvas.offsetToCenter()
    cx = (image_point.x() + offset.x()) * canvas.scale
    cy = (image_point.y() + offset.y()) * canvas.scale
    radius = Shape.point_size * scale / 2.0
    return [
        QtCore.QPoint(int(cx), int(cy)),
        QtCore.QPoint(int(cx - radius), int(cy)),
        QtCore.QPoint(int(cx + radius), int(cy)),
        QtCore.QPoint(int(cx), int(cy - radius)),
        QtCore.QPoint(int(cx), int(cy + radius)),
    ]


def test_removal_is_one_undo_transaction(app):
    canvas = _canvas(app)
    shape = _polygon([(80, 80), (200, 80), (260, 160), (200, 240), (80, 240)])
    canvas.loadShapes([shape])
    original_points = [(p.x(), p.y()) for p in shape.points]
    backups_before = len(canvas.shapesBackups)
    moved = []
    canvas.shapeMoved.connect(lambda: moved.append(1))

    victim = QtCore.QPointF(shape.points[2])
    _move(canvas, victim)
    _press(canvas, victim, Qt.AltModifier | Qt.ShiftModifier)

    assert len(canvas.shapesBackups) == backups_before + 1
    assert len(moved) == 1

    # the release that follows the click must not add a second one
    _release(canvas, victim)
    assert len(canvas.shapesBackups) == backups_before + 1
    assert len(moved) == 1

    canvas.restoreShape()
    restored = canvas.shapes[0]
    assert [(p.x(), p.y()) for p in restored.points] == original_points


def test_removing_a_middle_vertex_does_not_drag_the_next_one(app):
    canvas = _canvas(app)
    shape = _polygon([(80, 80), (200, 80), (260, 160), (200, 240), (80, 240)])
    canvas.loadShapes([shape])

    victim = QtCore.QPointF(shape.points[2])       # the middle spike
    _move(canvas, victim)
    assert canvas.hVertex == 2

    _press(canvas, victim, Qt.AltModifier | Qt.ShiftModifier)
    assert len(shape.points) == 4
    assert all(p != victim for p in shape.points)

    # nothing may still be latched
    assert canvas.hVertex is None
    assert canvas.prevhVertex is None
    assert canvas.hShape is None
    assert canvas.prevhShape is None

    # dragging away must not move the vertex that took index 2
    before = [(p.x(), p.y()) for p in shape.points]
    _move(canvas, QtCore.QPointF(victim.x() + 60, victim.y() + 40),
          buttons=Qt.LeftButton)
    _release(canvas, QtCore.QPointF(victim.x() + 60, victim.y() + 40))
    assert [(p.x(), p.y()) for p in shape.points] == before


def test_removal_repaints_where_the_shape_used_to_draw(app):
    canvas = _canvas(app)
    shape = _polygon([(80, 80), (200, 80), (200, 160), (80, 160), (40, 330)])
    canvas.loadShapes([shape])
    spike = QtCore.QPointF(shape.points[4])
    _move(canvas, spike)

    region, full = _record_updates(canvas, lambda: _press(
        canvas, spike, Qt.AltModifier | Qt.ShiftModifier))

    assert len(shape.points) == 4
    assert full or not region.isEmpty(), "the removal invalidated nothing"
    # the whole handle, not just its centre
    for point in _handle_outline(canvas, spike):
        assert full or region.contains(point), (
            f"the removed vertex's handle at {point} was not repainted: "
            f"{list(region.rects())}"
        )


@pytest.mark.parametrize(
    "shape_maker",
    [
        lambda: _polygon([(10, 10), (60, 10), (35, 60)]),          # 3 points
        lambda: Shape(shape_type="rectangle", label="r"),
    ],
)
def test_a_shape_that_cannot_lose_a_vertex_is_untouched(app, shape_maker):
    canvas = _canvas(app)
    shape = shape_maker()
    if shape.shape_type == "rectangle":
        shape.addPoint(QtCore.QPointF(20, 20))
        shape.addPoint(QtCore.QPointF(120, 90))
        shape.close()
    canvas.loadShapes([shape])
    canvas.prevhShape = shape
    canvas.prevhVertex = 1

    before_points = [(p.x(), p.y()) for p in shape.points]
    before_modified = shape.modified_at
    backups = len(canvas.shapesBackups)
    moved = []
    canvas.shapeMoved.connect(lambda: moved.append(1))

    assert canvas.removeSelectedPoint() is False
    assert [(p.x(), p.y()) for p in shape.points] == before_points
    assert shape.modified_at == before_modified
    assert len(canvas.shapesBackups) == backups
    assert not moved


def test_adding_a_point_covers_the_new_handle(app):
    canvas = _canvas(app)
    shape = _polygon([(80, 80), (200, 80), (200, 160), (80, 160)])
    canvas.loadShapes([shape])
    canvas.prevhShape = shape
    canvas.prevhEdge = 1
    canvas.prevMovePoint = QtCore.QPointF(320, 120)

    region, full = _record_updates(canvas, canvas.addPointToEdge)

    assert len(shape.points) == 5
    assert canvas.hVertex == 1
    # the inserted vertex is highlighted, so its handle is 3x the usual size
    for point in _handle_outline(
        canvas, QtCore.QPointF(320, 120), scale=3.0
    ):
        assert full or region.contains(point), (
            f"the inserted vertex's handle at {point} was not repainted: "
            f"{list(region.rects())}"
        )
