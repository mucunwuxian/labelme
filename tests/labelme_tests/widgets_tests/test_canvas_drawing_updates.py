import pytest
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from labelme.shape import Shape
from labelme.widgets.canvas import Canvas


class _RecordingCanvas(Canvas):
    def __init__(self):
        self.paint_records = []
        super().__init__()

    def paintEvent(self, event):
        highlight = None if self.current is None else self.current._highlightIndex
        self.paint_records.append((highlight, QtGui.QRegion(event.region())))
        super().paintEvent(event)


def _make_drawing_canvas(canvas_type=Canvas, create_mode="polygon"):
    canvas = canvas_type()
    canvas.resize(300, 300)
    pixmap = QtGui.QPixmap(300, 300)
    pixmap.fill(QtGui.QColor(32, 32, 32))
    canvas.loadPixmap(pixmap)
    canvas.setEditing(False)
    canvas.createMode = create_mode

    canvas.current = Shape(label="new", shape_type=create_mode)
    canvas.current._is_creating = True
    canvas.current.addPoint(QtCore.QPointF(60, 60))
    if create_mode == "polygon":
        canvas.current.addPoint(QtCore.QPointF(120, 60))
        canvas.current.addPoint(QtCore.QPointF(120, 120))
    return canvas


def _move(canvas, x, y):
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(x, y),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    canvas.mouseMoveEvent(event)


@pytest.mark.gui
@pytest.mark.parametrize("create_mode", ["polygon", "rectangle"])
def test_drawing_mouse_moves_are_coalesced(qtbot, create_mode):
    canvas = _make_drawing_canvas(_RecordingCanvas, create_mode)
    qtbot.addWidget(canvas)
    canvas.show()
    QtWidgets.QApplication.processEvents()

    # Prime a previous preview position, then leave the event queue empty.
    _move(canvas, 150, 140)
    QtWidgets.QApplication.processEvents()
    canvas.paint_records.clear()

    for i in range(12):
        _move(canvas, 180 + i, 140 + (i % 3))

    # update() is queued, unlike repaint(): no paint occurs in the mouse
    # handler and all pending preview regions can be merged into one frame.
    assert canvas.paint_records == []
    QtCore.QCoreApplication.sendPostedEvents(canvas, QtCore.QEvent.UpdateRequest)
    QtWidgets.QApplication.processEvents()
    assert len(canvas.paint_records) == 1

    if create_mode == "polygon":
        _, region = canvas.paint_records[0]
        assert region.contains(QtCore.QPoint(150, 140))
        assert region.contains(QtCore.QPoint(191, 142))


@pytest.mark.gui
def test_near_start_highlight_tracks_latest_move_and_does_not_persist(qtbot):
    canvas = _make_drawing_canvas(_RecordingCanvas)
    canvas.current.points = [
        QtCore.QPointF(40, 40),
        QtCore.QPointF(180, 40),
        QtCore.QPointF(180, 150),
    ]
    canvas.current.point_labels = [1, 1, 1]
    qtbot.addWidget(canvas)
    canvas.show()
    QtWidgets.QApplication.processEvents()
    canvas.paint_records.clear()

    _move(canvas, 42, 41)
    assert canvas._near_start_point is True
    assert canvas.current._highlightIndex == 0
    assert canvas.current._highlightMode == Shape.NEAR_VERTEX
    assert canvas.line[1] == canvas.current[0]
    QtWidgets.QApplication.processEvents()
    assert len(canvas.paint_records) == 1
    highlight, region = canvas.paint_records[-1]
    assert highlight == 0
    assert region.contains(QtCore.QPoint(40, 40))
    assert canvas.current._highlightIndex == 0

    canvas.paint_records.clear()
    _move(canvas, 200, 180)
    assert canvas._near_start_point is False
    assert canvas.current._highlightIndex is None
    QtWidgets.QApplication.processEvents()
    assert len(canvas.paint_records) == 1
    highlight, region = canvas.paint_records[-1]
    assert highlight is None
    assert region.contains(QtCore.QPoint(40, 40))

    _move(canvas, 41, 42)
    finished = canvas.current
    canvas.finalise()
    assert canvas.shapes[-1] is finished
    assert finished._highlightIndex is None
    assert canvas._near_start_point is False
