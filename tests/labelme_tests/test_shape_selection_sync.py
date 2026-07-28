"""Mirroring a canvas selection into the label list must not feed back.

The label list emits a selection change while we are setting it, and its
handler would push that straight back into the canvas. The sync is guarded, so
a re-entrant request is dropped and the selection the caller asked for is the
one that survives.
"""

import numpy as np
import PIL.Image
import pytest
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import labelme.app
from labelme.app import MainWindow
from labelme.shape import Shape


@pytest.fixture(autouse=True)
def _isolated_qtsettings(tmp_path, monkeypatch):
    """Keep the real "labelme" QSettings out of the test run.

    MainWindow reads and writes recent files, window geometry and canvas
    settings; without this the suite would edit the developer's own config.
    """
    settings = QtCore.QSettings(
        str(tmp_path / "qtsettings.ini"), QtCore.QSettings.IniFormat
    )
    monkeypatch.setattr(
        labelme.app.QtCore, "QSettings", lambda *args, **kwargs: settings
    )
    yield


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path):
    path = tmp_path / "img.png"
    PIL.Image.fromarray(np.full((200, 300, 3), 210, np.uint8)).save(path)
    window = MainWindow()
    window.resize(600, 500)
    window.show()
    app.processEvents()
    assert window._load_file(str(path))

    shapes = []
    for i in range(4):
        shape = Shape(shape_type="polygon", label=f"l{i}")
        for dx, dy in ((0, 0), (40, 0), (40, 40), (0, 40)):
            shape.addPoint(QtCore.QPointF(20 + i * 50 + dx, 20 + dy))
        shape.close()
        shapes.append(shape)
    window._load_shapes(shapes)
    app.processEvents()
    yield window, shapes
    # Closing a dirty window pops the "save annotations?" dialog and blocks.
    window._is_changed = False
    window.close()
    window.deleteLater()
    app.processEvents()


def test_syncing_the_list_does_not_echo_back_into_the_canvas(win, app):
    window, shapes = win
    # The echo would reach the canvas through selectShapes(), so watch its
    # signal rather than patching the handler.
    seen = []
    window.canvas.selectionChanged.connect(lambda s: seen.append(list(s)))

    window.shapeSelectionChanged([shapes[0], shapes[1]])
    app.processEvents()

    assert seen == []
    assert window.canvas.selectedShapes == [shapes[0], shapes[1]]


def test_a_re_entrant_request_is_dropped_and_the_outer_one_wins(win, app):
    window, shapes = win
    calls = []

    def nested_request():
        calls.append(1)
        if len(calls) < 3:
            window.shapeSelectionChanged([shapes[3]])

    window.labelList.itemSelectionChanged.connect(nested_request)
    try:
        window.shapeSelectionChanged([shapes[0], shapes[1]])
        app.processEvents()
    finally:
        window.labelList.itemSelectionChanged.disconnect(nested_request)

    # exactly one re-entry, and it did not recurse further
    assert len(calls) == 1
    assert window.canvas.selectedShapes == [shapes[0], shapes[1]]
    assert shapes[0].selected and shapes[1].selected
    assert not shapes[3].selected
    # the guard is released again
    assert window._syncing_label_selection is False

    # and the next ordinary selection still works
    window.shapeSelectionChanged([shapes[2]])
    assert window.canvas.selectedShapes == [shapes[2]]
    assert not shapes[0].selected


def test_selection_flags_are_repaired(win, app):
    window, shapes = win
    window.shapeSelectionChanged([shapes[0], shapes[1]])
    shapes[0].selected = False  # drifted out of sync
    window.shapeSelectionChanged([shapes[0], shapes[1]])
    assert shapes[0].selected and shapes[1].selected


def test_removing_a_point_never_repaints_the_whole_canvas(win, app):
    """The menu / shortcut path used to end in an argument-less update()."""
    window, shapes = win
    shape = shapes[0]
    for extra in ((60, 20), (60, 60)):
        shape.addPoint(QtCore.QPointF(*extra))
    window.canvas.prevhShape = shape
    window.canvas.prevhVertex = 2

    full_repaints = []
    canvas = window.canvas
    original_update = canvas.update
    original_repaint = canvas.repaint

    def wrap(original):
        def recorder(*args):
            if not args:
                full_repaints.append(True)
            return original(*args)

        return recorder

    canvas.update = wrap(original_update)
    canvas.repaint = wrap(original_repaint)
    try:
        window.removeSelectedPoint()
        app.processEvents()
    finally:
        canvas.update = original_update
        canvas.repaint = original_repaint

    assert not full_repaints, "the vertex removal repainted the whole canvas"
