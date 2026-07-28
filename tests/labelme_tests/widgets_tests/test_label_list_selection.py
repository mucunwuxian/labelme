"""Selecting shapes updates the label list in one selection change instead of
one per shape. The end state must be indistinguishable from the per-item loop
it replaces: same rows selected, same order, same scroll position.
"""

import pytest
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from labelme.shape import Shape
from labelme.widgets.label_list_widget import LabelListWidget
from labelme.widgets.label_list_widget import LabelListWidgetItem


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_list(n=50):
    widget = LabelListWidget()
    shapes = []
    for i in range(n):
        shape = Shape(shape_type="polygon", label=f"l{i}")
        for x, y in ((0, 0), (10, 0), (10, 10)):
            shape.addPoint(QtCore.QPointF(x + i, y + i))
        shape.close()
        shapes.append(shape)
        widget.addItem(LabelListWidgetItem(shape.label, shape))
    return widget, shapes


def _selected_rows(widget):
    return sorted(i.row() for i in widget.selectedIndexes())


def _legacy_select(widget, shapes):
    widget.clearSelection()
    for shape in shapes:
        item = widget.findItemByShape(shape)
        if item is not None:
            widget.selectItem(item)


@pytest.mark.parametrize("picked", [[], [0], [3, 7, 9], list(range(0, 50, 2)),
                                    list(range(50))])
def test_select_only_items_matches_the_per_item_loop(app, picked):
    widget, shapes = _make_list()
    chosen = [shapes[i] for i in picked]

    _legacy_select(widget, chosen)
    expected = _selected_rows(widget)

    widget.clearSelection()
    by_shape = widget.itemsByShape()
    widget.selectOnlyItems([by_shape[s] for s in chosen])
    assert _selected_rows(widget) == expected


def test_select_only_items_replaces_the_previous_selection(app):
    widget, shapes = _make_list()
    by_shape = widget.itemsByShape()
    widget.selectOnlyItems([by_shape[shapes[1]], by_shape[shapes[2]]])
    assert _selected_rows(widget) == [1, 2]
    widget.selectOnlyItems([by_shape[shapes[5]]])
    assert _selected_rows(widget) == [5]
    widget.selectOnlyItems([])
    assert _selected_rows(widget) == []


def test_items_by_shape_agrees_with_find_item_by_shape(app):
    widget, shapes = _make_list()
    by_shape = widget.itemsByShape()
    assert len(by_shape) == len(shapes)
    for shape in shapes:
        assert by_shape[shape] is widget.findItemByShape(shape)

    # a shape that is not in the list at all
    stray = Shape(shape_type="polygon", label="stray")
    stray.addPoint(QtCore.QPointF(0, 0))
    assert stray not in by_shape
    assert widget.findItemByShape(stray) is None

    # and it stays in sync after rows are removed
    widget.removeItem(widget.findItemByShape(shapes[0]))
    by_shape = widget.itemsByShape()
    assert shapes[0] not in by_shape
    assert len(by_shape) == len(shapes) - 1


def test_one_selection_change_for_the_whole_selection(app):
    widget, shapes = _make_list()
    by_shape = widget.itemsByShape()
    changes = []
    widget.selectionModel().selectionChanged.connect(
        lambda *a: changes.append(1)
    )
    widget.selectOnlyItems([by_shape[s] for s in shapes[:20]])
    assert len(changes) == 1
