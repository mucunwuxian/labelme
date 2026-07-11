"""Regression tests for the navigator / update-distribution scene-layer cache.

The cache compares an immutable render snapshot on every paint; these tests
guard the three invariants that matter:
  1. cached painting is bit-identical to direct painting
  2. mutating a drawn attribute (incl. in-place QPointF changes) rebuilds
  3. viewport moves never rebuild
"""

import numpy as np
import pytest
from PyQt5 import QtCore
from PyQt5 import QtGui

from labelme.shape import Shape
from labelme.widgets.navigator import NavigatorWidget
from labelme.widgets.update_distribution import UpdateDistributionWidget


def _qimage_to_np(qimg):
    qimg = qimg.convertToFormat(QtGui.QImage.Format_ARGB32)
    h = qimg.height()
    bpl = qimg.bytesPerLine()
    ptr = qimg.constBits()
    ptr.setsize(qimg.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, bpl)
    return arr[:, : qimg.width() * 4].copy()


def _grab(widget):
    return _qimage_to_np(widget.grab().toImage())


def _make_shape(shape_type, pts, color, closed=False, modified_at=None):
    s = Shape(shape_type=shape_type)
    for x, y in pts:
        s.addPoint(QtCore.QPointF(x, y))
    s.line_color = QtGui.QColor(*color)
    if closed:
        s._closed = True
    s.modified_at = modified_at
    return s


def _make_shapes():
    return [
        _make_shape(
            "rectangle",
            [(20, 20), (180, 140)],
            (255, 0, 0, 200),
            modified_at="2026-07-01T00:00:00",
        ),
        _make_shape(
            "polygon",
            [(60, 40), (200, 60), (170, 180), (50, 150)],
            (0, 128, 255, 180),
            closed=True,
        ),
        _make_shape("circle", [(260, 100), (300, 130)], (0, 200, 0, 220)),
        _make_shape("point", [(320, 40)], (255, 0, 255, 255)),
        _make_shape(
            "linestrip", [(30, 200), (120, 230), (220, 200)], (255, 165, 0, 255)
        ),
        _make_shape("line", [(250, 180), (350, 240)], (0, 100, 100, 255)),
        _make_shape("points", [(340, 150), (355, 170)], (200, 100, 0, 255)),
        _make_shape("mask", [(300, 30), (380, 90)], (90, 90, 220, 160)),
        _make_shape("polygon", [], (0, 0, 0, 255)),  # empty -> skipped
    ]


def _make_widget(cls, shapes):
    w = cls()
    w.resize(400, 300)
    img = QtGui.QImage(400, 260, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor(230, 230, 230))
    w.setPixmap(QtGui.QPixmap.fromImage(img))
    w.setShapes(shapes)
    w.setViewportRect(0.1, 0.1, 0.4, 0.5)
    return w


@pytest.mark.gui
@pytest.mark.parametrize("cls", [NavigatorWidget, UpdateDistributionWidget])
def test_cache_matches_direct_painting(qtbot, cls):
    w = _make_widget(cls, _make_shapes())
    qtbot.addWidget(w)
    if w.devicePixelRatioF() != int(w.devicePixelRatioF()):
        pytest.skip("cache is bypassed at fractional device pixel ratio")
    w._scene_layer_enabled = False
    direct = _grab(w)
    w._scene_layer_enabled = True
    cached = _grab(w)
    assert direct.shape == cached.shape
    assert np.array_equal(direct, cached), "cached paint must be bit-identical"


@pytest.mark.gui
@pytest.mark.parametrize("cls", [NavigatorWidget, UpdateDistributionWidget])
def test_mutations_rebuild_cache(qtbot, cls):
    shapes = _make_shapes()
    w = _make_widget(cls, shapes)
    qtbot.addWidget(w)
    if w.devicePixelRatioF() != int(w.devicePixelRatioF()):
        pytest.skip("cache is bypassed at fractional device pixel ratio")
    _grab(w)
    before = w._scene_layer_builds

    # In-place QPointF mutation inside the same list object must be detected.
    shapes[0].points[0].setX(shapes[0].points[0].x() + 3)
    _grab(w)
    assert w._scene_layer_builds == before + 1

    shapes[3].modified_at = "2026-07-11T12:00:00"
    _grab(w)
    assert w._scene_layer_builds == before + 2

    # Unchanged repaints must hit the cache.
    _grab(w)
    _grab(w)
    assert w._scene_layer_builds == before + 2


@pytest.mark.gui
@pytest.mark.parametrize("cls", [NavigatorWidget, UpdateDistributionWidget])
def test_fractional_dpr_bypasses_cache(qtbot, cls):
    """At non-integer DPR the layer grid cannot align with physical pixels,
    so the cache must be bypassed (direct painting) regardless of platform."""

    class _FractionalDpr(cls):
        def devicePixelRatioF(self):
            return 1.5

    w = _make_widget(_FractionalDpr, _make_shapes())
    qtbot.addWidget(w)
    _grab(w)
    _grab(w)
    assert w._scene_layer_builds == 0


@pytest.mark.gui
@pytest.mark.parametrize("cls", [NavigatorWidget, UpdateDistributionWidget])
def test_viewport_moves_do_not_rebuild(qtbot, cls):
    w = _make_widget(cls, _make_shapes())
    qtbot.addWidget(w)
    if w.devicePixelRatioF() != int(w.devicePixelRatioF()):
        pytest.skip("cache is bypassed at fractional device pixel ratio")
    _grab(w)
    before = w._scene_layer_builds
    for i in range(50):
        w.setViewportRect(0.002 * i, 0.003 * i, 0.3, 0.4)
        _grab(w)
    assert w._scene_layer_builds == before
