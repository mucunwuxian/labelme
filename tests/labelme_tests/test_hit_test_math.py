"""Hit-test math runs for every shape on every mouse move, so it compares
squared distances instead of calling sqrt. These tests pin the results to the
straightforward sqrt-based formulas they replaced.

The shape types the canvas actually hit-tests with nearestEdge (polygon,
linestrip) must match exactly; for the others a tie between two edges may be
broken differently, which the tie test below makes explicit.
"""

import math
import random

import pytest
from PyQt5 import QtCore

import labelme.utils
from labelme.shape import Shape


def ref_distancetoline(point, line):
    p1, p2 = line
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()
    x3, y3 = point.x(), point.y()
    dx21, dy21 = x2 - x1, y2 - y1
    dx31, dy31 = x3 - x1, y3 - y1
    if dx31 * dx21 + dy31 * dy21 < 0:
        return math.sqrt(dx31 * dx31 + dy31 * dy31)
    dx32, dy32 = x3 - x2, y3 - y2
    if dx32 * -dx21 + dy32 * -dy21 < 0:
        return math.sqrt(dx32 * dx32 + dy32 * dy32)
    seg_len = math.sqrt(dx21 * dx21 + dy21 * dy21)
    if seg_len == 0:
        return math.sqrt(dx31 * dx31 + dy31 * dy31)
    return abs(dx21 * (y1 - y3) - dy21 * (x1 - x3)) / seg_len


def ref_nearest_vertex(shape, point, epsilon):
    min_distance = float("inf")
    min_i = None
    if shape.shape_type == "point":
        visual_size = (
            shape.point_object_size * 1.5
            if shape.selected
            else shape.point_object_size
        )
        epsilon = max(epsilon, visual_size / 2)
    point = QtCore.QPointF(point.x() * shape.scale, point.y() * shape.scale)
    for i, p in enumerate(shape.points):
        p = QtCore.QPointF(p.x() * shape.scale, p.y() * shape.scale)
        dist = labelme.utils.distance(p - point)
        if dist <= epsilon and dist < min_distance:
            min_distance = dist
            min_i = i
    return min_i


def ref_nearest_edge(shape, point, epsilon):
    min_distance = float("inf")
    post_i = None
    point = QtCore.QPointF(point.x() * shape.scale, point.y() * shape.scale)
    for i in range(len(shape.points)):
        start = shape.points[i - 1]
        end = shape.points[i]
        start = QtCore.QPointF(start.x() * shape.scale, start.y() * shape.scale)
        end = QtCore.QPointF(end.x() * shape.scale, end.y() * shape.scale)
        dist = ref_distancetoline(point, [start, end])
        if dist <= epsilon and dist < min_distance:
            min_distance = dist
            post_i = i
    return post_i


def _shape(kind, points, scale=1.0):
    s = Shape(shape_type=kind, label="x")
    for x, y in points:
        s.addPoint(QtCore.QPointF(x, y))
    s.close()
    s.scale = scale
    return s


def _random_shape(rng, kind, scale):
    cx, cy = rng.uniform(50, 800), rng.uniform(50, 600)
    if kind in ("rectangle", "circle", "line"):
        pts = [(cx, cy), (cx + rng.uniform(10, 90), cy + rng.uniform(10, 90))]
    elif kind == "point":
        pts = [(cx, cy)]
    else:
        n = rng.randint(3, 12)
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            r = rng.uniform(15, 70)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return _shape(kind, pts, scale)


def test_distancetoline_matches_sqrt_formula():
    rng = random.Random(0)
    for i in range(20000):
        p = QtCore.QPointF(rng.uniform(-50, 950), rng.uniform(-50, 750))
        a = QtCore.QPointF(rng.uniform(0, 900), rng.uniform(0, 700))
        b = (
            QtCore.QPointF(a.x(), a.y())  # degenerate segment
            if i % 20 == 0
            else QtCore.QPointF(rng.uniform(0, 900), rng.uniform(0, 700))
        )
        got = labelme.utils.distancetoline(p, [a, b])
        exp = ref_distancetoline(p, [a, b])
        assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)
        assert labelme.utils.distancetoline_sq(p, [a, b]) == pytest.approx(
            exp * exp, rel=1e-12, abs=1e-12
        )


@pytest.mark.parametrize("scale", [0.25, 1.0, 3.0])
@pytest.mark.parametrize(
    "kind", ["polygon", "linestrip", "rectangle", "circle", "point", "line"]
)
def test_nearest_vertex_matches_reference(kind, scale):
    rng = random.Random(1)
    for _ in range(60):
        shape = _random_shape(rng, kind, scale)
        for _ in range(20):
            base = shape.points[rng.randrange(len(shape.points))]
            p = QtCore.QPointF(
                base.x() + rng.uniform(-30, 30), base.y() + rng.uniform(-30, 30)
            )
            for eps in (2.0, 10.0, 40.0):
                assert shape.nearestVertex(p, eps) == ref_nearest_vertex(
                    shape, p, eps
                )


@pytest.mark.parametrize("scale", [0.25, 1.0, 3.0])
@pytest.mark.parametrize("kind", ["polygon", "linestrip"])
def test_nearest_edge_matches_reference(kind, scale):
    """These are the types the canvas hit-tests, so require an exact match."""
    rng = random.Random(2)
    for _ in range(60):
        shape = _random_shape(rng, kind, scale)
        for _ in range(20):
            base = shape.points[rng.randrange(len(shape.points))]
            p = QtCore.QPointF(
                base.x() + rng.uniform(-30, 30), base.y() + rng.uniform(-30, 30)
            )
            for eps in (2.0, 10.0, 40.0):
                assert shape.nearestEdge(p, eps) == ref_nearest_edge(
                    shape, p, eps
                )


def test_nearest_edge_ties_pick_an_equally_close_edge():
    """A square's centre is the same distance from all four edges.

    Which one wins is arbitrary; what must hold is that the winner really is
    at the minimum distance.
    """
    square = _shape("polygon", [(0, 0), (10, 0), (10, 10), (0, 10)])
    centre = QtCore.QPointF(5, 5)
    i = square.nearestEdge(centre, 20.0)
    assert i is not None
    dists = [
        ref_distancetoline(
            centre, [square.points[k - 1], square.points[k]]
        )
        for k in range(len(square.points))
    ]
    assert dists[i] == pytest.approx(min(dists))


def test_epsilon_boundary_is_inclusive():
    line = _shape("linestrip", [(0, 0), (100, 0)])
    p = QtCore.QPointF(50, 10)
    assert line.nearestEdge(p, 10.0) is not None  # exactly epsilon away
    assert line.nearestEdge(p, 9.999) is None
    v = _shape("polygon", [(0, 0), (100, 0), (100, 100)])
    assert v.nearestVertex(QtCore.QPointF(0, 5), 5.0) == 0
    assert v.nearestVertex(QtCore.QPointF(0, 5), 4.999) is None
