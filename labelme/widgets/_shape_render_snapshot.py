def shape_render_snapshot(shapes) -> tuple:
    """Immutable snapshot of exactly the Shape attributes the minimap docks
    (NavigatorWidget / UpdateDistributionWidget) read while painting.

    Compared on every paint to decide whether the cached shape layer is
    still valid. This deliberately avoids external revision counters or
    invalidation hooks in shape-mutation paths: an in-place change of any
    drawn attribute (including QPointF coordinates inside the same list
    object) always produces a different snapshot.

    Order matters (kept as-is: painting order affects overlaps). Attributes
    the docks do not draw (selected, label, mask, ...) are intentionally
    excluded so unrelated changes do not invalidate the cache.
    """
    result = []
    for s in shapes:
        line_color = getattr(s, "line_color", None)
        result.append(
            (
                s.shape_type,
                tuple((p.x(), p.y()) for p in s.points),
                line_color.rgba() if line_color is not None else None,
                bool(getattr(s, "_closed", False)),
                bool(getattr(s, "modified_at", None)),
            )
        )
    return tuple(result)
