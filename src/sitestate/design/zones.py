"""Named zones: axis-aligned regions of the project frame ("Room A",
"corridor", ...) declared in the project configuration as
`{"zones": {name: [x0, y0, x1, y1]}}`. Claims are tagged with the zone
containing their centroid so reports and queries can speak the site
team's language instead of coordinates.
"""

from __future__ import annotations


def zone_of(zones: dict[str, list[float]] | None, x: float, y: float) -> str:
    if not zones:
        return ""
    for name, (x0, y0, x1, y1) in zones.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return ""
