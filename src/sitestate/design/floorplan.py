"""Designed-state model: a minimal floor plan the as-built state is
compared against (Phase 3 of the roadmap).

Deliberately a tiny neutral JSON schema — walls as 2D segments — rather
than IFC: a real IFC/BIM import belongs in a future *input adapter* that
translates into this internal form, keeping the comparison plug-ins
format-agnostic (same philosophy as the sensor adapters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Segment = tuple[tuple[float, float], tuple[float, float]]


@dataclass
class FloorPlan:
    name: str = "unnamed plan"
    walls: list[Segment] = field(default_factory=list)

    def add_box(self, x: float, y: float, w: float, h: float) -> None:
        c = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        for i in range(4):
            self.walls.append((c[i], c[(i + 1) % 4]))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FloorPlan":
        return FloorPlan(
            name=d.get("name", "unnamed plan"),
            walls=[((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in d.get("walls", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "walls": [[list(a), list(b)] for a, b in self.walls]}

    def rasterize(self, x0: float, y0: float, nx: int, ny: int, res: float) -> np.ndarray:
        """Boolean grid of cells the plan expects to be occupied by walls."""
        mask = np.zeros((ny, nx), dtype=bool)
        for (ax, ay), (bx, by) in self.walls:
            length = float(np.hypot(bx - ax, by - ay))
            n = max(int(length / (res / 2)), 1)
            for k in range(n + 1):
                px = ax + (bx - ax) * k / n
                py = ay + (by - ay) * k / n
                i = int((px - x0) / res)
                j = int((py - y0) / res)
                if 0 <= i < nx and 0 <= j < ny:
                    mask[j, i] = True
        return mask
