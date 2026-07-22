"""Shared occupancy-grid helpers for the 2D processing plug-ins."""

from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


class Grid2D:
    def __init__(self, x0: float, y0: float, width_m: float, height_m: float, res: float):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx = int(np.ceil(width_m / res))
        self.ny = int(np.ceil(height_m / res))
        self.hits = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.passes = np.zeros((self.ny, self.nx), dtype=np.int32)

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        i = int((x - self.x0) / self.res)
        j = int((y - self.y0) / self.res)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def add_ray(self, x0: float, y0: float, x1: float, y1: float, hit: bool) -> None:
        """Trace free-space cells from (x0,y0) to (x1,y1); mark endpoint as hit."""
        c0, c1 = self.cell(x0, y0), self.cell(x1, y1)
        if c0 is None and c1 is None:
            return
        # Bresenham over cell indices (clamped into the grid)
        i0, j0 = c0 if c0 else self._clamped(x0, y0)
        i1, j1 = c1 if c1 else self._clamped(x1, y1)
        di, dj = abs(i1 - i0), -abs(j1 - j0)
        si, sj = (1 if i1 > i0 else -1), (1 if j1 > j0 else -1)
        err = di + dj
        i, j = i0, j0
        while True:
            if i == i1 and j == j1:
                break
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self.passes[j, i] += 1
            e2 = 2 * err
            if e2 >= dj:
                err += dj
                i += si
            if e2 <= di:
                err += di
                j += sj
        if hit and c1 is not None:
            self.hits[j1, i1] += 1
        elif not hit and c1 is not None:
            self.passes[j1, i1] += 1

    def _clamped(self, x: float, y: float) -> tuple[int, int]:
        i = min(max(int((x - self.x0) / self.res), 0), self.nx - 1)
        j = min(max(int((y - self.y0) / self.res), 0), self.ny - 1)
        return i, j

    def classify(self, min_hits: int = 2, min_passes: int = 3) -> np.ndarray:
        occ = np.full((self.ny, self.nx), UNKNOWN, dtype=np.int8)
        total = self.hits + self.passes
        occupied = (self.hits >= min_hits) & (self.hits >= 0.2 * total)
        free = (~occupied) & (self.passes >= min_passes)
        occ[occupied] = OCCUPIED
        occ[free] = FREE
        return occ


def connected_regions(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask; returns arrays of (j, i) cells."""
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[np.ndarray] = []
    ny, nx = mask.shape
    for j0 in range(ny):
        for i0 in range(nx):
            if not mask[j0, i0] or visited[j0, i0]:
                continue
            stack = [(j0, i0)]
            visited[j0, i0] = True
            cells = []
            while stack:
                j, i = stack.pop()
                cells.append((j, i))
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii] and not visited[jj, ii]:
                        visited[jj, ii] = True
                        stack.append((jj, ii))
            regions.append(np.array(cells))
    return regions
