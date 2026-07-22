"""Shared occupancy-grid machinery for the 2D processing plug-ins.

v1.0: rays are integrated vectorially (numpy) and cells are classified by
log-odds with an explicit inverse sensor model, so the grid carries a real
per-cell occupancy probability - the geometric uncertainty that mapping
confidence and change confidence are built from.
"""

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

    # -- coordinates -------------------------------------------------------
    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        i = int((x - self.x0) / self.res)
        j = int((y - self.y0) / self.res)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def _cells_of(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = ((xs - self.x0) / self.res).astype(np.int64)
        j = ((ys - self.y0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return i, j, ok

    # -- ray integration ---------------------------------------------------
    def integrate_scan(self, origin: tuple[float, float], ends: np.ndarray, hits: np.ndarray) -> None:
        """Vectorised integration of one scan: free-space samples along each
        ray (one per cell length) plus endpoint hit/pass accounting.

        `ends` is (R, 2) world endpoints, `hits` a boolean array: True where
        the ray terminated on a surface, False where it reached max range.
        """
        ox, oy = float(origin[0]), float(origin[1])
        d = ends - np.array([ox, oy])
        dist = np.hypot(d[:, 0], d[:, 1])
        dist = np.maximum(dist, 1e-9)

        # free samples at `res` spacing, stopping one cell short of the
        # endpoint so a hit cell is never also marked free by its own ray
        n_free = np.maximum(((dist - self.res * 0.6) / self.res).astype(np.int64), 0)
        max_n = int(n_free.max()) if len(n_free) else 0
        if max_n > 0:
            k = np.arange(1, max_n + 1, dtype=np.float64)
            frac = (k[None, :] * self.res) / dist[:, None]  # (R, max_n)
            valid = k[None, :] <= n_free[:, None]
            px = ox + d[:, 0:1] * frac
            py = oy + d[:, 1:2] * frac
            i, j, ok = self._cells_of(px[valid], py[valid])
            np.add.at(self.passes, (j[ok], i[ok]), 1)

        # endpoints
        i, j, ok = self._cells_of(ends[:, 0], ends[:, 1])
        hit_ok = ok & hits.astype(bool)
        pass_ok = ok & ~hits.astype(bool)
        np.add.at(self.hits, (j[hit_ok], i[hit_ok]), 1)
        np.add.at(self.passes, (j[pass_ok], i[pass_ok]), 1)

    def add_ray(self, x0: float, y0: float, x1: float, y1: float, hit: bool) -> None:
        """Single-ray convenience wrapper around integrate_scan."""
        self.integrate_scan(
            (x0, y0), np.array([[x1, y1]], dtype=float), np.array([hit], dtype=bool)
        )

    # -- classification ----------------------------------------------------
    def log_odds(self, l_occ: float = 1.2, l_free: float = -0.35) -> np.ndarray:
        """Per-cell log-odds under a simple inverse sensor model: each hit
        adds l_occ, each free-space pass adds l_free."""
        return self.hits * l_occ + self.passes * l_free

    def probability(self, l_occ: float = 1.2, l_free: float = -0.35) -> np.ndarray:
        L = np.clip(self.log_odds(l_occ, l_free), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-L))

    def classify(
        self,
        l_occ: float = 1.2,
        l_free: float = -0.35,
        occ_thresh: float = 1.2,
        free_thresh: float = -1.0,
    ) -> np.ndarray:
        """OCCUPIED where cumulative evidence favours occupancy, FREE where
        it favours emptiness, UNKNOWN where evidence is insufficient."""
        L = self.log_odds(l_occ, l_free)
        occ = np.full((self.ny, self.nx), UNKNOWN, dtype=np.int8)
        occ[L >= occ_thresh] = OCCUPIED
        occ[L <= free_thresh] = FREE
        return occ

    def evidential(
        self, w_occ: float = 0.55, w_free: float = 0.25
    ) -> dict[str, np.ndarray]:
        """Dempster-Shafer evidential layer (see docs/sota-review.md §2).

        Each hit is a simple support function for {Occupied} with mass
        w_occ, each pass one for {Free} with mass w_free; combining N of a
        kind gives belief b = 1 - (1-w)^N. Dempster's rule then fuses the
        two single-focus beliefs, yielding per cell:

          m_occ, m_free - belief masses for the two states
          ignorance     - mass on {O ∪ F}: epistemic "never saw enough"
          conflict      - normalization mass K: evidence actively
                          disagrees - the signature of moved objects,
                          dynamics or residual misregistration, and a
                          different thing from ignorance.
        """
        b_occ = 1.0 - np.power(1.0 - w_occ, self.hits)
        b_free = 1.0 - np.power(1.0 - w_free, self.passes)
        conflict = b_occ * b_free
        denom = np.maximum(1.0 - conflict, 1e-9)
        return {
            "m_occ": (b_occ * (1.0 - b_free) / denom).astype(np.float32),
            "m_free": (b_free * (1.0 - b_occ) / denom).astype(np.float32),
            "ignorance": ((1.0 - b_occ) * (1.0 - b_free) / denom).astype(np.float32),
            "conflict": conflict.astype(np.float32),
        }

    def decision_entropy(self, l_occ: float = 1.2, l_free: float = -0.35) -> float:
        """Mean binary entropy over decided (non-unknown) cells: 0 = every
        decision is certain, 1 = coin flips. A legible map-quality scalar."""
        occ = self.classify(l_occ, l_free)
        p = self.probability(l_occ, l_free)[occ != UNKNOWN]
        if p.size == 0:
            return 1.0
        p = np.clip(p, 1e-9, 1 - 1e-9)
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return float(h.mean())


def dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Binary dilation by N cells (4-neighbourhood), used for tolerance
    bands in plan comparison and obstacle inflation in traversability."""
    out = mask.astype(bool).copy()
    for _ in range(cells):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def region_stats(cells: np.ndarray, x0: float, y0: float, res: float) -> dict:
    """Common world-frame geometry facts for a connected cell region."""
    js, is_ = cells[:, 0], cells[:, 1]
    return {
        "centroid": [x0 + (is_.mean() + 0.5) * res, y0 + (js.mean() + 0.5) * res],
        "bbox": [
            x0 + is_.min() * res,
            y0 + js.min() * res,
            x0 + (is_.max() + 1) * res,
            y0 + (js.max() + 1) * res,
        ],
        "n_cells": int(len(cells)),
        "area_m2": float(len(cells) * res * res),
    }


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
