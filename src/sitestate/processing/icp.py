"""Shared 2D rigid-fit and ICP utilities.

Used by control-point registration (direct Kabsch on matched fiducials),
pose refinement (scan-to-map ICP) and registration verification
(map-to-map ICP). Nearest neighbours use a bucket hash so no external
spatial-index dependency is needed.
"""

from __future__ import annotations

import math

import numpy as np


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (R, t) with dst ~ R @ src + t (Kabsch)."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    h = (src - sc).T @ (dst - dc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    t = dc - r @ sc
    return r, t


class PointHash:
    """Vectorised approximate nearest-neighbour lookup for 2D point sets.

    Reference points are rastered into a cell grid holding the per-cell
    centroid; a query checks the 3x3 neighbourhood of its cell, entirely in
    numpy. With cell >= max query distance, any true neighbour within that
    distance is guaranteed to be in the neighbourhood; matching against the
    per-cell centroid is the standard voxel approximation used to keep ICP
    dependency-free and fast.
    """

    def __init__(self, points: np.ndarray, cell: float):
        self.cell = cell
        keys = np.floor(points / cell).astype(np.int64)
        self.kmin = keys.min(axis=0) - 1  # one-cell margin for the 3x3 window
        shape = keys.max(axis=0) - self.kmin + 2
        self.nx, self.ny = int(shape[0]), int(shape[1])
        cnt = np.zeros((self.nx, self.ny))
        sx = np.zeros((self.nx, self.ny))
        sy = np.zeros((self.nx, self.ny))
        ij = keys - self.kmin
        np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1.0)
        np.add.at(sx, (ij[:, 0], ij[:, 1]), points[:, 0])
        np.add.at(sy, (ij[:, 0], ij[:, 1]), points[:, 1])
        self.valid = cnt > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cx = np.where(self.valid, sx / cnt, np.inf)
            self.cy = np.where(self.valid, sy / cnt, np.inf)

    def nearest_batch(self, ps: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
        """For each query point: (matched reference point, within-max_dist mask)."""
        ij = np.floor(ps / self.cell).astype(np.int64) - self.kmin
        ij[:, 0] = np.clip(ij[:, 0], 1, self.nx - 2)
        ij[:, 1] = np.clip(ij[:, 1], 1, self.ny - 2)
        best_d2 = np.full(len(ps), np.inf)
        best_xy = np.zeros((len(ps), 2))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                qx = self.cx[ij[:, 0] + dx, ij[:, 1] + dy]
                qy = self.cy[ij[:, 0] + dx, ij[:, 1] + dy]
                d2 = (qx - ps[:, 0]) ** 2 + (qy - ps[:, 1]) ** 2
                better = d2 < best_d2
                best_d2 = np.where(better, d2, best_d2)
                best_xy[better] = np.stack([qx, qy], axis=1)[better]
        ok = best_d2 <= max_dist * max_dist
        return best_xy, ok


def icp_2d(
    src: np.ndarray,
    ref_hash: PointHash,
    max_pair_dist: float = 0.3,
    iterations: int = 4,
    min_pairs: int = 20,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Align src points to the reference set: returns (R, t, rmse, n_pairs).

    Identity with rmse=inf when too few correspondences are found - the
    caller must treat that as 'cannot verify', never as 'aligned'.
    """
    r_total = np.eye(2)
    t_total = np.zeros(2)
    moved = src.copy()
    rmse, n_pairs = float("inf"), 0
    for _ in range(iterations):
        matched, ok = ref_hash.nearest_batch(moved, max_pair_dist)
        n_pairs = int(ok.sum())
        if n_pairs < min_pairs:
            return np.eye(2), np.zeros(2), float("inf"), n_pairs
        pairs_src = moved[ok]
        pairs_dst = matched[ok]
        r, t = rigid_fit(pairs_src, pairs_dst)
        moved = moved @ r.T + t
        r_total = r @ r_total
        t_total = r @ t_total + t
        rmse = float(np.sqrt(np.mean(np.sum((pairs_src @ r.T + t - pairs_dst) ** 2, axis=1))))
    return r_total, t_total, rmse, n_pairs


def transform_pose(r: np.ndarray, t: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to a (x, y, theta) pose."""
    xy = r @ pose[:2] + t
    theta = pose[2] + math.atan2(r[1, 0], r[0, 0])
    return np.array([xy[0], xy[1], theta])
