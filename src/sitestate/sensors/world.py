"""Simulated site and carrier used for development and the demo pipeline.

The simulator stands in for a real indoor construction area so the whole
platform can be exercised end-to-end. Real deployments replace only the
sensor adapters (e.g. with ROS 2 topic adapters); everything downstream of
the adapter contract is unchanged - that is the point of the architecture.

The carrier maintains a deliberately imperfect pose estimate: a rigid
map-frame offset (wrong initial alignment) plus random-walk drift, the way
real odometry/SLAM estimates degrade. Sensors render geometry from the
true pose but record it against the estimated pose, so registration to
control points is genuinely required, exactly as on a real site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Segment = tuple[tuple[float, float], tuple[float, float]]


@dataclass
class SimWorld:
    """Walls/objects as 2D segments plus surveyed fiducial control points."""

    walls: list[Segment] = field(default_factory=list)
    fiducials: dict[str, tuple[float, float]] = field(default_factory=dict)

    def add_box(self, x: float, y: float, w: float, h: float) -> None:
        c = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        for i in range(4):
            self.walls.append((c[i], c[(i + 1) % 4]))

    def _segment_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([s[0] for s in self.walls], dtype=float)
        b = np.array([s[1] for s in self.walls], dtype=float)
        return a, b

    def raycast(self, origin: tuple[float, float], angles: np.ndarray, max_range: float) -> np.ndarray:
        """Nearest wall hit per ray angle; max_range where nothing is hit."""
        a, b = self._segment_arrays()
        ab = b - a
        o = np.asarray(origin, dtype=float)
        ao = a - o
        ranges = np.full(len(angles), max_range, dtype=float)
        for i, ang in enumerate(angles):
            d = np.array([math.cos(ang), math.sin(ang)])
            denom = d[0] * ab[:, 1] - d[1] * ab[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (ao[:, 0] * ab[:, 1] - ao[:, 1] * ab[:, 0]) / denom
                u = (ao[:, 0] * d[1] - ao[:, 1] * d[0]) / denom
            valid = (np.abs(denom) > 1e-12) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
            if valid.any():
                ranges[i] = min(max_range, float(t[valid].min()))
        return ranges


class SimCarrier:
    """A supervised carrier (robot/trolley) following waypoints at set speed."""

    def __init__(
        self,
        waypoints: list[tuple[float, float]],
        speed: float = 0.5,
        frame_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        drift_rate: float = 0.002,
        seed: int = 0,
        carrier_type: str = "wheeled_robot",
        name: str = "sim-carrier",
    ):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.speed = speed
        self.frame_offset = frame_offset
        self.drift_rate = drift_rate
        self.carrier_type = carrier_type
        self.name = name
        self._rng = np.random.default_rng(seed)
        self._walk: dict[float, np.ndarray] = {}
        self._walk_order: list[float] = []

        # cumulative arc length along the waypoint path
        self._seg_len = [
            float(np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]))
            for i in range(len(self.waypoints) - 1)
        ]
        self.path_length = sum(self._seg_len)
        self.duration = self.path_length / self.speed

    def describe(self) -> dict:
        return {"type": self.carrier_type, "name": self.name, "speed_mps": self.speed}

    # -- ground truth ------------------------------------------------------
    def true_pose(self, t: float) -> tuple[float, float, float]:
        s = min(max(t, 0.0) * self.speed, self.path_length - 1e-9)
        for i, seg in enumerate(self._seg_len):
            if s <= seg or i == len(self._seg_len) - 1:
                p0, p1 = self.waypoints[i], self.waypoints[i + 1]
                frac = 0.0 if seg == 0 else min(s / seg, 1.0)
                p = p0 + (p1 - p0) * frac
                heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                return float(p[0]), float(p[1]), heading
            s -= seg
        raise RuntimeError("unreachable")

    # -- estimated pose (what the sensors record) --------------------------
    def _random_walk(self, t: float) -> np.ndarray:
        if t in self._walk:
            return self._walk[t]
        last_t = self._walk_order[-1] if self._walk_order else 0.0
        last = self._walk.get(last_t, np.zeros(3))
        dt = max(t - last_t, 0.0)
        step = self._rng.normal(0.0, self.drift_rate * math.sqrt(max(dt, 1e-9)), size=3)
        step[2] *= 0.2  # heading drifts more slowly than position
        walk = last + step
        self._walk[t] = walk
        self._walk_order.append(t)
        return walk

    def estimated_pose(self, t: float) -> tuple[float, float, float, float]:
        """Returns (x, y, theta, sigma_xy): the drifted estimate the carrier
        believes, with a 1-sigma position uncertainty that grows with time."""
        x, y, th = self.true_pose(t)
        ox, oy, oth = self.frame_offset
        c, s = math.cos(oth), math.sin(oth)
        ex = c * x - s * y + ox
        ey = s * x + c * y + oy
        eth = th + oth
        walk = self._random_walk(t)
        sigma = 0.01 + self.drift_rate * math.sqrt(max(t, 0.0)) * 1.5
        return ex + walk[0], ey + walk[1], eth + walk[2], sigma
