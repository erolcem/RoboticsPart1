"""Traversability analysis: the robot-facing world model (Phase 4).

Derives where a ground robot of a given radius can drive, honestly
distinguishing *known free* from *never observed*: unobserved cells are
non-traversable by policy, not assumed clear. Produces a class grid and a
cost grid (ROS-style convention: 0 free rising near obstacles, 253
inflated, 254 lethal, 255 unknown) as derived evidence, consumed by the
robot-costmap output adapter and the query API.
"""

from __future__ import annotations

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import FREE, OCCUPIED, dilate

# class codes stored in the traversability grid
TRAVERSABLE, INFLATED, OBSTACLE, UNKNOWN = 0, 1, 2, 3
OBSERVED = 2  # coverage code


class TraversabilityAnalysis(ProcessingPlugin):
    _manifest = PluginManifest(
        name="traversability-analysis",
        version="1.0.0",
        consumes=["occupancy_geometry", "coverage"],
        produces=["traversability"],
        mode="offline",
        description="Robot traversability and navigation cost grid from occupancy + coverage",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, robot_radius_m: float = 0.3, **params) -> None:
        geoms = ctx.claims("occupancy_geometry")
        covs = ctx.claims("coverage")
        if not geoms or not covs:
            ctx.note("needs accepted occupancy_geometry and coverage claims")
            return
        geom, cov = geoms[0], covs[0]
        g = ctx.payload(geom["payload"]["evidence_id"])
        c = ctx.payload(cov["payload"]["evidence_id"])
        occ, coverage = g["occ"], c["coverage"]
        res = float(g["res"])

        obstacle = occ == OCCUPIED
        inflate_cells = max(1, round(robot_radius_m / res))
        inflated = dilate(obstacle, inflate_cells) & ~obstacle
        known_free = (occ == FREE) & (coverage == OBSERVED)

        classes = np.full(occ.shape, UNKNOWN, dtype=np.int8)
        classes[known_free] = TRAVERSABLE
        classes[inflated & known_free] = INFLATED
        classes[obstacle] = OBSTACLE

        cost = np.full(occ.shape, 255, dtype=np.uint8)  # unknown
        cost[classes == TRAVERSABLE] = 0
        cost[classes == INFLATED] = 253
        cost[classes == OBSTACLE] = 254

        trav_ev = ctx.store_derived(
            "traversability_grid",
            {
                "classes": classes,
                "cost": cost,
                "x0": g["x0"],
                "y0": g["y0"],
                "res": g["res"],
            },
            meta={"robot_radius_m": robot_radius_m, "frame": "project"},
        )
        n = classes.size
        ctx.emit_claim(
            kind="traversability",
            payload={
                "evidence_id": trav_ev,
                "robot_radius_m": robot_radius_m,
                "fractions": {
                    "traversable": float((classes == TRAVERSABLE).sum() / n),
                    "inflated": float((classes == INFLATED).sum() / n),
                    "obstacle": float((classes == OBSTACLE).sum() / n),
                    "unknown": float((classes == UNKNOWN).sum() / n),
                },
            },
            confidence=min(geom["confidence"], cov["confidence"]),
            evidence_ids=[
                trav_ev,
                geom["payload"]["evidence_id"],
                cov["payload"]["evidence_id"],
            ],
            subject="traversability:main",
        )
