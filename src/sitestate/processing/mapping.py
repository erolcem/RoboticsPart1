"""Occupancy mapping: fuse registered 2D scans into a project-frame grid.

Consumes lidar scans plus the mission's registration claim; produces an
occupancy grid as derived evidence and an occupancy_geometry claim whose
confidence reflects registration quality and pose uncertainty.
"""

from __future__ import annotations

import math

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import Grid2D


class OccupancyMapping(ProcessingPlugin):
    _manifest = PluginManifest(
        name="occupancy-mapping",
        version="0.1.0",
        consumes=["scan_2d", "registration"],
        produces=["occupancy_geometry"],
        mode="offline",
        description="Project-frame 2D occupancy grid from registered lidar scans",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, res: float = 0.1, **params) -> None:
        regs = ctx.claims("registration")
        if not regs:
            ctx.note("no accepted registration claim; refusing to map in an unaligned frame")
            return
        reg = regs[0]
        r = np.array(reg["payload"]["rotation"])
        t = np.array(reg["payload"]["translation"])
        rot = math.radians(reg["payload"]["rotation_deg"])

        bounds = ctx.project.get("bounds", {"x0": 0.0, "y0": 0.0, "w": 20.0, "h": 20.0})
        grid = Grid2D(bounds["x0"], bounds["y0"], bounds["w"], bounds["h"], res)

        pose_sigmas: list[float] = []
        n_scans = 0
        for obs in ctx.observations("scan_2d"):
            for ev in ctx.evidence_for(obs["id"]):
                p = ctx.payload(ev["id"])
                pose = np.asarray(p["pose_est"], dtype=float)
                origin = r @ pose[:2] + t
                heading = pose[2] + rot
                angles = np.asarray(p["angles"]) + heading
                ranges = np.asarray(p["ranges"])
                hits = np.asarray(p["hit"]).astype(bool)
                ends_x = origin[0] + ranges * np.cos(angles)
                ends_y = origin[1] + ranges * np.sin(angles)
                for ex_, ey_, h in zip(ends_x, ends_y, hits):
                    grid.add_ray(origin[0], origin[1], float(ex_), float(ey_), bool(h))
                n_scans += 1
                pose_sigmas.append(obs.get("quality", {}).get("pose_sigma_xy", 0.0))

        occ = grid.classify()
        grid_ev = ctx.store_derived(
            "occupancy_grid",
            {
                "occ": occ,
                "hits": grid.hits,
                "passes": grid.passes,
                "x0": grid.x0,
                "y0": grid.y0,
                "res": grid.res,
            },
            meta={"frame": "project", "n_scans": n_scans},
        )
        mean_sigma = float(np.mean(pose_sigmas)) if pose_sigmas else 0.0
        confidence = float(
            np.clip(reg["confidence"] * (1.0 - mean_sigma / 0.5), 0.0, 0.99)
        )
        ctx.emit_claim(
            kind="occupancy_geometry",
            payload={
                "evidence_id": grid_ev,
                "res_m": res,
                "n_scans": n_scans,
                "registration_rmse_m": reg["payload"]["rmse_m"],
                "mean_pose_sigma_m": mean_sigma,
                "occupied_cells": int((occ == 1).sum()),
                "free_cells": int((occ == 0).sum()),
                "unknown_cells": int((occ == -1).sum()),
            },
            confidence=confidence,
            evidence_ids=[grid_ev],
            subject="grid:main",
        )
