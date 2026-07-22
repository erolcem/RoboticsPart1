"""Occupancy mapping v2: registered 2D scans fused into a project-frame
log-odds grid with an evidential (Dempster-Shafer) uncertainty layer.

Consumes lidar scans plus the mission's registration claim; when a
`pose_corrections` claim exists (from the pose-graph refinement plug-in)
the optimized per-scan poses are used instead of the coarse ones - a soft
dependency: the plug-in never fails for lack of refinement, it just maps
more sharply when refinement ran first.

Confidence combines registration quality, mean pose uncertainty and the
map's own decision entropy. The stored grid carries per-cell probability,
plus evidential ignorance (epistemic: never saw enough) and conflict
(evidence disagrees: dynamics or misregistration) - see docs/sota-review.md.
"""

from __future__ import annotations

import math

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import Grid2D
from .icp import transform_pose


class OccupancyMapping(ProcessingPlugin):
    _manifest = PluginManifest(
        name="occupancy-mapping",
        version="2.0.0",
        consumes=["scan_2d", "registration"],
        produces=["occupancy_geometry"],
        mode="offline",
        description="Project-frame log-odds occupancy grid from registered lidar scans",
        validation="benchmark-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(
        self,
        ctx: ProcessingContext,
        res: float = 0.1,
        l_occ: float = 1.2,
        l_free: float = -0.35,
        **params,
    ) -> None:
        regs = ctx.claims("registration")
        if not regs:
            ctx.note("no accepted registration claim; refusing to map in an unaligned frame")
            return
        reg = regs[0]
        r = np.array(reg["payload"]["rotation"])
        t = np.array(reg["payload"]["translation"])

        # optional refined poses (project frame) keyed by observation id
        refined: dict[str, np.ndarray] = {}
        used_corrections = False
        corr_claims = ctx.claims("pose_corrections")
        if corr_claims:
            corr = ctx.payload(corr_claims[0]["payload"]["evidence_id"])
            refined = {
                str(oid): pose for oid, pose in zip(corr["obs_ids"], corr["pose_after"])
            }
            used_corrections = True

        bounds = ctx.project.get("bounds", {"x0": 0.0, "y0": 0.0, "w": 20.0, "h": 20.0})
        grid = Grid2D(bounds["x0"], bounds["y0"], bounds["w"], bounds["h"], res)

        pose_sigmas: list[float] = []
        n_scans = 0
        for obs in ctx.observations("scan_2d"):
            for ev in ctx.evidence_for(obs["id"]):
                p = ctx.payload(ev["id"])
                if obs["id"] in refined:
                    pose = refined[obs["id"]]
                else:
                    pose = transform_pose(r, t, np.asarray(p["pose_est"], dtype=float))
                angles = np.asarray(p["angles"]) + pose[2]
                ranges = np.asarray(p["ranges"])
                hits = np.asarray(p["hit"]).astype(bool)
                ends = np.stack(
                    [pose[0] + ranges * np.cos(angles), pose[1] + ranges * np.sin(angles)],
                    axis=1,
                )
                grid.integrate_scan((pose[0], pose[1]), ends, hits)
                n_scans += 1
                pose_sigmas.append(obs.get("quality", {}).get("pose_sigma_xy", 0.0))

        occ = grid.classify(l_occ, l_free)
        prob = grid.probability(l_occ, l_free).astype(np.float32)
        entropy = grid.decision_entropy(l_occ, l_free)
        ev_layer = grid.evidential()
        grid_ev = ctx.store_derived(
            "occupancy_grid",
            {
                "occ": occ,
                "prob": prob,
                "hits": grid.hits,
                "passes": grid.passes,
                "ignorance": ev_layer["ignorance"],
                "conflict": ev_layer["conflict"],
                "x0": grid.x0,
                "y0": grid.y0,
                "res": grid.res,
            },
            meta={"frame": "project", "n_scans": n_scans, "sensor_model": [l_occ, l_free]},
        )
        mean_sigma = float(np.mean(pose_sigmas)) if pose_sigmas else 0.0
        confidence = float(
            np.clip(
                reg["confidence"] * (1.0 - mean_sigma / 0.5) * (1.0 - 0.5 * entropy),
                0.0,
                0.99,
            )
        )
        ctx.emit_claim(
            kind="occupancy_geometry",
            payload={
                "evidence_id": grid_ev,
                "res_m": res,
                "n_scans": n_scans,
                "registration_rmse_m": reg["payload"]["rmse_m"],
                "mean_pose_sigma_m": mean_sigma,
                "decision_entropy": entropy,
                "mean_ignorance": float(ev_layer["ignorance"].mean()),
                "mean_conflict": float(ev_layer["conflict"].mean()),
                "high_conflict_cells": int((ev_layer["conflict"] > 0.2).sum()),
                "used_pose_corrections": used_corrections,
                "occupied_cells": int((occ == 1).sum()),
                "free_cells": int((occ == 0).sum()),
                "unknown_cells": int((occ == -1).sum()),
            },
            confidence=confidence,
            evidence_ids=[grid_ev],
            subject="grid:main",
        )
