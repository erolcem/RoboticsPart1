"""Progress tracking (v2): per-zone completion of the designed plan.

The construction-monitoring literature (docs/sota-review.md §3) converges
on one customer-facing number: element-wise completion of the as-planned
model. This plug-in computes its grid-level analogue - of the designed
elements whose location was actually observed, what fraction exists
as-built - overall and per named zone. Trended across missions it becomes
the progress curve; the observed-fraction is reported alongside so a
low number is never mistaken for demolition when it is really a coverage
gap (the platform's honesty rule, applied to schedules).
"""

from __future__ import annotations

import numpy as np

from ..design.floorplan import FloorPlan
from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import OCCUPIED, dilate

OBSERVED = 2  # coverage code


class ProgressTracking(ProcessingPlugin):
    _manifest = PluginManifest(
        name="progress-tracking",
        version="2.0.0",
        consumes=["occupancy_geometry", "coverage"],
        produces=["progress"],
        mode="offline",
        description="Per-zone as-built completion of the designed plan",
        validation="benchmark-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, tolerance_m: float = 0.2, **params) -> None:
        design = ctx.project.get("design")
        if not design:
            ctx.note("project has no design/floor-plan; nothing to measure progress against")
            return
        geoms = ctx.claims("occupancy_geometry")
        covs = ctx.claims("coverage")
        if not geoms or not covs:
            ctx.note("needs accepted occupancy_geometry and coverage claims")
            return
        geom, cov_claim = geoms[0], covs[0]
        g = ctx.payload(geom["payload"]["evidence_id"])
        c = ctx.payload(cov_claim["payload"]["evidence_id"])
        occ, coverage = g["occ"], c["coverage"]
        x0, y0, res = float(g["x0"]), float(g["y0"]), float(g["res"])
        ny, nx = occ.shape

        plan = FloorPlan.from_dict(design)
        plan_mask = plan.rasterize(x0, y0, nx, ny, res)
        tol_cells = max(1, round(tolerance_m / res))
        occ_band = dilate(occ == OCCUPIED, tol_cells)
        observed_plan = plan_mask & (coverage == OBSERVED)
        built = observed_plan & occ_band

        def completion(mask: np.ndarray) -> dict:
            n_plan = int((plan_mask & mask).sum())
            n_obs = int((observed_plan & mask).sum())
            n_built = int((built & mask).sum())
            return {
                "completion": n_built / n_obs if n_obs else None,
                "observed_fraction": n_obs / n_plan if n_plan else None,
                "planned_cells": n_plan,
                "built_cells": n_built,
            }

        everywhere = np.ones(occ.shape, dtype=bool)
        overall = completion(everywhere)

        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y0 + (np.arange(ny) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        by_zone = {
            name: completion((xx >= zx0) & (xx <= zx1) & (yy >= zy0) & (yy <= zy1))
            for name, (zx0, zy0, zx1, zy1) in (ctx.project.get("zones") or {}).items()
        }

        ctx.emit_claim(
            kind="progress",
            payload={
                "plan_name": plan.name,
                "overall_completion": overall["completion"],
                "observed_plan_fraction": overall["observed_fraction"],
                "by_zone": by_zone,
                "tolerance_m": tolerance_m,
            },
            confidence=min(geom["confidence"], cov_claim["confidence"]),
            evidence_ids=[geom["payload"]["evidence_id"],
                          cov_claim["payload"]["evidence_id"]],
            subject="progress:plan",
        )
