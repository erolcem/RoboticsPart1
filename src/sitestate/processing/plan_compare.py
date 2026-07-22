"""As-built vs designed-state comparison (Phase 3).

Compares the observed occupancy grid against the project's floor plan and
emits `deviation` claims:

* built_not_planned - something occupies space where the plan has nothing
  (within a tolerance band that absorbs registration error).
* planned_not_built - a planned wall is missing where the area was well
  observed as free.

Coverage-aware like change detection: cells that were not sufficiently
observed are never reported as deviations.
"""

from __future__ import annotations

import numpy as np

from ..design.floorplan import FloorPlan
from ..design.zones import zone_of
from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import OCCUPIED, connected_regions, dilate, region_stats

OBSERVED = 2  # coverage code from CoverageAnalysis


class PlanComparison(ProcessingPlugin):
    _manifest = PluginManifest(
        name="plan-comparison",
        version="1.0.0",
        consumes=["occupancy_geometry", "coverage"],
        produces=["deviation", "deviation_summary"],
        mode="offline",
        description="Coverage-aware diff of as-built occupancy against the designed floor plan",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(
        self,
        ctx: ProcessingContext,
        tolerance_m: float = 0.2,
        min_region_cells: int = 4,
        **params,
    ) -> None:
        design = ctx.project.get("design")
        if not design:
            ctx.note("project has no design/floor-plan; nothing to compare against")
            return
        geoms = ctx.claims("occupancy_geometry")
        covs = ctx.claims("coverage")
        if not geoms or not covs:
            ctx.note("needs accepted occupancy_geometry and coverage claims")
            return
        geom, cov = geoms[0], covs[0]
        g = ctx.payload(geom["payload"]["evidence_id"])
        c = ctx.payload(cov["payload"]["evidence_id"])
        occ, coverage = g["occ"], c["coverage"]
        x0, y0, res = float(g["x0"]), float(g["y0"]), float(g["res"])
        ny, nx = occ.shape

        plan = FloorPlan.from_dict(design)
        plan_mask = plan.rasterize(x0, y0, nx, ny, res)
        tol_cells = max(1, round(tolerance_m / res))
        plan_band = dilate(plan_mask, tol_cells)
        occ_mask = occ == OCCUPIED
        occ_band = dilate(occ_mask, tol_cells)
        observed = coverage == OBSERVED

        built_not_planned = occ_mask & ~plan_band & observed
        planned_not_built = plan_mask & ~occ_band & observed

        plan_ev = ctx.store_derived(
            "design_grid",
            {"plan": plan_mask.astype(np.int8), "x0": x0, "y0": y0, "res": res},
            meta={"plan_name": plan.name, "tolerance_m": tolerance_m},
        )
        evidence_ids = [geom["payload"]["evidence_id"], cov["payload"]["evidence_id"], plan_ev]

        n_regions = 0
        for dev_type, mask in (
            ("built_not_planned", built_not_planned),
            ("planned_not_built", planned_not_built),
        ):
            for cells in connected_regions(mask):
                if len(cells) < min_region_cells:
                    continue
                n_regions += 1
                stats = region_stats(cells, x0, y0, res)
                size_factor = 0.6 + 0.4 * float(np.clip(len(cells) / 25.0, 0.0, 1.0))
                confidence = float(np.clip(geom["confidence"] * size_factor, 0.0, 0.99))
                ctx.emit_claim(
                    kind="deviation",
                    payload={
                        "deviation_type": dev_type,
                        "plan_name": plan.name,
                        "zone": zone_of(ctx.project.get("zones"),
                                        stats["centroid"][0], stats["centroid"][1]),
                        **stats,
                    },
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                    subject=(
                        f"deviation:{dev_type}:"
                        f"{stats['centroid'][0]:.1f},{stats['centroid'][1]:.1f}"
                    ),
                )

        ctx.emit_claim(
            kind="deviation_summary",
            payload={
                "plan_name": plan.name,
                "n_regions": n_regions,
                "tolerance_m": tolerance_m,
                "built_not_planned_cells": int(built_not_planned.sum()),
                "planned_not_built_cells": int(planned_not_built.sum()),
            },
            confidence=geom["confidence"],
            evidence_ids=evidence_ids,
            subject="deviation:summary",
        )
