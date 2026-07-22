"""Semantic labeling of occupied structures (a first Phase 4 semantic step).

Clusters occupied cells into entities and assigns class *probabilities*
(never a bare label) from simple geometric features and plan overlap:

* planned_structure  - matches the designed floor plan
* unplanned_partition - wall-like but not in the plan (e.g. temporary wall)
* movable_object     - compact free-standing blob (pallet, stack, ...)

This heuristic model is intentionally the weakest link and the clearest
candidate for replacement by a learned model: swap it by registering a new
plug-in with the same `produces=["entity"]` - both interpretations then
coexist as competing claims until reviewed.
"""

from __future__ import annotations

import numpy as np

from ..design.floorplan import FloorPlan
from ..design.zones import zone_of
from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import OCCUPIED, connected_regions, dilate, region_stats


class SemanticLabeling(ProcessingPlugin):
    _manifest = PluginManifest(
        name="semantic-labeling",
        version="1.0.0",
        consumes=["occupancy_geometry"],
        produces=["entity"],
        mode="offline",
        description="Heuristic entity classification with class probabilities",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, min_region_cells: int = 3, **params) -> None:
        geoms = ctx.claims("occupancy_geometry")
        if not geoms:
            ctx.note("no occupancy_geometry claim available")
            return
        geom = geoms[0]
        g = ctx.payload(geom["payload"]["evidence_id"])
        occ = g["occ"]
        x0, y0, res = float(g["x0"]), float(g["y0"]), float(g["res"])
        ny, nx = occ.shape

        plan_band = None
        design = ctx.project.get("design")
        if design:
            plan = FloorPlan.from_dict(design)
            plan_band = dilate(plan.rasterize(x0, y0, nx, ny, res), max(1, round(0.2 / res)))

        for cells in connected_regions(occ == OCCUPIED):
            if len(cells) < min_region_cells:
                continue
            stats = region_stats(cells, x0, y0, res)
            js, is_ = cells[:, 0], cells[:, 1]
            bw = (is_.max() - is_.min() + 1) * res
            bh = (js.max() - js.min() + 1) * res
            thin = min(bw, bh) <= 0.35 and max(bw, bh) / max(min(bw, bh), res) >= 3.0
            plan_overlap = float(plan_band[js, is_].mean()) if plan_band is not None else 0.0

            scores = {
                "planned_structure": max(plan_overlap, 0.05),
                "unplanned_partition": (1.0 - plan_overlap) * (1.0 if thin else 0.2),
                "movable_object": (1.0 - plan_overlap) * (0.15 if thin else 1.0),
            }
            total = sum(scores.values())
            probs = {k: v / total for k, v in scores.items()}
            top_class = max(probs, key=probs.get)

            ctx.emit_claim(
                kind="entity",
                payload={
                    "top_class": top_class,
                    "class_probs": probs,
                    "plan_overlap": plan_overlap,
                    "extent_m": [float(bw), float(bh)],
                    "zone": zone_of(ctx.project.get("zones"),
                                    stats["centroid"][0], stats["centroid"][1]),
                    **stats,
                },
                confidence=float(np.clip(probs[top_class] * geom["confidence"], 0.0, 0.99)),
                evidence_ids=[geom["payload"]["evidence_id"]],
                subject=f"entity:{stats['centroid'][0]:.1f},{stats['centroid'][1]:.1f}",
            )
