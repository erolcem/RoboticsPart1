"""Change detection between two registered missions of the same area.

Compares the occupancy grids of the current mission and a baseline
mission cell-by-cell, but only where BOTH missions actually observed the
cell (coverage-aware: absence of evidence is never reported as change).
Contiguous changed cells become change-region claims with confidence
derived from observation strength and registration quality, each linked
to the grid evidence of both missions for review.
"""

from __future__ import annotations

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import FREE, OCCUPIED, connected_regions


class ChangeDetection(ProcessingPlugin):
    _manifest = PluginManifest(
        name="occupancy-change-detection",
        version="0.1.0",
        consumes=["occupancy_geometry"],
        produces=["change", "change_summary"],
        mode="offline",
        description="Coverage-aware occupancy diff between two missions",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(
        self,
        ctx: ProcessingContext,
        baseline_mission_id: str = "",
        min_region_cells: int = 4,
        **params,
    ) -> None:
        if not baseline_mission_id:
            ctx.note("baseline_mission_id parameter is required")
            return
        cur = ctx.claims("occupancy_geometry")
        base = ctx.claims("occupancy_geometry", mission_id=baseline_mission_id)
        if not cur or not base:
            ctx.note("both missions need an accepted occupancy_geometry claim")
            return
        cur_claim, base_claim = cur[0], base[0]
        g2 = ctx.payload(cur_claim["payload"]["evidence_id"])
        g1 = ctx.payload(base_claim["payload"]["evidence_id"])
        if g1["occ"].shape != g2["occ"].shape or float(g1["res"]) != float(g2["res"]):
            ctx.note("grids are not comparable (different bounds/resolution)")
            return

        occ1, occ2 = g1["occ"], g2["occ"]
        known = (occ1 != -1) & (occ2 != -1)
        appeared = known & (occ1 == FREE) & (occ2 == OCCUPIED)
        disappeared = known & (occ1 == OCCUPIED) & (occ2 == FREE)

        res = float(g2["res"])
        x0, y0 = float(g2["x0"]), float(g2["y0"])
        base_conf = min(cur_claim["confidence"], base_claim["confidence"])
        strength1 = g1["hits"] + g1["passes"]
        strength2 = g2["hits"] + g2["passes"]

        evidence_ids = [
            cur_claim["payload"]["evidence_id"],
            base_claim["payload"]["evidence_id"],
        ]
        n_regions = 0
        for change_type, mask in (("appeared", appeared), ("disappeared", disappeared)):
            for cells in connected_regions(mask):
                if len(cells) < min_region_cells:
                    continue
                n_regions += 1
                js, is_ = cells[:, 0], cells[:, 1]
                cx = x0 + (is_.mean() + 0.5) * res
                cy = y0 + (js.mean() + 0.5) * res
                # observation strength inside the region, both missions
                s = min(
                    float(np.median(strength1[js, is_])),
                    float(np.median(strength2[js, is_])),
                )
                strength_factor = float(np.clip(s / 10.0, 0.0, 1.0))
                # small regions near noise/registration limits get visibly
                # lower confidence instead of being silently dropped
                size_factor = 0.6 + 0.4 * float(np.clip(len(cells) / 25.0, 0.0, 1.0))
                confidence = float(
                    np.clip(base_conf * (0.5 + 0.5 * strength_factor) * size_factor, 0.0, 0.99)
                )
                ctx.emit_claim(
                    kind="change",
                    payload={
                        "change_type": change_type,
                        "centroid": [cx, cy],
                        "bbox": [
                            x0 + is_.min() * res,
                            y0 + js.min() * res,
                            x0 + (is_.max() + 1) * res,
                            y0 + (js.max() + 1) * res,
                        ],
                        "n_cells": int(len(cells)),
                        "area_m2": float(len(cells) * res * res),
                        "baseline_mission_id": baseline_mission_id,
                        "median_observations": s,
                    },
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                    subject=f"change:{change_type}:{cx:.1f},{cy:.1f}",
                )

        comparable = float(known.sum() / known.size)
        ctx.emit_claim(
            kind="change_summary",
            payload={
                "baseline_mission_id": baseline_mission_id,
                "n_regions": n_regions,
                "comparable_fraction": comparable,
                "appeared_cells": int(appeared.sum()),
                "disappeared_cells": int(disappeared.sum()),
            },
            confidence=base_conf,
            evidence_ids=evidence_ids,
            subject="change:summary",
        )
