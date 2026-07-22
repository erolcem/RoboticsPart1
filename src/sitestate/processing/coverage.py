"""Coverage analysis: which areas were observed, insufficiently observed or
never seen - so a missing area is never mistaken for an inspected one."""

from __future__ import annotations

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin

UNOBSERVED, INSUFFICIENT, OBSERVED = 0, 1, 2


class CoverageAnalysis(ProcessingPlugin):
    _manifest = PluginManifest(
        name="coverage-analysis",
        version="0.1.0",
        consumes=["occupancy_geometry"],
        produces=["coverage"],
        mode="offline",
        description="Per-cell observation density from the occupancy grid evidence",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, strong_obs: int = 5, **params) -> None:
        geoms = ctx.claims("occupancy_geometry")
        if not geoms:
            ctx.note("no occupancy_geometry claim available")
            return
        geom = geoms[0]
        grid = ctx.payload(geom["payload"]["evidence_id"])
        total = grid["hits"] + grid["passes"]

        cov = np.full(total.shape, UNOBSERVED, dtype=np.int8)
        cov[total >= 1] = INSUFFICIENT
        cov[total >= strong_obs] = OBSERVED

        n = cov.size
        fractions = {
            "observed": float((cov == OBSERVED).sum() / n),
            "insufficient": float((cov == INSUFFICIENT).sum() / n),
            "unobserved": float((cov == UNOBSERVED).sum() / n),
        }
        cov_ev = ctx.store_derived(
            "coverage_grid",
            {"coverage": cov, "x0": grid["x0"], "y0": grid["y0"], "res": grid["res"]},
            meta={"frame": "project", "strong_obs_threshold": strong_obs},
        )
        ctx.emit_claim(
            kind="coverage",
            payload={"evidence_id": cov_ev, "fractions": fractions, "strong_obs": strong_obs},
            confidence=geom["confidence"],
            evidence_ids=[cov_ev, geom["payload"]["evidence_id"]],
            subject="coverage:main",
        )
