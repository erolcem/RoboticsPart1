"""Coverage analysis: which areas were observed, insufficiently observed or
never seen - so a missing area is never mistaken for an inspected one."""

from __future__ import annotations

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin

UNOBSERVED, INSUFFICIENT, OBSERVED = 0, 1, 2


class CoverageAnalysis(ProcessingPlugin):
    _manifest = PluginManifest(
        name="coverage-analysis",
        version="1.0.0",
        consumes=["occupancy_geometry"],
        produces=["coverage"],
        mode="offline",
        description="Per-cell observation density, overall and per named zone",
        validation="benchmark-validated",
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

        def fractions_of(mask: np.ndarray) -> dict[str, float]:
            n = int(mask.sum())
            if n == 0:
                return {"observed": 0.0, "insufficient": 0.0, "unobserved": 1.0}
            return {
                "observed": float(((cov == OBSERVED) & mask).sum() / n),
                "insufficient": float(((cov == INSUFFICIENT) & mask).sum() / n),
                "unobserved": float(((cov == UNOBSERVED) & mask).sum() / n),
            }

        everywhere = np.ones(cov.shape, dtype=bool)
        fractions = fractions_of(everywhere)

        # per-zone coverage so "did we actually see Room A?" has an answer
        x0, y0, res = float(grid["x0"]), float(grid["y0"]), float(grid["res"])
        ny, nx = cov.shape
        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y0 + (np.arange(ny) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        by_zone = {}
        for name, (zx0, zy0, zx1, zy1) in (ctx.project.get("zones") or {}).items():
            by_zone[name] = fractions_of((xx >= zx0) & (xx <= zx1) & (yy >= zy0) & (yy <= zy1))

        cov_ev = ctx.store_derived(
            "coverage_grid",
            {"coverage": cov, "x0": grid["x0"], "y0": grid["y0"], "res": grid["res"]},
            meta={"frame": "project", "strong_obs_threshold": strong_obs},
        )
        ctx.emit_claim(
            kind="coverage",
            payload={
                "evidence_id": cov_ev,
                "fractions": fractions,
                "by_zone": by_zone,
                "strong_obs": strong_obs,
            },
            confidence=geom["confidence"],
            evidence_ids=[cov_ev, geom["payload"]["evidence_id"]],
            subject="coverage:main",
        )
