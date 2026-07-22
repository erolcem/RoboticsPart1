"""Change detection v2 between two registered missions of the same area.

Compares occupancy grids cell-by-cell, but only where BOTH missions
actually observed the cell (coverage-aware: absence of evidence is never
reported as change). Contiguous changed cells become change-region claims
with confidence from observation strength, region size and registration
quality.

v1.0 additions:
* registration-artifact screening - a "changed" region hugging occupied
  structure in the OTHER mission is the classic signature of sub-cell
  misregistration at surface boundaries, not of real change; such regions
  are kept but flagged `likely_registration_artifact` with sharply
  reduced confidence, so review can dismiss them in one glance;
* spatially linked imagery - each region claim carries the depth-camera
  frames whose field of view covered the region in both missions, so a
  reviewer opens the actual supporting pictures from the report;
* zone tagging for site-language reporting.
"""

from __future__ import annotations

import math

import numpy as np

from ..design.zones import zone_of
from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import FREE, OCCUPIED, connected_regions, dilate, region_stats
from .icp import transform_pose


def _imagery_covering(
    ctx: ProcessingContext,
    mission_id: str,
    point: tuple[float, float],
    max_links: int = 3,
) -> list[str]:
    """Evidence ids of depth frames whose FOV covered the point, nearest first."""
    regs = ctx.claims("registration", mission_id=mission_id)
    if not regs:
        return []
    r = np.array(regs[0]["payload"]["rotation"])
    t = np.array(regs[0]["payload"]["translation"])
    candidates: list[tuple[float, str]] = []
    for obs in ctx.observations("depth_image", mission_id=mission_id):
        for ev in ctx.evidence_for(obs["id"]):
            p = ctx.payload(ev["id"])
            pose = transform_pose(r, t, np.asarray(p["pose_est"], dtype=float))
            dx, dy = point[0] - pose[0], point[1] - pose[1]
            dist = math.hypot(dx, dy)
            angles = np.asarray(p["angles"])
            half_fov = float(angles.max() - angles.min()) / 2
            bearing = math.atan2(dy, dx) - pose[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if dist <= 8.0 and abs(bearing) <= half_fov:
                candidates.append((dist, ev["id"]))
    return [ev_id for _, ev_id in sorted(candidates)[:max_links]]


class ChangeDetection(ProcessingPlugin):
    _manifest = PluginManifest(
        name="occupancy-change-detection",
        version="2.0.0",
        consumes=["occupancy_geometry"],
        produces=["change", "change_summary"],
        mode="offline",
        description="Coverage-aware occupancy diff with artifact screening and linked imagery",
        validation="benchmark-validated",
        cross_mission=True,
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(
        self,
        ctx: ProcessingContext,
        baseline_mission_id: str = "",
        min_region_cells: int = 4,
        artifact_adjacency: float = 0.7,
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
        # structure adjacency masks for artifact screening: a disappeared
        # region next to current-mission structure (or appeared next to
        # baseline structure) is likely a surface-boundary registration echo
        near_occ1 = dilate(occ1 == OCCUPIED, 1)
        near_occ2 = dilate(occ2 == OCCUPIED, 1)
        zones = ctx.project.get("zones")

        grid_evidence = [
            cur_claim["payload"]["evidence_id"],
            base_claim["payload"]["evidence_id"],
        ]
        n_regions, n_flagged = 0, 0
        for change_type, mask, structure_near in (
            ("appeared", appeared, near_occ1),
            ("disappeared", disappeared, near_occ2),
        ):
            for cells in connected_regions(mask):
                if len(cells) < min_region_cells:
                    continue
                n_regions += 1
                js, is_ = cells[:, 0], cells[:, 1]
                stats = region_stats(cells, x0, y0, res)
                cx, cy = stats["centroid"]

                s = min(
                    float(np.median(strength1[js, is_])),
                    float(np.median(strength2[js, is_])),
                )
                strength_factor = float(np.clip(s / 10.0, 0.0, 1.0))
                size_factor = 0.6 + 0.4 * float(np.clip(len(cells) / 25.0, 0.0, 1.0))
                confidence = float(
                    np.clip(base_conf * (0.5 + 0.5 * strength_factor) * size_factor, 0.0, 0.99)
                )

                edge_frac = float(structure_near[js, is_].mean())
                artifact = edge_frac >= artifact_adjacency
                if artifact:
                    confidence = float(confidence * 0.4)
                    n_flagged += 1

                # evidential conflict of the region in either mission: high
                # conflict means the sensors themselves disagreed there
                conflict = 0.0
                if "conflict" in g1 and "conflict" in g2:
                    conflict = max(
                        float(np.median(g1["conflict"][js, is_])),
                        float(np.median(g2["conflict"][js, is_])),
                    )

                imagery = {
                    "current": _imagery_covering(ctx, ctx.mission_id, (cx, cy)),
                    "baseline": _imagery_covering(ctx, baseline_mission_id, (cx, cy)),
                }
                ctx.emit_claim(
                    kind="change",
                    payload={
                        "change_type": change_type,
                        "baseline_mission_id": baseline_mission_id,
                        "median_observations": s,
                        "structure_adjacency": edge_frac,
                        "evidential_conflict": conflict,
                        "likely_registration_artifact": bool(artifact),
                        "zone": zone_of(zones, cx, cy),
                        "imagery": imagery,
                        **stats,
                    },
                    confidence=confidence,
                    evidence_ids=grid_evidence + imagery["current"] + imagery["baseline"],
                    subject=f"change:{change_type}:{cx:.1f},{cy:.1f}",
                )

        comparable = float(known.sum() / known.size)
        ctx.emit_claim(
            kind="change_summary",
            payload={
                "baseline_mission_id": baseline_mission_id,
                "n_regions": n_regions,
                "n_flagged_as_artifact": n_flagged,
                "comparable_fraction": comparable,
                "appeared_cells": int(appeared.sum()),
                "disappeared_cells": int(disappeared.sum()),
            },
            confidence=base_conf,
            evidence_ids=grid_evidence,
            subject="change:summary",
        )
