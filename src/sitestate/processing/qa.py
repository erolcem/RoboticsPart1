"""Quality-assurance plug-ins (v1.0): the platform checking itself.

RegistrationVerification - after two missions are independently registered
to the project frame via control points, their maps should coincide. ICP
between the two occupied point sets measures any residual misalignment;
a residual beyond tolerance flags the registration chain for review
instead of letting downstream change detection blame the site.

CalibrationCheck - per-sensor bias of fiducial detections against the
surveyed coordinates (after registration). A drifted or knocked sensor
shows up as a systematic offset long before it ruins a dataset; the claim
carries the measured bias so the calibration-uncertainty dimension is a
number, not a hope.
"""

from __future__ import annotations

import math

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .grid import OCCUPIED
from .icp import PointHash, icp_2d


def _occupied_points(grid: dict) -> np.ndarray:
    occ = grid["occ"]
    js, is_ = np.nonzero(occ == OCCUPIED)
    x0, y0, res = float(grid["x0"]), float(grid["y0"]), float(grid["res"])
    return np.stack([x0 + (is_ + 0.5) * res, y0 + (js + 0.5) * res], axis=1)


class RegistrationVerification(ProcessingPlugin):
    _manifest = PluginManifest(
        name="registration-verification",
        version="1.0.0",
        consumes=["occupancy_geometry"],
        produces=["alignment_check"],
        mode="offline",
        description="Cross-mission ICP consistency check of independent registrations",
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
        tolerance_m: float = 0.1,
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
        g2 = ctx.payload(cur[0]["payload"]["evidence_id"])
        g1 = ctx.payload(base[0]["payload"]["evidence_id"])
        p2, p1 = _occupied_points(g2), _occupied_points(g1)
        if len(p1) < 50 or len(p2) < 50:
            ctx.note("too few occupied cells to verify alignment")
            return

        r, t, rmse, n_pairs = icp_2d(p2, PointHash(p1, 0.3), max_pair_dist=0.4)
        if not math.isfinite(rmse):
            ctx.emit_claim(
                kind="alignment_check",
                payload={"status": "unverifiable", "n_pairs": n_pairs,
                         "baseline_mission_id": baseline_mission_id},
                confidence=0.1,
                evidence_ids=[cur[0]["payload"]["evidence_id"],
                              base[0]["payload"]["evidence_id"]],
                subject="alignment:cross-mission",
            )
            return

        residual_t = float(np.linalg.norm(t))
        residual_deg = float(abs(math.degrees(math.atan2(r[1, 0], r[0, 0]))))
        within = residual_t <= tolerance_m and residual_deg <= 1.0
        ctx.emit_claim(
            kind="alignment_check",
            payload={
                "baseline_mission_id": baseline_mission_id,
                "residual_translation_m": residual_t,
                "residual_rotation_deg": residual_deg,
                "icp_rmse_m": rmse,
                "n_pairs": n_pairs,
                "tolerance_m": tolerance_m,
                "within_tolerance": bool(within),
            },
            confidence=float(np.clip(1.0 - residual_t / (2 * tolerance_m), 0.05, 0.99)),
            evidence_ids=[cur[0]["payload"]["evidence_id"],
                          base[0]["payload"]["evidence_id"]],
            subject="alignment:cross-mission",
        )
        if not within:
            ctx.note(
                f"cross-mission alignment residual {residual_t * 100:.1f} cm exceeds "
                f"tolerance - registration chain needs review before trusting changes"
            )


class CalibrationCheck(ProcessingPlugin):
    _manifest = PluginManifest(
        name="calibration-check",
        version="1.0.0",
        consumes=["fiducial_detection", "registration"],
        produces=["calibration_check"],
        mode="offline",
        description="Per-sensor systematic bias of fiducial detections vs surveyed points",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, bias_tolerance_m: float = 0.05, **params) -> None:
        regs = ctx.claims("registration")
        if not regs:
            ctx.note("no accepted registration claim")
            return
        reg = regs[0]
        r = np.array(reg["payload"]["rotation"])
        t = np.array(reg["payload"]["translation"])
        control_points: dict = ctx.project.get("control_points", {})

        # bias must be measured in the SENSOR frame: a constant mounting or
        # calibration offset rotates with the carrier in world coordinates
        # and would average itself away over a looping mission
        from .icp import transform_pose

        residuals_by_sensor: dict[str, list[np.ndarray]] = {}
        for obs in ctx.observations("fiducial_detection"):
            for ev in ctx.evidence_for(obs["id"]):
                payload = ctx.payload(ev["id"])
                fid = str(payload["fiducial_id"])
                if fid not in control_points:
                    continue
                surveyed = np.asarray(control_points[fid], dtype=float)
                if "pose_est" in payload and "relative" in payload:
                    pose = transform_pose(r, t, np.asarray(payload["pose_est"], dtype=float))
                    c, s = math.cos(-pose[2]), math.sin(-pose[2])
                    d = surveyed - pose[:2]
                    expected_rel = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
                    residual = np.asarray(payload["relative"], dtype=float) - expected_rel
                else:  # adapters without pose context: world-frame fallback
                    projected = r @ np.asarray(payload["position_est"], dtype=float) + t
                    residual = projected - surveyed
                residuals_by_sensor.setdefault(obs["sensor_id"], []).append(residual)

        for sensor_id, residuals in residuals_by_sensor.items():
            arr = np.array(residuals)
            bias = arr.mean(axis=0)
            bias_mag = float(np.linalg.norm(bias))
            spread = float(arr.std(axis=0).mean())
            ok = bias_mag <= bias_tolerance_m
            sensor = ctx.ledger.sensor(sensor_id) or {}
            ctx.emit_claim(
                kind="calibration_check",
                payload={
                    "sensor_id": sensor_id,
                    "sensor_name": (sensor.get("manifest") or {}).get("name", "?"),
                    "calibration_version": (sensor.get("manifest") or {}).get(
                        "calibration_version", "?"
                    ),
                    "bias_m": [float(bias[0]), float(bias[1])],
                    "bias_magnitude_m": bias_mag,
                    "residual_spread_m": spread,
                    "n_detections": int(len(arr)),
                    "tolerance_m": bias_tolerance_m,
                    "within_tolerance": bool(ok),
                },
                confidence=float(np.clip(1.0 - bias_mag / (2 * bias_tolerance_m), 0.05, 0.99)),
                subject=f"calibration:{sensor_id}",
            )
            if not ok:
                ctx.note(
                    f"sensor {sensor_id} shows {bias_mag * 100:.1f} cm systematic bias - "
                    f"calibration likely stale"
                )
