"""Registration to the project coordinate frame using surveyed control points.

Per the proposal's risk mitigation: do not rely solely on unconstrained
SLAM - use project control points and report measured residuals. Fits a
2D rigid transform (Kabsch) between where fiducials appeared in the
mission's estimated frame and their surveyed project-frame coordinates,
and reports RMSE residuals as the registration uncertainty.
"""

from __future__ import annotations

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (R, t) with dst ~ R @ src + t."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    h = (src - sc).T @ (dst - dc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    t = dc - r @ sc
    return r, t


class ControlPointRegistration(ProcessingPlugin):
    _manifest = PluginManifest(
        name="control-point-registration",
        version="0.1.0",
        consumes=["fiducial_detection"],
        produces=["registration", "trajectory"],
        mode="offline",
        description="Rigid alignment of a mission to the project frame via surveyed fiducials",
        validation="simulation-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, min_points: int = 3, **params) -> None:
        control_points: dict = ctx.project.get("control_points", {})
        detections: dict[str, list[np.ndarray]] = {}
        for obs in ctx.observations("fiducial_detection"):
            for ev in ctx.evidence_for(obs["id"]):
                payload = ctx.payload(ev["id"])
                fid = str(payload["fiducial_id"])
                detections.setdefault(fid, []).append(np.asarray(payload["position_est"]))

        matched = sorted(fid for fid in detections if fid in control_points)
        if len(matched) < min_points:
            ctx.note(
                f"only {len(matched)} control points observed (need {min_points}); "
                "mission cannot be registered - flagging instead of guessing"
            )
            ctx.emit_claim(
                kind="registration",
                payload={"status": "insufficient_control_points", "matched": matched},
                confidence=0.0,
                subject="registration:project-frame",
                status="rejected",
            )
            return

        src = np.array([np.mean(detections[fid], axis=0) for fid in matched])
        dst = np.array([control_points[fid] for fid in matched])
        r, t = _rigid_fit(src, dst)
        residuals = np.linalg.norm((src @ r.T + t) - dst, axis=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        rotation_deg = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        confidence = float(np.clip(1.0 - rmse / 0.2, 0.0, 0.99))

        ctx.emit_claim(
            kind="registration",
            payload={
                "rotation": r.tolist(),
                "translation": t.tolist(),
                "rotation_deg": rotation_deg,
                "rmse_m": rmse,
                "n_control_points": len(matched),
                "per_point_residuals_m": {f: float(x) for f, x in zip(matched, residuals)},
                "n_detections": int(sum(len(v) for v in detections.values())),
            },
            confidence=confidence,
            subject="registration:project-frame",
        )

        # corrected project-frame trajectory as derived evidence + claim
        poses, times, sigmas = [], [], []
        for obs in ctx.observations("pose_estimate"):
            for ev in ctx.evidence_for(obs["id"]):
                payload = ctx.payload(ev["id"])
                p = np.asarray(payload["pose_est"], dtype=float)
                xy = r @ p[:2] + t
                poses.append([xy[0], xy[1], p[2] + np.radians(rotation_deg)])
                times.append(obs["t"])
                sigmas.append(obs.get("quality", {}).get("pose_sigma_xy", 0.0))
        if poses:
            traj_ev = ctx.store_derived(
                "trajectory",
                {"poses": np.array(poses), "t": np.array(times), "sigma_xy": np.array(sigmas)},
                meta={"frame": "project"},
            )
            ctx.emit_claim(
                kind="trajectory",
                payload={
                    "evidence_id": traj_ev,
                    "n_poses": len(poses),
                    "max_pose_sigma_m": float(np.max(sigmas)) if sigmas else 0.0,
                    "path_length_m": float(
                        np.sum(np.linalg.norm(np.diff(np.array(poses)[:, :2], axis=0), axis=1))
                    ),
                },
                confidence=confidence,
                evidence_ids=[traj_ev],
                subject="trajectory:project-frame",
            )
