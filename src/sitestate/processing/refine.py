"""Pose-graph trajectory optimization (v2).

The standard modern SLAM back-end architecture (LIO-SAM lineage; see
docs/sota-review.md) scaled to the platform's 2D representation: every
scan pose is a variable in one nonlinear least-squares problem combining

* odometry factors   - relative motion between consecutive poses from the
                       carrier's own estimate (locally accurate, drifts);
* landmark factors   - fiducial detections against surveyed control
                       points (absolute, sparse);
* scan-match factors - scan-to-map ICP results (dense, occasionally
                       wrong), robustified with a Huber kernel so a bad
                       match is down-weighted instead of dragging the
                       whole trajectory.

Solved by damped Gauss-Newton with analytic Jacobians; poses whose final
correction exceeds a sanity bound revert to their coarse estimate and are
counted as unrefined. Emits the same `pose_corrections` claim contract as
v1, so occupancy mapping and everything downstream is unchanged - the
back-end got smarter behind a stable boundary, which is the point of the
plug-in architecture.
"""

from __future__ import annotations

import math

import numpy as np

from ..plugins.base import PluginManifest, ProcessingContext, ProcessingPlugin
from .icp import PointHash, icp_2d, transform_pose


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(a), np.cos(a))


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _drot_T(theta: float) -> np.ndarray:
    """d(R(theta)^T)/dtheta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[-s, c], [-c, -s]])


def _scan_points_world(payload: dict, pose: np.ndarray) -> np.ndarray:
    hits = np.asarray(payload["hit"]).astype(bool)
    angles = np.asarray(payload["angles"])[hits] + pose[2]
    ranges = np.asarray(payload["ranges"])[hits]
    return np.stack(
        [pose[0] + ranges * np.cos(angles), pose[1] + ranges * np.sin(angles)], axis=1
    )


class PoseGraph:
    """Dense 2D pose-graph solver (N poses -> 3N variables)."""

    def __init__(self, initial: np.ndarray):
        self.poses = initial.copy()  # (N, 3)
        self.odo: list[tuple[int, int, np.ndarray, float, float]] = []
        self.landmarks: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        self.absolute: list[tuple[int, np.ndarray, float, float]] = []

    def add_odometry(self, i: int, j: int, z: np.ndarray,
                     w_xy: float, w_th: float) -> None:
        self.odo.append((i, j, z, w_xy, w_th))

    def add_landmark(self, i: int, landmark_xy: np.ndarray, measured_rel: np.ndarray,
                     w: float) -> None:
        self.landmarks.append((i, landmark_xy, measured_rel, w))

    def add_absolute(self, i: int, measured_pose: np.ndarray,
                     w_xy: float, w_th: float) -> None:
        self.absolute.append((i, measured_pose, w_xy, w_th))

    def _absolute_weights(self, kernel: str, k: float, mu: float) -> np.ndarray:
        """Per-factor robust weights for the scan-match (absolute) factors.

        'gnc': graduated non-convexity with a Geman-McClure kernel
        (Yang et al., arXiv:1909.08605) - starts near-convex (mu large,
        weights ~1) and anneals toward the true non-convex kernel, which
        REJECTS gross outliers rather than merely dampening them.
        'huber': classic IRLS fallback, kept for comparison.
        """
        r2 = np.array([
            (self.poses[i][0] - m[0]) ** 2 + (self.poses[i][1] - m[1]) ** 2
            for i, m, _, _ in self.absolute
        ])
        if kernel == "huber":
            norm = np.sqrt(r2)
            return np.where(norm <= k, 1.0, k / np.maximum(norm, 1e-12))
        mu_k2 = mu * k * k
        return (mu_k2 / (r2 + mu_k2)) ** 2

    def optimize(self, iterations: int = 12, damping: float = 1e-6,
                 huber_k: float = 0.15, kernel: str = "gnc") -> tuple[int, float]:
        n = len(self.poses)
        cost = float("inf")
        # GNC schedule: start near-convex, anneal the control parameter
        mu = 1.0
        if kernel == "gnc" and self.absolute:
            r2_max = max(
                (self.poses[i][0] - m[0]) ** 2 + (self.poses[i][1] - m[1]) ** 2
                for i, m, _, _ in self.absolute
            )
            mu = max(1.0, 2.0 * r2_max / (huber_k * huber_k))
        for it in range(iterations):
            abs_weights = self._absolute_weights(kernel, huber_k, mu)
            h = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)
            cost = 0.0

            for i, j, z, w_xy, w_th in self.odo:
                pi, pj = self.poses[i], self.poses[j]
                rt = _rot(pi[2]).T
                r_xy = rt @ (pj[:2] - pi[:2]) - z[:2]
                r_th = float(_wrap(pj[2] - pi[2] - z[2]))
                ji = np.zeros((3, 3))
                jj = np.zeros((3, 3))
                ji[:2, :2] = -rt
                ji[:2, 2] = _drot_T(pi[2]) @ (pj[:2] - pi[:2])
                ji[2, 2] = -1.0
                jj[:2, :2] = rt
                jj[2, 2] = 1.0
                w = np.diag([w_xy, w_xy, w_th])
                r = np.array([r_xy[0], r_xy[1], r_th])
                cost += float(r @ w @ r)
                for a, ja in ((i, ji), (j, jj)):
                    b[3 * a:3 * a + 3] += ja.T @ w @ r
                    for c, jc in ((i, ji), (j, jj)):
                        h[3 * a:3 * a + 3, 3 * c:3 * c + 3] += ja.T @ w @ jc

            for i, lm, rel, w_s in self.landmarks:
                pi = self.poses[i]
                rt = _rot(pi[2]).T
                r = rt @ (lm - pi[:2]) - rel
                j = np.zeros((2, 3))
                j[:, :2] = -rt
                j[:, 2] = _drot_T(pi[2]) @ (lm - pi[:2])
                cost += w_s * float(r @ r)
                b[3 * i:3 * i + 3] += w_s * (j.T @ r)
                h[3 * i:3 * i + 3, 3 * i:3 * i + 3] += w_s * (j.T @ j)

            for f_idx, (i, m, w_xy, w_th) in enumerate(self.absolute):
                pi = self.poses[i]
                r = np.array([pi[0] - m[0], pi[1] - m[1], float(_wrap(pi[2] - m[2]))])
                scale = float(abs_weights[f_idx])
                w = np.diag([w_xy * scale, w_xy * scale, w_th * scale])
                cost += float(r @ w @ r)
                b[3 * i:3 * i + 3] += w @ r
                h[3 * i:3 * i + 3, 3 * i:3 * i + 3] += w

            h += damping * np.eye(3 * n)
            try:
                delta = np.linalg.solve(h, -b)
            except np.linalg.LinAlgError:
                return it, cost
            self.poses += delta.reshape(n, 3)
            self.poses[:, 2] = _wrap(self.poses[:, 2])
            converged = float(np.abs(delta).max()) < 1e-6
            if kernel == "gnc" and mu > 1.0:
                mu = max(1.0, mu / 1.4)  # anneal toward the true kernel
            elif converged:
                return it + 1, cost
        return iterations, cost


class PoseRefinement(ProcessingPlugin):
    _manifest = PluginManifest(
        name="pose-refinement",
        version="2.1.0",
        consumes=["scan_2d", "registration"],
        produces=["pose_corrections"],
        mode="offline",
        description="Pose-graph optimization: odometry + fiducial landmark + "
                    "GNC-robust scan-match factors (Gauss-Newton)",
        validation="benchmark-validated",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(
        self,
        ctx: ProcessingContext,
        max_pair_dist: float = 0.3,
        max_correction_m: float = 0.5,
        sigma_odo_xy: float = 0.03,
        sigma_odo_th: float = 0.01,
        sigma_fid: float = 0.02,
        sigma_icp_xy: float = 0.04,
        sigma_icp_th: float = 0.02,
        robust_kernel: str = "gnc",
        **params,
    ) -> None:
        regs = ctx.claims("registration")
        if not regs:
            ctx.note("no accepted registration claim; refusing to refine in an unaligned frame")
            return
        reg = regs[0]
        r = np.array(reg["payload"]["rotation"])
        t = np.array(reg["payload"]["translation"])
        control_points: dict = ctx.project.get("control_points", {})

        # scan poses (variables), coarse-initialised in the project frame
        scans: list[tuple[str, float, dict, np.ndarray, np.ndarray]] = []
        for obs in ctx.observations("scan_2d"):
            for ev in ctx.evidence_for(obs["id"]):
                payload = ctx.payload(ev["id"])
                raw = np.asarray(payload["pose_est"], dtype=float)
                coarse = transform_pose(r, t, raw)
                scans.append((obs["id"], obs["t"], payload, raw, coarse))
        if len(scans) < 3:
            ctx.note("too few scans to build a pose graph")
            return
        scans.sort(key=lambda s: s[1])
        times = np.array([s[1] for s in scans])
        coarse_poses = np.array([s[4] for s in scans])

        graph = PoseGraph(coarse_poses)
        w_odo_xy, w_odo_th = 1 / sigma_odo_xy**2, 1 / sigma_odo_th**2
        w_fid = 1 / sigma_fid**2
        w_icp_xy, w_icp_th = 1 / sigma_icp_xy**2, 1 / sigma_icp_th**2

        # odometry factors from the raw carrier estimate (relative motion is
        # frame-invariant under the rigid registration)
        for i in range(len(scans) - 1):
            pi_raw, pj_raw = scans[i][3], scans[i + 1][3]
            rel = _rot(pi_raw[2]).T @ (pj_raw[:2] - pi_raw[:2])
            z = np.array([rel[0], rel[1], float(_wrap(pj_raw[2] - pi_raw[2]))])
            graph.add_odometry(i, i + 1, z, w_odo_xy, w_odo_th)

        # landmark factors: fiducial detections tied to the nearest scan pose
        n_landmark = 0
        for obs in ctx.observations("fiducial_detection"):
            idx = int(np.argmin(np.abs(times - obs["t"])))
            if abs(times[idx] - obs["t"]) > 0.5:
                continue
            for ev in ctx.evidence_for(obs["id"]):
                payload = ctx.payload(ev["id"])
                fid = str(payload["fiducial_id"])
                if fid not in control_points:
                    continue
                graph.add_landmark(
                    idx,
                    np.asarray(control_points[fid], dtype=float),
                    np.asarray(payload["relative"], dtype=float),
                    w_fid,
                )
                n_landmark += 1

        # scan-match factors from ICP against the coarse aggregate map
        ref_points = np.concatenate(
            [_scan_points_world(p, pose) for _, _, p, _, pose in scans], axis=0
        )
        ref_hash = PointHash(ref_points, cell=max_pair_dist)
        icp_rmses: list[float] = []
        n_icp = 0
        for i, (_, _, payload, _, coarse) in enumerate(scans):
            pts = _scan_points_world(payload, coarse)
            if len(pts) < 10:
                icp_rmses.append(float("nan"))
                continue
            ri, ti, rmse, _ = icp_2d(pts, ref_hash, max_pair_dist=max_pair_dist)
            if not math.isfinite(rmse):
                icp_rmses.append(float("nan"))
                continue
            graph.add_absolute(i, transform_pose(ri, ti, coarse), w_icp_xy, w_icp_th)
            icp_rmses.append(rmse)
            n_icp += 1

        iterations, cost = graph.optimize(kernel=robust_kernel)

        # sanity-bound corrections; revert any pose the optimizer flung away
        obs_ids, before, after = [], [], []
        n_refined = 0
        for i, (obs_id, _, _, _, coarse) in enumerate(scans):
            optimized = graph.poses[i]
            shift = float(np.hypot(*(optimized[:2] - coarse[:2])))
            if shift > max_correction_m:
                optimized = coarse
            else:
                n_refined += 1
            obs_ids.append(obs_id)
            before.append(coarse)
            after.append(optimized)

        corrections = np.linalg.norm(
            np.array(after)[:, :2] - np.array(before)[:, :2], axis=1
        )
        corr_ev = ctx.store_derived(
            "pose_corrections",
            {
                "obs_ids": np.array(obs_ids),
                "pose_before": np.array(before),
                "pose_after": np.array(after),
                "icp_rmse": np.array(icp_rmses),
            },
            meta={"frame": "project", "method": "pose-graph"},
        )
        finite = np.array([x for x in icp_rmses if math.isfinite(x)])
        ctx.emit_claim(
            kind="pose_corrections",
            payload={
                "evidence_id": corr_ev,
                "method": "pose-graph",
                "robust_kernel": robust_kernel,
                "n_scans": len(scans),
                "n_refined": n_refined,
                "factors": {
                    "odometry": len(scans) - 1,
                    "landmark": n_landmark,
                    "scan_match": n_icp,
                },
                "gauss_newton_iterations": iterations,
                "final_cost": cost,
                "mean_correction_m": float(corrections.mean()),
                "max_correction_m": float(corrections.max()),
                "mean_icp_rmse_m": float(finite.mean()) if finite.size else None,
            },
            confidence=float(
                np.clip(reg["confidence"] * (n_refined / max(len(scans), 1)), 0.0, 0.99)
            ),
            evidence_ids=[corr_ev],
            subject="pose_corrections:main",
        )
