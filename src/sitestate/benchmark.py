"""Benchmark harness: the proposal's §16 measures table, executable.

Runs the full two-mission scenario end-to-end over several random seeds
(different frame offsets, drift and noise realizations each time) and
scores the pipeline against simulation ground truth:

* trajectory_rmse_m      - corrected trajectory vs true carrier path
                           (repeat-registration accuracy)
* map_precision/recall   - occupied cells vs the true wall geometry
* change_precision/recall- detected change regions vs the seeded changes
* coverage_honesty       - fraction of physically unseen cells (interiors
                           of closed boxes) NOT reported as observed
* traceability           - fraction of claims whose provenance reaches
                           both evidence and a sensor manifest
* processing_latency_s   - mission end to processed claims

This is how "perfecting the models" stays honest: every algorithm change
must move these numbers, not just look better in a demo.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import demo
from .design.floorplan import FloorPlan
from .processing.grid import OCCUPIED, dilate

# ground-truth change regions seeded by demo.build_world (bboxes, project frame)
TRUE_CHANGES = [
    {"type": "appeared", "bbox": [12.6, 5.3, 13.8, 6.5]},   # pallet
    {"type": "appeared", "bbox": [10.1, 1.9, 10.4, 3.7]},   # partition (new pos)
    {"type": "disappeared", "bbox": [8.9, 1.9, 9.2, 3.7]},  # partition (old pos)
    {"type": "disappeared", "bbox": [2.4, 6.3, 3.8, 7.4]},  # material stack
]


def _in_bbox(p: list[float], b: list[float], slack: float = 0.15) -> bool:
    return (b[0] - slack <= p[0] <= b[2] + slack) and (b[1] - slack <= p[1] <= b[3] + slack)


def _true_occupancy_mask(world, grid: dict) -> np.ndarray:
    plan = FloorPlan(walls=world.walls)
    ny, nx = grid["occ"].shape
    return plan.rasterize(float(grid["x0"]), float(grid["y0"]), nx, ny, float(grid["res"]))


def run_scenario(seed: int, root: Path) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    off1 = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.04, 0.04))
    off2 = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.04, 0.04))

    platform = demo.make_platform(root)
    world1, world2 = demo.build_world(1), demo.build_world(2)

    m1 = demo.run_capture(platform, world1, "bench day1", frame_offset=off1, seed=seed * 100 + 1)
    m2 = demo.run_capture(platform, world2, "bench day2", frame_offset=off2, seed=seed * 100 + 2)

    t0 = time.perf_counter()
    platform.process_all(m1.id)
    platform.process_all(m2.id, baseline_mission_id=m1.id)
    latency = time.perf_counter() - t0

    ledger = platform.ledger
    metrics: dict[str, Any] = {"seed": seed, "processing_latency_s": latency}

    # --- trajectory accuracy: corrected trajectory vs true carrier path ---
    # rebuild the same carriers to query ground truth (same waypoints/speed)
    from .sensors import SimCarrier

    errs_coarse, errs_refined = [], []
    for mid, off, cseed in ((m1.id, off1, seed * 100 + 1), (m2.id, off2, seed * 100 + 2)):
        carrier = SimCarrier(demo.WAYPOINTS, speed=0.7, frame_offset=off,
                             drift_rate=0.004, seed=cseed)
        trajs = ledger.claims(mission_id=mid, kind="trajectory", status="accepted")
        if trajs:
            tp = ledger.evidence_payload(trajs[0]["payload"]["evidence_id"])
            for pose, t in zip(tp["poses"], tp["t"]):
                tx, ty, _ = carrier.true_pose(float(t))
                errs_coarse.append(math.hypot(pose[0] - tx, pose[1] - ty))
        # the poses mapping actually consumes: pose-graph refined
        pcs = ledger.claims(mission_id=mid, kind="pose_corrections", status="accepted")
        if pcs:
            ev = ledger.evidence_payload(pcs[0]["payload"]["evidence_id"])
            obs_t = {o["id"]: o["t"] for o in ledger.observations(mid, "scan_2d")}
            for oid, pose in zip(ev["obs_ids"], ev["pose_after"]):
                t = obs_t.get(str(oid))
                if t is None:
                    continue
                tx, ty, _ = carrier.true_pose(float(t))
                errs_refined.append(math.hypot(pose[0] - tx, pose[1] - ty))
    metrics["coarse_trajectory_rmse_m"] = (
        float(np.sqrt(np.mean(np.square(errs_coarse)))) if errs_coarse else None
    )
    metrics["trajectory_rmse_m"] = (
        float(np.sqrt(np.mean(np.square(errs_refined))))
        if errs_refined
        else metrics["coarse_trajectory_rmse_m"]
    )

    # --- map accuracy vs true geometry (day 2) ------------------------------
    geoms = ledger.claims(mission_id=m2.id, kind="occupancy_geometry", status="accepted")
    grid = ledger.evidence_payload(geoms[0]["payload"]["evidence_id"])
    true_mask = _true_occupancy_mask(world2, grid)
    occ_mask = grid["occ"] == OCCUPIED
    near_true = dilate(true_mask, 1)
    near_occ = dilate(occ_mask, 1)
    metrics["map_precision"] = float((occ_mask & near_true).sum() / max(occ_mask.sum(), 1))
    seen = grid["hits"] + grid["passes"] > 0
    observable_true = true_mask & seen
    metrics["map_recall"] = float(
        (observable_true & near_occ).sum() / max(observable_true.sum(), 1)
    )

    # --- change detection precision / recall --------------------------------
    # precision is scored over claims NOT flagged as registration artifacts:
    # a flagged claim is explicitly presented as suspect, so it is a
    # screened-out candidate, not a false report
    changes = ledger.claims(mission_id=m2.id, kind="change", status="accepted")
    unflagged = [c for c in changes
                 if not c["payload"].get("likely_registration_artifact")]
    matched_truth = set()
    tp_claims = 0
    for c in unflagged:
        hit = False
        for k, truth in enumerate(TRUE_CHANGES):
            if truth["type"] == c["payload"]["change_type"] and _in_bbox(
                c["payload"]["centroid"], truth["bbox"]
            ):
                matched_truth.add(k)
                hit = True
        tp_claims += int(hit)
    metrics["change_precision"] = tp_claims / len(unflagged) if unflagged else None
    metrics["change_recall"] = len(matched_truth) / len(TRUE_CHANGES)
    metrics["n_change_claims"] = len(changes)
    metrics["n_change_flagged"] = len(changes) - len(unflagged)
    # (confidence, was-it-real) pairs for calibration measurement across seeds
    metrics["change_confidence_pairs"] = [
        [
            float(c["confidence"]),
            any(
                truth["type"] == c["payload"]["change_type"]
                and _in_bbox(c["payload"]["centroid"], truth["bbox"])
                for truth in TRUE_CHANGES
            ),
        ]
        for c in changes
    ]

    # --- plan completion (progress tracking) --------------------------------
    progress = ledger.claims(mission_id=m2.id, kind="progress", status="accepted")
    metrics["plan_completion"] = (
        float(progress[0]["payload"]["overall_completion"]) if progress else None
    )

    # --- coverage honesty: closed-box interiors must not read "observed" ----
    covs = ledger.claims(mission_id=m2.id, kind="coverage", status="accepted")
    cov = ledger.evidence_payload(covs[0]["payload"]["evidence_id"])["coverage"]
    x0, y0, res = float(grid["x0"]), float(grid["y0"]), float(grid["res"])
    interiors = []
    for bx, by, bw, bh in [(12.7, 5.4, 1.0, 1.0), (10.0, 4.2, 0.5, 0.5)]:
        i0, j0 = int((bx + 0.15 - x0) / res), int((by + 0.15 - y0) / res)
        i1, j1 = int((bx + bw - 0.15 - x0) / res), int((by + bh - 0.15 - y0) / res)
        if i1 > i0 and j1 > j0:
            interiors.append(cov[j0:j1, i0:i1])
    hidden = np.concatenate([a.ravel() for a in interiors])
    metrics["coverage_honesty"] = float((hidden != 2).sum() / max(hidden.size, 1))

    # --- provenance traceability --------------------------------------------
    total, traced = 0, 0
    for c in ledger.claims(status="accepted"):
        if c["kind"].endswith("_summary"):
            continue
        total += 1
        tr = ledger.trace(c["id"])
        if tr["sensors"] and (tr["evidence"] or tr["activity"].get("input_evidence_ids")):
            traced += 1
    metrics["traceability"] = traced / max(total, 1)

    # --- QA claims ----------------------------------------------------------
    aligns = ledger.claims(mission_id=m2.id, kind="alignment_check", status="accepted")
    metrics["alignment_within_tolerance"] = (
        bool(aligns[0]["payload"].get("within_tolerance")) if aligns else None
    )
    metrics["registration_rmse_m"] = float(
        np.mean([r["payload"]["rmse_m"]
                 for r in ledger.claims(kind="registration", status="accepted")])
    )
    return metrics


def run_benchmark(
    seeds: int = 3, out_dir: str | Path | None = None, verbose: bool = True
) -> dict[str, Any]:
    import tempfile

    runs = []
    for seed in range(1, seeds + 1):
        with tempfile.TemporaryDirectory() as tmp:
            runs.append(run_scenario(seed, Path(tmp)))
            if verbose:
                r = runs[-1]
                cp = r["change_precision"]
                print(f"  seed {seed}: traj RMSE {r['trajectory_rmse_m'] * 100:.1f} cm | "
                      f"map P/R {r['map_precision']:.2f}/{r['map_recall']:.2f} | "
                      f"change P/R {cp if cp is None else format(cp, '.2f')}"
                      f"/{r['change_recall']:.2f} | "
                      f"honesty {r['coverage_honesty']:.2f} | "
                      f"trace {r['traceability']:.2f} | {r['processing_latency_s']:.1f}s")

    def agg(key: str):
        vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    # confidence calibration: does a confidence of 0.8 mean "right 80% of
    # the time"? Reliability bins + expected calibration error across seeds.
    pairs = [p for r in runs for p in r.get("change_confidence_pairs", [])]
    bins = []
    ece, n_total = 0.0, max(len(pairs), 1)
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        in_bin = [p for p in pairs if lo <= p[0] < hi or (hi == 1.0 and p[0] == 1.0)]
        if not in_bin:
            continue
        mean_conf = float(np.mean([p[0] for p in in_bin]))
        accuracy = float(np.mean([1.0 if p[1] else 0.0 for p in in_bin]))
        bins.append({"range": [lo, hi], "n": len(in_bin),
                     "mean_confidence": mean_conf, "empirical_accuracy": accuracy})
        ece += len(in_bin) / n_total * abs(accuracy - mean_conf)

    summary = {
        "schema": "sitestate/benchmark@2.0",
        "seeds": seeds,
        "measures": {
            k: agg(k)
            for k in (
                "trajectory_rmse_m", "coarse_trajectory_rmse_m",
                "registration_rmse_m", "map_precision", "map_recall",
                "change_precision", "change_recall", "coverage_honesty",
                "traceability", "plan_completion", "processing_latency_s",
            )
        },
        "calibration": {
            "n_claims": len(pairs),
            "expected_calibration_error": ece,
            "reliability_bins": bins,
        },
        "runs": runs,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_report.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print("\n  measure                mean ± std")
        for k, v in summary["measures"].items():
            if v:
                print(f"  {k:<22s} {v['mean']:.3f} ± {v['std']:.3f}")
    return summary
