# Changelog

## 2.0.0 — July 2026

Grounded in a fresh state-of-the-art survey (docs/sota-review.md, with
sources): the v2 upgrades implement the field's consensus architectures
where feasible in-core, and confirm the plug-in seams where the frontier
is moving fastest (SAM-class segmentation, 3DGS reconstruction, conformal
calibration).

### Algorithms
- **Pose-graph trajectory optimization** (pose-refinement v2.0.0): every
  scan pose becomes a variable in one Gauss-Newton problem with odometry
  factors, fiducial **landmark factors** and **Huber-robust scan-match
  (ICP) factors** — the standard modern SLAM back-end (LIO-SAM lineage)
  in 2D. Measured: trajectory RMSE 2.4 cm → **1.1 cm** vs registration
  alone (benchmark now reports both `trajectory_rmse_m` and
  `coarse_trajectory_rmse_m`).
- **Evidential occupancy layer** (occupancy-mapping v2.0.0): per-cell
  Dempster-Shafer masses with explicit **ignorance** (epistemic: never
  observed enough) separated from **conflict** (evidence disagrees:
  dynamics/misregistration). Stored in the grid evidence; summarized on
  the mapping claim; change claims report their region's evidential
  conflict.
- **Progress tracking** (new plug-in): per-zone and overall completion of
  the designed plan, measured only over observed plan elements (a
  coverage gap can never masquerade as demolition). Rendered as progress
  bars in the report; `plan_completion` is a benchmark measure.

### Measurement
- **Confidence calibration**: the benchmark aggregates
  (confidence, was-it-real) pairs for change claims across seeds into
  reliability bins and an **expected calibration error**. Current finding:
  the detector is *underconfident* (reported 0.76–0.83 where empirical
  accuracy is 1.00 in simulation) — documented rather than tuned away, to
  avoid overfitting confidence to the simulator.

### Deployment (the "deployable v2")
- **Dockerfile** (non-root, /data volume, HEALTHCHECK), **docker-compose**
  (demo + server), **Makefile**, **GitHub Actions CI** (pytest on 3.10 &
  3.12, benchmark smoke, Docker build + container smoke test). Container
  build and serve verified locally end-to-end.
- Server: `/healthz` and `/version` probes; `--host` binding option with
  an explicit no-auth warning beyond localhost. docs/deployment.md.

### Fixes
- Benchmark trajectory measure now scores the poses mapping actually
  consumes (refined), not just the coarse trajectory claim.
- Costmap export breaks `observed_at` ties by mission recency.
- Freshness policy covers `progress` claims.

## 1.0.0 — July 2026

### Algorithms
- **Log-odds occupancy mapping** with an explicit inverse sensor model,
  per-cell occupancy probability, and decision entropy as a map-quality
  scalar folded into map confidence. Ray integration fully vectorised.
- **Scan-to-map ICP pose refinement** (`pose-refinement`): removes the
  residual odometry random walk after control-point registration; mapping
  consumes the refined poses automatically when present.
- **Registration verification** (`registration-verification`): cross-mission
  ICP consistency check; residual misalignment beyond tolerance is flagged
  for review instead of contaminating change detection.
- **Calibration check** (`calibration-check`): per-sensor systematic bias of
  fiducial detections measured in the *sensor frame* (rotation-invariant),
  against surveyed control points.
- **Change detection v1.0**: registration-artifact screening (regions
  hugging structure in the other mission are flagged with reduced
  confidence), spatially linked depth imagery per change region, zone tags.
- **Simulator realism**: line-of-sight fiducial occlusion, injectable
  calibration bias, mid-mission sensor failure; raycasting vectorised.

### Platform
- `process_all`: dependency-ordered pipeline runner (iterates to fixpoint
  as produced claims unlock further plug-ins; cross-mission plug-ins get
  the baseline automatically; skipped plug-ins leave explanatory records).
- `project.json` persistence + `sitestate init`; `load_platform()` factory.
- **Named zones**: per-zone coverage fractions; change/deviation/entity
  claims carry the zone containing them.
- **Capture planning** (`sitestate plan-capture`, `/api/plan`): coverage
  gaps (frontier-filtered to reachable space) plus human recapture requests
  become an ordered waypoint tour for the next mission.
- **Benchmark harness** (`sitestate benchmark`): the proposal's §16
  measures scored against simulation ground truth over N seeds —
  trajectory RMSE, map precision/recall, change precision/recall, coverage
  honesty, provenance traceability, latency.

### Outputs & interfaces
- HTML report: embedded PNG depth-frame evidence per change (pure-stdlib
  PNG encoder), QA sections (alignment + calibration verdicts), per-zone
  coverage tables, artifact flags, toggleable map layers.
- **Live web viewer** (`/viewer`): click anywhere on the map to query the
  site state at that point (occupancy, coverage, traversability,
  freshness, claims, source ids).
- New CLI commands: `init`, `process`, `plan-capture`, `benchmark`.
- Fixed: ledger SQLite connection usable from server worker threads.

### Measured (3-seed benchmark, simulation ground truth)
trajectory RMSE ≈ 2.4 cm · registration RMSE ≈ 1.6 cm · map P/R ≈ 1.00/1.00
· change P/R = 1.00/1.00 · coverage honesty = 1.00 · traceability = 1.00
· processing ≈ 1.2 s per two-mission scenario.

## 0.2.0 — July 2026
Full-vision build: plan comparison, semantics with class probabilities,
traversability + costmap/planning-scene export, competing claims, review
workflow, query API with freshness, dataset record/replay, CLI, HTTP API.

## 0.1.0 — July 2026
Initial scaffold: entities, observation ledger, plug-in contracts,
registration/mapping/coverage/change, versioned state model, HTML/JSON
exports, simulated sensors, two-mission demo.
