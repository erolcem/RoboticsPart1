# Contributing / working on this codebase

## Setup

```bash
pip install -e .[dev]     # numpy + pytest; nothing else
pytest tests/ -q          # 37 tests, ~50 s
python3 -m pyflakes src/ tests/ examples/   # keep it clean
```

## The three rules

**1. The benchmark is the referee.** Any change to a processing plug-in
must be justified against `sitestate benchmark` (and the thresholds
asserted in tests). If trajectory RMSE, map/change precision-recall,
coverage honesty or traceability regress, the change is wrong no matter
how much better it looks in a demo. New algorithms should add measures
when they claim new capabilities.

**2. Claims are contracts.** Payload keys documented in
docs/data-reference.md are load-bearing: add keys freely, never rename or
remove without bumping the export schema version and updating the doc.
Plug-in manifests use semver; re-running the *same* plug-in name
supersedes old claims, a *different* name competes — pick deliberately.
Confidence must mean something: it is the model's honest self-assessment,
and the calibration section of the benchmark will expose wishful numbers.

**3. Honesty invariants are non-negotiable.** Unobserved is never
unchanged; unseen space is never traversable; a coverage gap is never
progress or demolition; missing preconditions are flagged (`skipped`/
`rejected`), never guessed around. Tests encode these; keep them passing.

## Layout conventions

- Core stays numpy-only. Heavy or learned models (torch, SAM, 3DGS, ROS)
  live in *plug-ins/adapters*, ideally in a separate package that depends
  on `sitestate`, never the other way around.
- New sensor → `SensorAdapter` (see `examples/custom_asset_tracking.py`).
  New model → `ProcessingPlugin`. New export → `OutputAdapter`. If a
  change touches core *and* a plug-in, the contract is probably in the
  wrong place.
- The simulator (`sensors/world.py`, `sensors/sim.py`) is the permanent
  regression harness: when a real-world failure mode bites, encode it in
  the sim (like LOS occlusion, calibration bias and sensor dropout) so it
  can never bite silently again.

## Checklist for a change

1. `pytest tests/ -q` green; add tests for new behaviour.
2. `sitestate benchmark --seeds 3` — measures moved the right way (or
   unchanged).
3. pyflakes clean.
4. Docs updated: data-reference for schema changes, api-reference for
   public-surface changes, sota-review if the change tracks a paper,
   CHANGELOG entry.
5. If it affects deployment: `docker build -t sitestate . && docker
   compose up` still serves `/healthz`.
