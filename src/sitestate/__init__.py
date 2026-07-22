"""Site State Platform.

A plug-and-play sensing platform that converts evidence from sensor
carriers into a versioned, uncertainty-aware model of a changing site.

Layers (proposal §9; see README for the full map):
  sensors/     - sensor & carrier adapters (SensorAdapter + SensorManifest)
  ingest/      - portable dataset recorder/replay (the hardware bridge)
  ledger/      - observation ledger: evidence + metadata, append-only
  processing/  - replaceable plug-ins: registration, pose-graph SLAM,
                 evidential occupancy, coverage, change, plan/progress,
                 semantics, traversability, QA self-checks
  design/      - designed-state floor plan + named zones
  statemodel/  - versioned Site State Model built from evidence-linked claims
  outputs/     - output adapters (HTML report, JSON package, robot costmap)
  review.py    - human review queue (accept/reject/request re-capture)
  query.py     - belief-at-a-point API with freshness decay
  planning.py  - coverage-gap + recapture-request capture planner
  benchmark.py - ground-truth scoring incl. confidence calibration
  serve.py     - read-only HTTP API + interactive viewer
  platform.py  - orchestrator + process_all pipeline runner

Contracts and payload schemas: docs/data-reference.md.
Extension tutorial in executable form: examples/custom_asset_tracking.py.
"""

from .platform import SiteStatePlatform, load_platform

__all__ = ["SiteStatePlatform", "load_platform"]
__version__ = "2.0.1"
