"""Site State Platform.

A plug-and-play sensing platform that converts evidence from sensor
carriers into a versioned, uncertainty-aware model of a changing site.

Layers (see the v0.1 proposal document):
  sensors/     - sensor & carrier adapters (SensorAdapter + SensorManifest)
  ledger/      - observation ledger: evidence + metadata, append-only
  processing/  - replaceable processing plug-ins (ProcessingPlugin)
  statemodel/  - versioned Site State Model built from evidence-linked claims
  outputs/     - output adapters (reports, machine-readable exports)
  platform.py  - orchestrator tying the layers together
"""

from .platform import SiteStatePlatform

__all__ = ["SiteStatePlatform"]
__version__ = "2.0.0"
