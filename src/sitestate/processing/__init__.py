from .registration import ControlPointRegistration
from .refine import PoseRefinement
from .mapping import OccupancyMapping
from .coverage import CoverageAnalysis
from .change import ChangeDetection
from .plan_compare import PlanComparison
from .semantics import SemanticLabeling
from .traversability import TraversabilityAnalysis
from .qa import CalibrationCheck, RegistrationVerification
from .progress import ProgressTracking

# registration order doubles as the preferred execution order for
# process_all: earlier plug-ins' products unlock later plug-ins' inputs
ALL_PLUGINS = (
    ControlPointRegistration,
    PoseRefinement,
    OccupancyMapping,
    CoverageAnalysis,
    CalibrationCheck,
    ChangeDetection,
    RegistrationVerification,
    PlanComparison,
    ProgressTracking,
    SemanticLabeling,
    TraversabilityAnalysis,
)

__all__ = [
    "ControlPointRegistration",
    "PoseRefinement",
    "OccupancyMapping",
    "CoverageAnalysis",
    "CalibrationCheck",
    "ChangeDetection",
    "RegistrationVerification",
    "PlanComparison",
    "ProgressTracking",
    "SemanticLabeling",
    "TraversabilityAnalysis",
    "ALL_PLUGINS",
]
