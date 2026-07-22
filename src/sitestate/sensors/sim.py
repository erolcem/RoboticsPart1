"""Simulated sensor adapters.

Each one fulfils the same SensorAdapter contract a real driver would
(e.g. a ROS 2 topic adapter or vendor SDK adapter): declare a manifest,
answer health checks, and emit typed samples. Swapping these for real
sensors requires no change anywhere downstream.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.entities import SensorManifest
from ..plugins.base import Sample, SensorAdapter
from .world import SimCarrier, SimWorld


class SimLidar2D(SensorAdapter):
    def __init__(
        self,
        carrier: SimCarrier,
        world: SimWorld,
        n_rays: int = 180,
        max_range: float = 12.0,
        range_noise: float = 0.01,
        rate_hz: float = 1.0,
        seed: int = 1,
        name: str = "sim-lidar-2d",
        fail_after: float | None = None,
    ):
        self.carrier = carrier
        self.world = world
        self.n_rays = n_rays
        self.max_range = max_range
        self.range_noise = range_noise
        # simulated hardware failure at mission time `fail_after` seconds;
        # exercises the degraded-operation requirement
        self.fail_after = fail_after
        self.period = 1.0 / rate_hz
        self._last_t = -1e9
        self._rng = np.random.default_rng(seed)
        self._manifest = SensorManifest(
            name=name,
            sensor_type="lidar_2d",
            data_types=["scan_2d"],
            units={"range": "m", "angle": "rad"},
            timestamp_source="carrier_clock",
            expected_accuracy={"range_sigma_m": range_noise},
            mounting={"x": 0.0, "y": 0.0, "theta": 0.0},
            calibration_version="sim-cal-1",
            limitations=["2D horizontal plane only", f"max range {max_range} m"],
        )

    @property
    def manifest(self) -> SensorManifest:
        return self._manifest

    def health_check(self) -> dict:
        if self.fail_after is not None and self._last_t >= self.fail_after:
            return {"ok": False, "notes": [f"no data since t={self.fail_after}s (simulated fault)"]}
        return {"ok": self.fail_after != 0.0, "notes": ["simulated device"]}

    def sample(self, t: float) -> list[Sample]:
        if self.fail_after is not None and t >= self.fail_after:
            return []
        if t - self._last_t < self.period - 1e-9:
            return []
        self._last_t = t
        tx, ty, tth = self.carrier.true_pose(t)
        angles = np.linspace(0, 2 * math.pi, self.n_rays, endpoint=False)
        ranges = self.world.raycast((tx, ty), angles + tth, self.max_range)
        hit = ranges < self.max_range - 1e-6
        ranges = ranges + self._rng.normal(0, self.range_noise, size=ranges.shape)
        ex, ey, eth, sigma = self.carrier.estimated_pose(t)
        return [
            Sample(
                data_type="scan_2d",
                payload={
                    "angles": angles,  # sensor frame
                    "ranges": ranges,
                    "hit": hit.astype(np.int8),
                    "pose_est": np.array([ex, ey, eth]),
                    "max_range": self.max_range,
                },
                quality={"pose_sigma_xy": sigma, "range_sigma_m": self.range_noise},
                frame="mission-estimated",
            )
        ]


class SimOdometry(SensorAdapter):
    def __init__(self, carrier: SimCarrier, rate_hz: float = 2.0, name: str = "sim-odometry"):
        self.carrier = carrier
        self.period = 1.0 / rate_hz
        self._last_t = -1e9
        self._manifest = SensorManifest(
            name=name,
            sensor_type="odometry",
            data_types=["pose_estimate"],
            units={"x": "m", "y": "m", "theta": "rad"},
            expected_accuracy={"drift_m_per_sqrt_s": carrier.drift_rate},
            calibration_version="sim-cal-1",
            limitations=["unbounded drift without external registration"],
        )

    @property
    def manifest(self) -> SensorManifest:
        return self._manifest

    def health_check(self) -> dict:
        return {"ok": True, "notes": []}

    def sample(self, t: float) -> list[Sample]:
        if t - self._last_t < self.period - 1e-9:
            return []
        self._last_t = t
        ex, ey, eth, sigma = self.carrier.estimated_pose(t)
        return [
            Sample(
                data_type="pose_estimate",
                payload={"pose_est": np.array([ex, ey, eth]), "t": t},
                quality={"pose_sigma_xy": sigma},
                frame="mission-estimated",
            )
        ]


class SimFiducialCamera(SensorAdapter):
    """Detects surveyed fiducial targets (control points) within range.

    Reports where each detected fiducial appears to be in the carrier's
    estimated map frame; the registration plug-in compares these with the
    surveyed project-frame coordinates to align the mission.
    """

    def __init__(
        self,
        carrier: SimCarrier,
        world: SimWorld,
        detection_range: float = 6.0,
        noise: float = 0.015,
        rate_hz: float = 1.0,
        seed: int = 2,
        name: str = "sim-fiducial-camera",
        calibration_bias: tuple[float, float] = (0.0, 0.0),
    ):
        self.carrier = carrier
        self.world = world
        self.detection_range = detection_range
        self.noise = noise
        # a systematic sensor-frame offset simulating stale/knocked
        # calibration; the calibration-check plug-in must detect it
        self.calibration_bias = np.asarray(calibration_bias, dtype=float)
        self.period = 1.0 / rate_hz
        self._last_t = -1e9
        self._rng = np.random.default_rng(seed)
        self._manifest = SensorManifest(
            name=name,
            sensor_type="fiducial_camera",
            data_types=["fiducial_detection"],
            units={"position": "m"},
            expected_accuracy={"detection_sigma_m": noise},
            calibration_version="sim-cal-1",
            limitations=[f"detection range {detection_range} m", "requires line of sight"],
        )

    @property
    def manifest(self) -> SensorManifest:
        return self._manifest

    def health_check(self) -> dict:
        return {"ok": True, "notes": []}

    def sample(self, t: float) -> list[Sample]:
        if t - self._last_t < self.period - 1e-9:
            return []
        self._last_t = t
        tx, ty, tth = self.carrier.true_pose(t)
        ex, ey, eth, sigma = self.carrier.estimated_pose(t)
        samples: list[Sample] = []
        for fid, (fx, fy) in self.world.fiducials.items():
            dx, dy = fx - tx, fy - ty
            if math.hypot(dx, dy) > self.detection_range:
                continue
            if not self.world.visible((tx, ty), (fx, fy)):
                continue  # a wall blocks the target - no detection
            # true relative measurement (sensor frame), with noise and any
            # systematic calibration bias
            c, s = math.cos(-tth), math.sin(-tth)
            rel = np.array([c * dx - s * dy, s * dx + c * dy])
            rel += self._rng.normal(0, self.noise, size=2) + self.calibration_bias
            # where the carrier believes the fiducial is, in its estimated frame
            ce, se = math.cos(eth), math.sin(eth)
            est = np.array([ex + ce * rel[0] - se * rel[1], ey + se * rel[0] + ce * rel[1]])
            samples.append(
                Sample(
                    data_type="fiducial_detection",
                    payload={
                        "fiducial_id": fid,
                        "position_est": est,
                        "relative": rel,  # sensor-frame measurement
                        "pose_est": np.array([ex, ey, eth]),
                    },
                    quality={"detection_sigma_m": self.noise, "pose_sigma_xy": sigma},
                    frame="mission-estimated",
                )
            )
        return samples


class SimDepthCamera(SensorAdapter):
    """A minimal forward-facing depth camera producing spatially linked
    imagery: each frame is stored with its acquisition pose so evidence can
    be opened from its position in the model."""

    def __init__(
        self,
        carrier: SimCarrier,
        world: SimWorld,
        fov_deg: float = 90.0,
        width: int = 48,
        max_range: float = 8.0,
        rate_hz: float = 0.5,
        seed: int = 3,
        name: str = "sim-depth-camera",
    ):
        self.carrier = carrier
        self.world = world
        self.fov = math.radians(fov_deg)
        self.width = width
        self.max_range = max_range
        self.period = 1.0 / rate_hz
        self._last_t = -1e9
        self._rng = np.random.default_rng(seed)
        self._manifest = SensorManifest(
            name=name,
            sensor_type="depth_camera",
            data_types=["depth_image"],
            units={"depth": "m"},
            expected_accuracy={"depth_sigma_m": 0.02},
            mounting={"x": 0.1, "y": 0.0, "theta": 0.0},
            calibration_version="sim-cal-1",
            limitations=[f"horizontal FOV {fov_deg} deg", "1D depth row in simulation"],
        )

    @property
    def manifest(self) -> SensorManifest:
        return self._manifest

    def health_check(self) -> dict:
        return {"ok": True, "notes": []}

    def sample(self, t: float) -> list[Sample]:
        if t - self._last_t < self.period - 1e-9:
            return []
        self._last_t = t
        tx, ty, tth = self.carrier.true_pose(t)
        angles = np.linspace(-self.fov / 2, self.fov / 2, self.width)
        depths = self.world.raycast((tx, ty), angles + tth, self.max_range)
        depths = depths + self._rng.normal(0, 0.02, size=depths.shape)
        ex, ey, eth, sigma = self.carrier.estimated_pose(t)
        return [
            Sample(
                data_type="depth_image",
                payload={
                    "angles": angles,
                    "depths": depths,
                    "pose_est": np.array([ex, ey, eth]),
                },
                quality={"pose_sigma_xy": sigma, "depth_sigma_m": 0.02},
                frame="mission-estimated",
            )
        ]
