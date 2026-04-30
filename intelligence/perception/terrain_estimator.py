"""
Terrain Estimator — estimates terrain type and slope from foot contact forces
and IMU data, then adapts gait parameters accordingly.

Inputs:
    - IMU: roll, pitch, angular velocity
    - Foot contact forces (one per leg)

Outputs:
    - Terrain class: flat, slope, stairs, rough
    - Estimated slope angle (degrees)
    - Recommended gait adjustment

Usage:
    from intelligence.perception.terrain_estimator import TerrainEstimator
    estimator = TerrainEstimator()
    result = estimator.estimate(imu_roll=0.1, imu_pitch=0.05, contacts=[120, 115, 118, 122])
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
import math


class TerrainType(Enum):
    FLAT   = "flat"
    SLOPE  = "slope"
    STAIRS = "stairs"
    ROUGH  = "rough"


@dataclass
class TerrainEstimate:
    terrain_type: TerrainType
    slope_deg: float
    contact_variance: float
    recommended_speed_limit: float   # m/s
    recommended_foot_clearance: float  # m


class TerrainEstimator:
    def __init__(self):
        self.contact_history: List[List[float]] = []
        self.history_len = 20

    def estimate(
        self,
        imu_roll: float,
        imu_pitch: float,
        contacts: List[float],  # [FL, FR, RL, RR] normal forces (N)
    ) -> TerrainEstimate:

        if not contacts:
            contacts = [0.0, 0.0, 0.0, 0.0]

        slope_deg = math.degrees(math.atan(math.sqrt(imu_roll**2 + imu_pitch**2)))

        self.contact_history.append(contacts)
        if len(self.contact_history) > self.history_len:
            self.contact_history.pop(0)

        mean_force = sum(contacts) / len(contacts)
        variance = sum((f - mean_force) ** 2 for f in contacts) / len(contacts)

        # Classify terrain
        if slope_deg > 15:
            terrain = TerrainType.SLOPE
            speed_limit = 0.8
            clearance = 0.08
        elif variance > 2000:
            terrain = TerrainType.STAIRS
            speed_limit = 0.4
            clearance = 0.12
        elif variance > 800:
            terrain = TerrainType.ROUGH
            speed_limit = 1.0
            clearance = 0.07
        else:
            terrain = TerrainType.FLAT
            speed_limit = 3.0
            clearance = 0.05

        return TerrainEstimate(
            terrain_type=terrain,
            slope_deg=round(slope_deg, 2),
            contact_variance=round(variance, 2),
            recommended_speed_limit=speed_limit,
            recommended_foot_clearance=clearance,
        )
