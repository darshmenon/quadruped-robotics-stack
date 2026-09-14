"""
Gait Scheduler — switches between locomotion gaits based on speed or command.

Gaits supported:
    - stand:    all feet on ground, body held up
    - crouch:   stand with lowered body height
    - sit:      deepest stance hold
    - trot:     diagonal pairs move together (FL+RR, FR+RL)
    - walk:     one foot at a time, stable at low speed
    - pace:     lateral pairs (same-side legs) move together
    - canter:   three-beat gait, medium-high speed
    - bound:    front pair then rear pair, high speed
    - pronk:    all four feet leave ground simultaneously

Usage:
    from intelligence.gait.gait_scheduler import GaitScheduler
    scheduler = GaitScheduler()
    cmd = scheduler.get_gait_command(speed=0.8)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Gait(Enum):
    STAND  = "stand"
    CROUCH = "crouch"
    SIT    = "sit"
    WALK   = "walk"
    TROT   = "trot"
    PACE   = "pace"
    CANTER = "canter"
    BOUND  = "bound"
    PRONK  = "pronk"


@dataclass
class GaitParams:
    name: str
    frequency: float       # step frequency (Hz)
    duty_factor: float     # fraction of cycle each foot is on ground
    phase_offsets: List[float]  # [FL, FR, RL, RR] phase offsets (0-1)
    speed_range: tuple     # (min, max) m/s
    height_offset: float = 0.0  # relative to nominal stand height (m)


GAITS = {
    Gait.STAND:  GaitParams("stand",  0.0, 1.00, [0.0, 0.0, 0.0, 0.0],  (0.0, 0.05), 0.0),
    Gait.CROUCH: GaitParams("crouch", 0.0, 1.00, [0.0, 0.0, 0.0, 0.0],  (-1.0, -0.5), -0.10),
    Gait.SIT:    GaitParams("sit",    0.0, 1.00, [0.0, 0.0, 0.0, 0.0],  (-1.0, -0.5), -0.16),
    # Crawl walk: FL → RR → FR → RL. Old [0, 0.5, 0.25, 0.75] swung both
    # left legs then both right legs, which yawed/drifted every cycle.
    Gait.WALK:   GaitParams("walk",   1.4, 0.75, [0.0, 0.5, 0.75, 0.25],(0.05, 0.35)),
    # Prefer trot once the robot is actually moving — more stable than crawl
    # at the speeds Gazebo teleop can command after clipping.
    Gait.TROT:   GaitParams("trot",   2.0, 0.60, [0.0, 0.5, 0.5, 0.0],  (0.35, 1.5)),
    Gait.PACE:   GaitParams("pace",   2.0, 0.55, [0.0, 0.0, 0.5, 0.5],  (0.4, 1.2)),
    Gait.CANTER: GaitParams("canter", 2.8, 0.55, [0.0, 0.33, 0.66, 0.2],(1.5, 2.5)),
    Gait.BOUND:  GaitParams("bound",  3.5, 0.40, [0.0, 0.0, 0.5, 0.5],  (2.5, 4.0)),
    Gait.PRONK:  GaitParams("pronk",  3.0, 0.30, [0.0, 0.0, 0.0, 0.0],  (4.0, 6.0)),
}


# Speed-based auto-selection ignores stance poses (crouch/sit). select_gait
# returns the first range match, and PACE's speed_range (0.4, 1.2) sits
# entirely inside TROT's (0.35, 1.5) -- checking PACE first lets it claim
# that overlap, leaving TROT the (0.35, 0.4) and (1.2, 1.5) slivers either
# side; checking TROT first (the old order) made PACE unreachable for any
# speed at all.
_LOCOMOTION_GAITS = (Gait.STAND, Gait.WALK, Gait.PACE, Gait.TROT,
                     Gait.CANTER, Gait.BOUND, Gait.PRONK)


class GaitScheduler:
    def __init__(self):
        self.current_gait = Gait.STAND

    def select_gait(self, speed: float) -> Gait:
        if speed < 0:
            return Gait.STAND
        for gait in _LOCOMOTION_GAITS:
            params = GAITS[gait]
            if params.speed_range[0] <= speed < params.speed_range[1]:
                return gait
        return Gait.PRONK

    def select_named(self, name: str) -> Gait:
        key = name.strip().lower()
        for gait in Gait:
            if gait.value == key:
                return gait
        raise ValueError(f"Unknown gait '{name}'. Choose from {[g.value for g in Gait]}")

    def get_gait_params(self, speed: float = 0.0, name: Optional[str] = None) -> GaitParams:
        gait = self.select_named(name) if name else self.select_gait(speed)
        if gait != self.current_gait:
            print(f"Gait switch: {self.current_gait.value} -> {gait.value} at {speed:.2f} m/s")
            self.current_gait = gait
        return GAITS[gait]

    def get_phase(self, leg_index: int, t: float, speed: float,
                  name: Optional[str] = None) -> float:
        """Returns current swing/stance phase (0=stance, 1=swing) for a leg."""
        params = self.get_gait_params(speed=speed, name=name)
        if params.frequency == 0:
            return 0.0
        cycle_pos = (t * params.frequency + params.phase_offsets[leg_index]) % 1.0
        return 1.0 if cycle_pos > params.duty_factor else 0.0
