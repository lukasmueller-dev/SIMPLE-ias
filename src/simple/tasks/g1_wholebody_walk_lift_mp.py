# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Walk to the lab table and lift the bottle: the pour scene, the robot spawned WALK behind the pour stance."""

import dataclasses
from typing import Any, Optional

from simple.dr.language import LanguageDRCfg
from simple.dr.types import Box
from simple.tasks.g1_wholebody_pour_mp import ROBOT_JITTER, ROBOT_X, ROBOT_Z, G1WholebodyPourMP
from simple.tasks.registry import TaskRegistry

WALK = 0.35                  # m: sim xmovepick's walk (integrated vx command, median of its demos)
WALK_VX = 0.35               # m/s, as G1InspireWholebodyLocomotionMP
WALK_CMD = 0.44              # WalkSpec's target_distance, open loop: 0.35 landed 6 cm short of pour's grasp stance (pelvis x -0.48)
DRIFT_Y = 0.04               # the walk drifts left ~4-5 cm: spawn this far right, or the right hand pushes the bottle over
LIFT_MIN, HOLD_S = 0.08, 1.0  # success: the bottle this far above its start for this long
INSTRUCTION = "move forward to pick up the bottle"   # sim xmovepick's template


@TaskRegistry.register("g1_wholebody_walk_lift_mp")
class G1WholebodyWalkLiftMP(G1WholebodyPourMP):
    uid: str = "g1_wholebody_walk_lift_mp"
    label: str = "G1 Wholebody Walk Lift MP"
    description: str = "The G1 walks to the table, grasps the bottle and lifts it."

    dr_cfgs = {
        **G1WholebodyPourMP.dr_cfgs,
        "language": LanguageDRCfg(instructions=[INSTRUCTION]),
        "spatial": dataclasses.replace(G1WholebodyPourMP.dr_cfgs["spatial"], robot_region=Box(
            low=[ROBOT_X - WALK - ROBOT_JITTER, -DRIFT_Y - ROBOT_JITTER, ROBOT_Z],
            high=[ROBOT_X - WALK + ROBOT_JITTER, -DRIFT_Y + ROBOT_JITTER, ROBOT_Z])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._held_steps = 0
        self._lifted = False

    def reset(self, seed: int | None = None, options: Optional[dict[str, Any]] = None) -> None:
        super().reset(seed, options)
        self._instruction = INSTRUCTION
        self._held_steps = 0
        self._lifted = False

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        super().compute_reward(info, *args, **kwargs)   # _seen and the demo weld; the pour flags stay unused
        bottle = info.get("target")
        up = bottle is not None and bottle[2] >= self._start[2] + LIFT_MIN
        self._held_steps = self._held_steps + 1 if up else 0
        if self._held_steps >= HOLD_S * self.metadata["render_hz"]:
            self._lifted = True
        return float(self._lifted)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self._lifted

    def decompose(self):
        """Stand, walk WALK forward, then the pour task's squat, grasp and lift, held at the top."""
        from simple.datagen.subtask_spec import StandSpec, WalkSpec

        pour = super().decompose()
        lift = next(i for i, s in enumerate(pour) if s.phase == "lift")
        pour[lift].meta["hold_steps"] = int((HOLD_S + 0.5) * self.metadata["render_hz"])
        return [pour[0],
                WalkSpec("walk", vx=WALK_VX, vy=0.0, target_yaw=0.0, target_distance=WALK_CMD),
                StandSpec("stop"),
                *pour[1:lift + 1]]
