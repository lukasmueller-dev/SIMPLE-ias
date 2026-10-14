# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np

from simple.core.randomizer import RandomizerCfg
from simple.dr import *
from simple.dr.types import Box
from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from simple.tasks.registry import TaskRegistry

# 0.8 m square table top; its near edge stays where XMovePick's 1.25 m table has it (x = 0.3 - 1.25 / 2).
_TABLE_SIDE = 0.8
_TABLE_X = 0.3 - 0.5 * 1.25 + 0.5 * _TABLE_SIDE

_WHITE_TABLE = {
    "path": "data/vMaterials_2/Paint/Paint_Eggshell.mdl",
    "name": "Paint_Eggshell_White",
}


@TaskRegistry.register("g1_wholebody_white_table_pick_teleop")
class G1WholebodyWhiteTablePickTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    """XMovePick on a white square table with one object and no distractors.

    Robot, target, regions, instruction and success predicate are XMovePick's,
    so the released g1-xmovepick checkpoint runs on it unchanged.
    """

    uid: str = "g1_wholebody_white_table_pick_teleop"
    label: str = "G1 TELEOP White Table Pick"
    description: str = (
        "A task where the G1 robot must move forward to pick up one object from a white square table."
    )

    metadata: dict[str, Any] = dict(G1WholebodyXMovePickTaskTeleop.metadata)

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(
            instructions=[
                "move forward to pick up the {}",
            ]
        ),
        target=TargetDRCfg(asset_id="graspnet1b:12"),
        distractors=DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=0,
        ),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.9, -0.0, 0.0], high=[-1.0, 0.0, 0.0]),
            target_region=Box(low=[-0.3, -0.08], high=[-0.25, -0.04]),
            distractors_region=Box(low=[-0.2, -0.3], high=[0.4, 0.3]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=-0.15, high=0.15),
        ),
        camera=CameraDRCfg(
            cam_id="franka_camera",
        ),
        scene=TabletopSceneDRCfg(
            table_size=Box(low=[_TABLE_SIDE, _TABLE_SIDE, 0.1], high=[_TABLE_SIDE, _TABLE_SIDE, 0.1]),
            table_position=Box(low=[_TABLE_X, 0], high=[_TABLE_X, 0]),
            table_height=Box(low=0.58, high=0.45),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
        ),
        lighting=LightingDRCfg(
            light_mode="random",
            light_num=(2, 3),
            light_color_temperature=Box(low=2001, high=8001),
            light_intensity=Box(low=1e4 * 0.8, high=1e4 * 1.2),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 1.1),
            light_spacing=Box((1.0, 1.0), (2.0, 2.0)),
            light_position=Box((-0.5, -0.5, 1.3), (0.5, 0.5, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        # table_material pins the table; ground and object shaders still randomise
        material=MaterialDRCfg(
            material_mode="rand_all",
            table_material=_WHITE_TABLE,
        ),
    )
