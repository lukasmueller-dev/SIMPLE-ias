# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""G1 stands at a 120 x 60 x 63 cm OSB box table, grasps a Coke Zero bottle with the right hand,
pours into an orange cup on its left, puts the bottle back upright and lets go.

A twin of the IAS lab scene for fine-tuning psi0's real task3 checkpoint. The robot faces the 120 cm
side, centred; the bottle stands 22 cm from the edge in front of the right hand, the cup 11 cm to its left.
Success (SIMPLE_POUR_SUCCESS=pour, the default): the bottle axis tilts 90 deg or more with the cap
inside the cup's rim circle and above the rim, for 1 s. SIMPLE_POUR_SUCCESS=place (datagen) also
needs the bottle back upright on the table and out of the hand, so demos end after the release.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np
import transforms3d as t3d

from simple.core.actor import Actor, ObjectActor
from simple.core.randomizer import RandomizerCfg
from simple.dr import *
from simple.dr.types import Box
from simple.sensors import CameraCfg, SensorCfg, StereoCameraCfg
from simple.tasks.g1_wholebody_tabletop_grasp_mp import G1WholebodyTabletopGraspMP
from simple.tasks.registry import TaskRegistry

TABLE_DEPTH, TABLE_WIDTH, TABLE_HEIGHT = 0.60, 1.20, 0.63   # the OSB board's top, 2 cm thick
BOARD = 0.02
BOX_DEPTH, BOX_WIDTH = 0.40, 0.50              # plywood box under the board's middle
TABLE_X = 0.0                                  # table centre; near edge at TABLE_X - depth / 2
EDGE_X = TABLE_X - 0.5 * TABLE_DEPTH
ROBOT_X = EDGE_X - 0.12                        # pelvis 12 cm behind the edge, ~17 cm once AMO's stance settles (the real video)
ROBOT_Z = 0.785                                # G1Wholebody.z_offset: the robot frame is its pelvis
BOTTLE_XY = (EDGE_X + 0.22, -0.10)            # in front of the right hand, as in the lab photos: the right arm
CUP_XY = (EDGE_X + 0.22, 0.01)                 # cannot reach the midline at 63 cm; cup 11 cm left: the tilt alone swings the cap ~12 cm
JITTER = 0.03

BOTTLE_H, CUP_H, CUP_RIM_R = 0.215, 0.090, 0.037   # scripts/make_pour_assets.py
LIFT = 0.10
PLACE_GAP = 0.02                               # the return move stops this high over the table, the lower does the rest
CAP_CLEAR = 0.05                               # cap height over the rim while pouring
SQUAT = -0.12                                  # AMO height command: a 63 cm table is low for a standing G1
SQUEEZE = 0.50                                 # rad past the pregrasp fingers: the 6.5 cm bottle is thinner than the grasp's can
TILT = math.radians(98)                        # pour axis: this far from vertical, leaning left (toward the cup)
HOLD_S = 1.5
POUR_TILT_MIN, POUR_HOLD_S = math.radians(90), 1.0
HEAD_PITCH_UP = 9.0                            # deg, matched to wandb ra28b9f's first head frame

_OSB = {"path": "data/vMaterials_2/Wood/OSB_Wood.mdl", "name": "OSB_Wood"}
_PLASTIC = {"reflection_roughness_constant": 0.35, "metallic_constant": 0.0, "specular_level": 0.5}
_HAND = dict(grasp_type="bodex", hand_uid="dex3_right")
# after the grasp the waist stays where it is: free, the planner turns it and AMO's loose torso tracking moves the palm
_HELD = dict(_HAND, lock_links=["waist_roll_link", "torso_link", "left_hand_palm_link"])


@TaskRegistry.register("g1_wholebody_pour_mp")
class G1WholebodyPourMP(G1WholebodyTabletopGraspMP):
    uid: str = "g1_wholebody_pour_mp"
    label: str = "G1 Wholebody Pour MP"
    description: str = "The G1 grasps a bottle on the table, pours into the cup next to it and puts the bottle back."
    keep_ground_at_world_origin = True         # floor at z = 0 (pelvis spawned at ROBOT_Z); upstream puts the table top there
    # datagen only: the bottle is welded to the wrist from lift to put-down, so a 3-finger pinch cannot drop it mid-pour
    demo_weld = ("target", "right_wrist_yaw_link") if os.environ.get("SIMPLE_POUR_SUCCESS") == "place" else None

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,                       # the real G1 loop's rate
        "dr_level": 0,
        "version": 1.0,
        "need_gravity": True,
        "max_episode_steps": 1500,
    }

    robot_cfg: dict[str, Any] = dict(uid="g1_wholebody")

    # head: the real G1's RealSense D435i colour stream, 640 x 480, 55 deg horizontal
    sensor_cfgs: dict[str, SensorCfg] = dict(
        head_stereo=StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=480,
            focal_length=1.93,
            fov=np.deg2rad(55),
            near=0.1,
            far=5,
            baseline=0.05,
            # 9 deg up on the head link: the far table edge then sits where the real t=0 frame has it
            pose=dict(position=[0.0, 0.0, 0.0], quaternion=t3d.euler.euler2quat(0.0, math.radians(-HEAD_PITCH_UP), 0.0).tolist()),
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(instructions=["g1/pick_bottle_and_pour_into_cup"]),
        target=TargetDRCfg(asset_id="pour:900"),
        container=TargetDRCfg(asset_id="pour:901"),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[ROBOT_X, 0.0, ROBOT_Z], high=[ROBOT_X, 0.0, ROBOT_Z]),
            target_region=Box(low=[BOTTLE_XY[0] - JITTER, BOTTLE_XY[1] - JITTER],
                              high=[BOTTLE_XY[0] + JITTER, BOTTLE_XY[1] + JITTER]),
            container_region=Box(low=[CUP_XY[0] - JITTER, CUP_XY[1] - JITTER],
                                 high=[CUP_XY[0] + JITTER, CUP_XY[1] + JITTER]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=-math.pi, high=math.pi),   # label heading; grasps ignore it
            collide_threshold=0.03,
        ),
        camera=CameraDRCfg(cam_id="franka_camera"),
        scene=TabletopSceneDRCfg(
            table_size=Box(low=[TABLE_DEPTH, TABLE_WIDTH, BOARD], high=[TABLE_DEPTH, TABLE_WIDTH, BOARD]),
            table_position=Box(low=[TABLE_X, 0.0], high=[TABLE_X, 0.0]),
            table_height=Box(low=TABLE_HEIGHT, high=TABLE_HEIGHT),
            enable_table2=True,
            table2_size=Box(low=[BOX_DEPTH, BOX_WIDTH, TABLE_HEIGHT - BOARD], high=[BOX_DEPTH, BOX_WIDTH, TABLE_HEIGHT - BOARD]),
            table2_position=Box(low=[TABLE_X, 0.0], high=[TABLE_X, 0.0]),
            table2_height=Box(low=TABLE_HEIGHT - BOARD, high=TABLE_HEIGHT - BOARD),
            table2_rotation_z=Box(low=0.0, high=0.0),
            room_choices=["hssd:scene0", "hssd:scene1", "hssd:scene2", "hssd:scene4", "hssd:scene6"],
            scene_manager="hssd",
        ),
        lighting=LightingDRCfg(
            light_mode="random",
            light_num=(2, 3),
            light_color_temperature=Box(low=3000, high=7000),
            light_intensity=Box(low=1e4 * 0.7, high=1e4 * 1.3),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 1.1),
            light_spacing=Box((1.0, 1.0), (2.0, 2.0)),
            light_position=Box((-0.8, -0.8, 1.6), (0.8, 0.8, 2.2)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        # the table is OSB; ground and room shaders still randomise
        material=MaterialDRCfg(material_mode="rand_all", table_material=_OSB),
    )

    def __init__(self, *args, target_object: str | None = "pour:900", **kwargs):
        super().__init__(*args, target_object=target_object or "pour:900", **kwargs)
        self._container = None
        self._pour_steps = 0
        self._poured = False
        self._seen = {}
        self._palm_lifted = None
        self._weld_done = False
        self._start = np.zeros(3)

    @property
    def container(self) -> Actor:
        assert self._container is not None, "call reset() first"
        return self._container

    def reset(self, seed: int | None = None, options: Optional[dict[str, Any]] = None) -> None:
        super().reset(seed, options)
        self._instruction = "g1/pick_bottle_and_pour_into_cup"   # psi0's task string, as the real suite reads it
        self._container = self.layout.actors.get("container")
        self._start = np.array(self.layout.actors["target"].pose.position, dtype=float)   # the layout follows the sim later
        if "table2" in self.layout.actors:          # MaterialDR dresses the board only
            self.layout.actors["table2"].set_material(_OSB)
        for key in ("target", "container"):          # PET and PP, never metal: MaterialDR draws metallic up to 1
            self.layout.actors[key].set_material(_PLASTIC)
        self._pour_steps = 0
        self._poured = False
        self._seen = {}
        self._palm_lifted = None
        self._weld_done = False

    def preload_objects(self) -> list[Actor]:
        from simple.assets import AssetManager
        return [ObjectActor(asset=asset) for asset in AssetManager.get("pour")]

    # ------------------------------------------------------------------ success

    @staticmethod
    def _axis_and_cap(pose7):
        R = t3d.quaternions.quat2mat(pose7[3:7])
        return R[:, 2], np.asarray(pose7[:3]) + R[:, 2] * 0.5 * BOTTLE_H

    def _in_hand(self, mujoco_env) -> bool:
        if mujoco_env is None:
            return False
        data, model = mujoco_env.mjData, mujoco_env.mjModel
        label = str(self.target.asset.label)
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.body(model.geom(c.geom1).bodyid).name
            b2 = model.body(model.geom(c.geom2).bodyid).name
            if (label in b1 and "hand" in b2) or (label in b2 and "hand" in b1):
                return True
        return False

    def _update_weld(self, mujoco_env, bottle):
        """demo_weld on once the bottle is held and 1 cm up, off for good once it is back at table height."""
        if self.demo_weld is None or mujoco_env is None or bottle is None or self._weld_done:
            return
        import mujoco
        m, d = mujoco_env.mjModel, mujoco_env.mjData
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "demo_weld")
        if i < 0:
            return
        start_z = self._start[2]
        if not d.eq_active[i]:
            if bottle[2] > start_z + 0.01 and self._in_hand(mujoco_env):
                b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
                q1 = np.asarray(d.xquat[b1])
                m.eq_data[i][0:3] = 0.0
                m.eq_data[i][3:6] = t3d.quaternions.quat2mat(q1).T @ (np.asarray(d.xpos[b2]) - np.asarray(d.xpos[b1]))
                m.eq_data[i][6:10] = t3d.quaternions.qmult(t3d.quaternions.qinverse(q1), np.asarray(d.xquat[b2]))
                m.eq_data[i][10] = 1.0
                d.eq_active[i] = 1
        elif self._poured and abs(bottle[2] - start_z) < 0.012 and self._axis_and_cap(bottle)[0][2] > math.cos(math.radians(10)):
            d.eq_active[i] = 0
            self._weld_done = True

    def _palm(self, mujoco_env):
        """World position of the planner's right palm link (URDF: right_wrist_yaw_link + 4.15 cm), or None."""
        if mujoco_env is None:
            return None
        model, data = mujoco_env.mjModel, mujoco_env.mjData
        if getattr(self, "_wrist_id", None) is None:
            names = [model.body(i).name for i in range(model.nbody)]
            self._wrist_id = next(i for i, n in enumerate(names) if n.endswith("right_wrist_yaw_link"))
        R = np.asarray(data.xmat[self._wrist_id]).reshape(3, 3)
        return np.asarray(data.xpos[self._wrist_id]) + R @ np.array([0.0415, -0.003, 0.0])

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        bottle, cup = info.get("target"), info.get("container")
        self._seen = {"target": bottle, "container": cup, "palm": self._palm(kwargs.get("mujoco_env"))}
        self._update_weld(kwargs.get("mujoco_env"), bottle)
        if bottle is None or cup is None:
            return 0.0
        axis, cap = self._axis_and_cap(bottle)
        tilt = math.acos(float(np.clip(axis[2], -1.0, 1.0)))
        rim = np.asarray(cup[:3]) + [0.0, 0.0, 0.5 * CUP_H]
        over_cup = np.linalg.norm(cap[:2] - rim[:2]) <= CUP_RIM_R and cap[2] >= rim[2]
        self._pour_steps = self._pour_steps + 1 if (tilt >= POUR_TILT_MIN and over_cup) else 0
        if self._pour_steps >= POUR_HOLD_S * self.metadata["render_hz"]:
            self._poured = True
        return float(self._poured)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        if not self._poured:
            return False
        if os.environ.get("SIMPLE_POUR_SUCCESS", "pour") != "place":
            return True
        axis, _ = self._axis_and_cap(info["target"])
        upright = axis[2] >= math.cos(math.radians(15))
        on_table = abs(info["target"][2] - (TABLE_HEIGHT + 0.5 * BOTTLE_H)) < 0.02
        return bool(upright and on_table and not self._in_hand(kwargs.get("mujoco_env")))

    # ------------------------------------------------------------------ demo

    def decompose(self):
        """Re-read at each phase break with the bottle where the sim has it (compute_reward caches it)."""
        from simple.datagen.subtask_spec import (GraspObjectSpec, HeightAdjustSpec, LiftSpec, LowerSpec,
                                                 MoveEEFToPoseSpec, OpenGripperSpec, PhaseBreakSpec,
                                                 RetreatSpec, StandSpec)

        start = self._start                                           # spawn pose: the layout follows the sim
        now = self._seen.get("target")
        centre = np.asarray(now[:3], dtype=float) if now is not None else start
        axis = self._axis_and_cap(now)[0] if now is not None else np.array([0.0, 0.0, 1.0])
        seen_cup = self._seen.get("container")
        cup = np.asarray(seen_cup[:3] if seen_cup is not None else self.container.pose.position, dtype=float)
        rim = cup + [0.0, 0.0, 0.5 * CUP_H]
        h = 0.5 * BOTTLE_H
        cap = centre + axis * h
        palm = self._seen.get("palm")
        palm = palm if palm is not None else centre
        # the tilt turns the bottle's present axis onto the pour axis: whatever lean the grip left is absorbed
        pour_axis = np.array([0.0, math.sin(TILT), math.cos(TILT)])
        k = np.cross(axis, pour_axis)
        angle = math.atan2(np.linalg.norm(k), float(np.dot(axis, pour_axis)))
        R = t3d.axangles.axangle2mat(k / np.linalg.norm(k), angle) if np.linalg.norm(k) > 1e-9 else np.eye(3)
        tilt_rpy, untilt_rpy = t3d.euler.mat2euler(R), t3d.euler.mat2euler(R.T)
        cap_tilted = palm + R @ (cap - palm)              # where the tilt about the palm takes the cap
        target_cap = rim + [0.0, 0.0, CAP_CLEAR]
        to_cup = target_cap - cap_tilted                   # move first, then tilt: the cap lands over the rim
        # level: turn the axis back onto z (tilt x first, then y); a few degrees are left as they are
        level = (math.atan2(axis[1], axis[2]), -math.asin(float(np.clip(axis[0], -1.0, 1.0))), 0.0)
        if math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0)))) < 3.0:
            level = (0.0, 0.0, 0.0)
        # back: the palm returns to where it stood after the lift, a pose this locked waist reached
        if self._palm_lifted is None and centre[2] > start[2] + 0.06 and palm is not centre:
            self._palm_lifted, self._z_lifted = np.array(palm), float(centre[2])
        back = (self._palm_lifted - palm) if self._palm_lifted is not None else np.zeros(3)
        if self._palm_lifted is not None:                  # and down to 2 cm over the bottle's spot on the table
            back = back + [0.0, 0.0, -(self._z_lifted - start[2] - PLACE_GAP)]
        hold = int(HOLD_S * self.metadata["render_hz"])
        return [
            StandSpec("initialize"),
            HeightAdjustSpec("squat", height=SQUAT, keep_waist_pose=True),
            PhaseBreakSpec("settle", grasp_type="bodex"),
            GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False, squeeze=SQUEEZE,
                            lock_links=["left_hand_palm_link"], **_HAND),   # waist free: the torso leans in, as task3's data does
            LiftSpec("lift", up=LIFT, **_HAND),
            PhaseBreakSpec("lifted", grasp_type="bodex"),
            MoveEEFToPoseSpec("to_cup", delta_position=tuple(to_cup), **_HELD),
            PhaseBreakSpec("near_cup", grasp_type="bodex"),
            MoveEEFToPoseSpec("align", delta_position=tuple(to_cup), **_HELD),      # the rest, waist still
            MoveEEFToPoseSpec("tilt", delta_rpy=tilt_rpy, hold_steps=hold, **_HELD),
            MoveEEFToPoseSpec("untilt", delta_rpy=untilt_rpy, **_HELD),
            PhaseBreakSpec("poured", grasp_type="bodex"),
            MoveEEFToPoseSpec("level", delta_rpy=level, **_HELD),
            MoveEEFToPoseSpec("back", delta_position=tuple(back), **_HAND),      # waist free: the weld holds, placing is coarse
            LowerSpec("lower", down=PLACE_GAP - 0.004, **_HAND),
            OpenGripperSpec("release", **_HAND),
            RetreatSpec("retreat", **_HAND),
        ]
