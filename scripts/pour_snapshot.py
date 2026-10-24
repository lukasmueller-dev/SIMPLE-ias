# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Reset an env and save its cameras' first frames and the object poses: a quick scene check.

    .venv/bin/python scripts/pour_snapshot.py simple/G1WholebodyPourMP-v0 /out/snap [--seeds 0 1 2]
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
from PIL import Image

import simple.envs as _  # noqa: F401  registers the envs
from simple.core.action import ActionCmd

ap = argparse.ArgumentParser()
ap.add_argument("env_id")
ap.add_argument("out")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--steps", type=int, default=50, help="standing steps before the frames")
ap.add_argument("--overview", action="store_true", help="add two room cameras (pour task): behind the robot, across the table")
ap.add_argument("--light-scale", type=float, default=1.0, help="pour task: light intensity x this (calibration)")
ap.add_argument("--robot-dx", type=float, default=0.0, help="pour task: robot start moved this far along x (calibration)")
a = ap.parse_args()
if a.light_scale != 1.0 or a.robot_dx:
    from simple.dr.types import Box
    from simple.tasks.g1_wholebody_pour_mp import G1WholebodyPourMP as _P
    light, spatial = _P.dr_cfgs["lighting"], _P.dr_cfgs["spatial"]
    light.light_intensity = Box(low=light.light_intensity.low * a.light_scale, high=light.light_intensity.high * a.light_scale)
    lo, hi = list(spatial.robot_region.low), list(spatial.robot_region.high)
    lo[0] += a.robot_dx
    hi[0] += a.robot_dx
    spatial.robot_region = Box(low=lo, high=hi)
if a.overview:
    from simple.sensors import CameraCfg
    from simple.tasks.g1_wholebody_pour_mp import G1WholebodyPourMP as _T
    _view = lambda pos, eul: CameraCfg(uid="Realsense_D415_mono", mount="eye_on_base", width=640, height=360,  # noqa: E731
                                        focal_length=1.88, fov=np.deg2rad(75), near=0.1, far=12,
                                        pose=dict(position=pos, eulers=np.deg2rad(eul)))
    _T.sensor_cfgs = {**_T.sensor_cfgs, "behind": _view([-1.8, -1.0, 1.7], [0, 28, 28]),
                      "across": _view([2.2, 0.5, 1.5], [0, 18, 190])}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
env = gym.make(a.env_id, sim_mode="mujoco_isaac", headless=True, render_hz=30)
for seed in a.seeds:
    obs, info = env.reset(seed=seed)
    task = env.unwrapped.task
    mj = env.unwrapped.mujoco
    pelvis_reset = np.round(mj.mjData.qpos[:3], 3).tolist()
    for _ in range(a.steps):
        obs, *_rest, info = env.step(ActionCmd("loco_command", command=[0, 0, 0, 0, 0, 0, 0, 0], motion_type="stand"))
    pelvis_stand = np.round(mj.mjData.qpos[:3], 3).tolist()
    for k, v in obs.items():
        if getattr(v, "ndim", 0) == 3:
            Image.fromarray(np.asarray(v)).save(out / f"seed{seed}_{k}.png")
    poses = {k: np.round(np.asarray(v), 4).tolist() for k, v in info.items()}
    poses.update(pelvis_reset=pelvis_reset, pelvis_stand=pelvis_stand,
                 ground_z=float(mj.mjModel.geom("ground").pos[2]))
    (out / f"seed{seed}_poses.json").write_text(json.dumps(poses, indent=1))
env.close()
