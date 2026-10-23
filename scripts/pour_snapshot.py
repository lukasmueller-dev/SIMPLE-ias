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
a = ap.parse_args()
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
