# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""One pour demo with datagen's planner loop, kept whatever happens: frames of the head and side
cameras every 3rd step, and a per-step trace of the bottle's tilt and the cap's distance to the rim.

    .venv/bin/python scripts/pour_debug_episode.py /out/debug [--seed 0] [--ignore-target-collision]
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
from PIL import Image

import simple.envs as _  # noqa: F401
from simple.agents.mp import MotionPlannerAgent
from simple.mp.curobo import CuRoboPlanner
from simple.tasks import g1_wholebody_pour_mp as pour

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ignore-target-collision", action="store_true")
a = ap.parse_args()
out = Path(a.out)
(out / "frames").mkdir(parents=True, exist_ok=True)

env = gym.make("simple/G1WholebodyPourMP-v0", sim_mode="mujoco_isaac", headless=True, render_hz=30,
               max_episode_steps=1500)
task = env.unwrapped.task
import sys
sys.stdout = open(out / "stdout.log", "w", buffering=1)   # Isaac keeps print() in its kit log otherwise
planner = CuRoboPlanner(robot=task.robot, plan_dt=0.01, plan_batch_size=1,
                        ignore_target_collisions=a.ignore_target_collision)
agent = MotionPlannerAgent(task, planner)
obs, info = env.reset(seed=a.seed)
trace, step, phase, events = [], 0, 1, []


def record(obs, info):
    global step
    b, c = info.get("target"), info.get("container")
    if b is not None and c is not None:
        axis, cap = task._axis_and_cap(b)
        rim = np.asarray(c[:3]) + [0, 0, 0.5 * pour.CUP_H]
        trace.append({"step": step, "phase": phase, "bottle": np.round(b[:3], 3).tolist(),
                      "tilt_deg": round(math.degrees(math.acos(float(np.clip(axis[2], -1, 1)))), 1),
                      "cap_rim_xy": round(float(np.linalg.norm(cap[:2] - rim[:2])), 3),
                      "cap_above_rim": round(float(cap[2] - rim[2]), 3), "poured": task._poured})
    if step % 3 == 0:
        for k in ("head_stereo_left", "side_left"):
            if k in obs:
                Image.fromarray(np.asarray(obs[k])[..., :3]).save(out / "frames" / f"{k}_{step:05d}.png")
    step += 1


record(obs, info)
done = False
while not done:
    state = agent.synthesize()
    names = [sp.phase for sp in task.decompose()]
    events.append({"step": step, "synthesize": str(state), "spec": names[agent._subtask_index - 1]})
    if state is False:
        break
    while True:
        try:
            action = agent.get_action(obs, info)
        except StopIteration:
            events.append({"step": step, "event": "queue empty"})
            break
        obs, reward, terminated, truncated, info = env.step(action)
        record(obs, info)
        if terminated or truncated:
            events.append({"step": step, "event": "terminated" if terminated else "truncated"})
            done = True
            break
    if state != "phase_break":
        break
    phase += 1
(out / "trace.json").write_text(json.dumps({"events": events, "trace": trace}, indent=0))
env.close()
