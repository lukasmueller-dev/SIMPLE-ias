# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Pour demos into a LeRobot dataset: datagen's loop with the pour debug driver's env and agent setup.

    SIMPLE_POUR_SUCCESS=place .venv/bin/python scripts/pour_datagen.py /out/<id>/raw/block-0 --episodes 40

Keeps successful episodes only (the recorder drops the rest); progress goes to <save_dir>/progress.jsonl.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

import simple.envs as _  # noqa: F401
from simple.agents.mp import MotionPlannerAgent
from simple.envs.lerobot import LerobotRecorder
from simple.mp.curobo import CuRoboPlanner

ap = argparse.ArgumentParser()
ap.add_argument("save_dir")
ap.add_argument("--episodes", type=int, default=40, help="successes to keep")
ap.add_argument("--max-attempts", type=int, default=150)
ap.add_argument("--keep-failed", action="store_true", help="debug: record failed attempts too")
ap.add_argument("--env", default="simple/G1WholebodyPourMP-v0", help="also simple/G1WholebodyWalkLiftMP-v0")
a = ap.parse_args()
out = Path(a.save_dir)
out.mkdir(parents=True, exist_ok=True)

env = gym.make(a.env, sim_mode="mujoco_isaac", headless=True, render_hz=30, max_episode_steps=1500)
task = env.unwrapped.task
sys.stdout = open(out / "stdout.log", "a", buffering=1)   # Isaac keeps print() in its kit log otherwise
planner = CuRoboPlanner(robot=task.robot, plan_dt=0.01, plan_batch_size=1)
agent = MotionPlannerAgent(task, planner)
env = LerobotRecorder(env=env, agent=agent, shard_size=100, root_dir=str(out))
progress = open(out / "progress.jsonl", "a", buffering=1)

kept = attempts = 0
while kept < a.episodes and attempts < a.max_attempts:
    attempts += 1
    t0 = time.time()
    obs, info = env.reset(options={"state_dict": None})
    result, failed_spec = "plan_failed", None
    while True:
        state = agent.synthesize()
        if state is False:
            failed_spec = agent._subtask_index
            break
        ended = False
        while True:
            try:
                action = agent.get_action(obs, info)
            except StopIteration:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                result, ended = ("success" if terminated else "truncated"), True
                break
        if ended:
            break
        bottle = getattr(task, "_seen", {}).get("target")
        up = None if bottle is None else round(float(bottle[2] - task._start[2]), 3)
        print(f"[datagen] attempt {attempts} spec {agent._subtask_index}: robot "
              f"{[round(float(v), 3) for v in task.layout.actors['robot'].pose.position[:3]]}, bottle up {up}")
        if state != "phase_break":
            result = "no_success"
            break
    if result == "success":
        kept += 1
    elif a.keep_failed:          # debug: the failed attempt as an episode, for its video
        env.dataset.save_episode()
    else:
        env.clear_episode_buffer()
    progress.write(json.dumps({"attempt": attempts, "result": result, "failed_spec": failed_spec,
                               "kept": kept, "seconds": round(time.time() - t0, 1)}) + "\n")
env.close()
