# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Which pour-bottle grasps can the right arm reach? Palm-only IK per cached grasp, turned about the
bottle axis in steps, after the demo's stand and squat. Writes <out>.json (Isaac swallows stdout).

    .venv/bin/python scripts/pour_ik_probe.py /out/ikprobe [--squat -0.10] [--step 15]
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import torch
import transforms3d as t3d

import simple.envs as _  # noqa: F401
from simple.core.action import ActionCmd
from simple.utils import resolve_data_path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--squat", type=float, default=-0.10)
ap.add_argument("--step", type=int, default=15)
a = ap.parse_args()

env = gym.make("simple/G1WholebodyPourMP-v0", sim_mode="mujoco_isaac", headless=True, render_hz=30)
env.reset(seed=0)
task = env.unwrapped.task
for _ in range(60):
    env.step(ActionCmd("loco_command", command=[0, 0, 0, 0, 0, 0, 0, 0], motion_type="stand"))
for _ in range(60):
    env.step(ActionCmd("loco_command", command=[0, 0, 0, a.squat, 0, 0, 0, 0], desired_height=a.squat,
                       motion_type="height_adjust", keep_waist_pose=True))

from curobo.types.math import Pose
from simple.mp.curobo import CuRoboPlanner

robot = task.robot
planner = CuRoboPlanner(robot=robot, plan_dt=0.01, plan_batch_size=1)
robot.update_ee_link("dex3_right")
planner.reinit(robot)
solver = planner.init_js_ik_solver
T_world_robot = task.layout.actors["robot"].pose.as_matrix()
target = task.layout.actors["target"]
T_world_object = target.pose.as_matrix()
T_world_object[:3, :3] = np.eye(3)                                     # symmetric_z
stable = target.asset.stable_poses[0]
T_bw_object = np.eye(4)
T_bw_object[:3, 3] = [0.4, 0.0, stable[2]]

# where the palm is now (ready pose), in the robot frame
q_now = robot.get_robot_qpos()
q = torch.tensor([[q_now[n] for n in solver.joint_names]], device="cuda", dtype=torch.float32)
ee = solver.fk(q).ee_pose
now = {"position": ee.position.cpu().numpy()[0].tolist(), "quaternion": ee.quaternion.cpu().numpy()[0].tolist()}

folder = resolve_data_path(f"assets/graspnet/dex_grasp/dex3_right/{target.asset.uid}/")
files = sorted(f for f in os.listdir(folder) if f.endswith("_pregrasp.pt"))
result = {"robot": T_world_robot[:3, 3].tolist(), "bottle": T_world_object[:3, 3].tolist(), "palm_now": now, "grasps": {}}
for f in files:
    pre = torch.load(os.path.join(folder, f), map_location="cpu").numpy()
    T_bw = np.eye(4)
    T_bw[:3, 3], T_bw[:3, :3] = pre[:3], t3d.quaternions.quat2mat(pre[3:7])
    T_obj_grasp = np.linalg.inv(T_bw_object) @ T_bw
    rows = []
    for yaw in range(-180, 180, a.step):
        R = np.eye(4)
        R[:3, :3] = t3d.euler.euler2mat(0, 0, np.deg2rad(yaw))
        T_robot_goal = np.linalg.inv(T_world_robot) @ T_world_object @ R @ T_obj_grasp
        goal = Pose(position=torch.tensor(T_robot_goal[None, :3, 3], device="cuda", dtype=torch.float32),
                    quaternion=torch.tensor(t3d.quaternions.mat2quat(T_robot_goal[:3, :3])[None], device="cuda",
                                            dtype=torch.float32))
        res = solver.solve_single(goal)
        rows.append({"yaw": yaw, "ok": bool(res.success.any().item()),
                     "goal": np.round(T_robot_goal[:3, 3], 3).tolist()})
    result["grasps"][f] = rows
with open(a.out + ".json", "w") as fh:
    json.dump(result, fh, indent=1)
env.close()
