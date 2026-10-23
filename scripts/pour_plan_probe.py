# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Why does the bottle approach fail? Calls the planner's BODex approach for every cached grasp with
finger targets and locked links on or off, after the demo's stand and squat. Writes <out>.json.

    .venv/bin/python scripts/pour_plan_probe.py /out/planprobe [--squat -0.10]
"""

import argparse
import json
import os
from copy import deepcopy

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import transforms3d as t3d

import simple.envs as _  # noqa: F401
from simple.core.action import ActionCmd

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--squat", type=float, default=-0.10)
ap.add_argument("--bottle-y", type=float, nargs="+", default=[None], help="planning-only bottle y (world) to try")
ap.add_argument("--trials", type=int, default=3)
a = ap.parse_args()

env = gym.make("simple/G1WholebodyPourMP-v0", sim_mode="mujoco_isaac", headless=True, render_hz=30)
env.reset(seed=0)
task = env.unwrapped.task
for _ in range(60):
    env.step(ActionCmd("loco_command", command=[0, 0, 0, 0, 0, 0, 0, 0], motion_type="stand"))
for _ in range(60):
    env.step(ActionCmd("loco_command", command=[0, 0, 0, a.squat, 0, 0, 0, 0], desired_height=a.squat,
                       motion_type="height_adjust", keep_waist_pose=True))

from simple.grasps.bodex import Bodex
from simple.mp.curobo import CuRoboPlanner

robot = task.robot
LOCK = ["waist_roll_link", "torso_link", "left_hand_palm_link"]
out = {"cases": []}
for easy, bottle_y in [(False, y) for y in a.bottle_y]:
    ignore_target = True
    planner = CuRoboPlanner(robot=robot, plan_dt=0.01, plan_batch_size=1, ignore_target_collisions=True, easy_motion_gen=easy)
    robot.update_ee_link("dex3_right")
    planner.reinit(robot)
    for trial in range(a.trials):
        robot_actor, target = task.layout.actors["robot"], task.layout.actors["target"]
        if bottle_y is not None:
            target.pose.position = [target.pose.position[0], bottle_y, target.pose.position[2]]
        T_robot = robot_actor.pose.as_matrix()
        g = Bodex.load_cached_grasps(robot, target.asset, target_pose=target.pose, max_grasps=1, hand_uid="dex3_right")[0]
        pose = robot.get_grasp_pose_wrt_robot(g, pregrasp=False, robot_pose=T_robot)
        links = {}
        for k, lp in g["link_poses"].items():                      # world -> robot frame, as mp.py does
            Tw = np.eye(4)
            Tw[:3, 3] = lp.position.cpu().numpy().squeeze(0)
            Tw[:3, :3] = t3d.quaternions.quat2mat(lp.quaternion.cpu().numpy().squeeze(0))
            Tr = np.linalg.inv(T_robot) @ Tw
            lp = deepcopy(lp)
            import torch
            lp.position = torch.from_numpy(Tr[:3, 3]).float().cuda().unsqueeze(0)
            lp.quaternion = torch.from_numpy(t3d.quaternions.mat2quat(Tr[:3, :3])).float().cuda().unsqueeze(0)
            links[k] = lp
        row = {"easy": easy, "bottle_y": bottle_y, "trial": trial, "goal": np.round(pose[0], 3).tolist(),
               "links": sorted(links)}
        import copy, torch
        solver = planner.init_js_ik_solver
        qn = robot.get_robot_qpos()
        ee = solver.fk(torch.tensor([[qn[n] for n in solver.joint_names]], device="cuda", dtype=torch.float32)).ee_pose
        cur = np.concatenate([ee.position.cpu().numpy()[0], ee.quaternion.cpu().numpy()[0]])
        fwd = cur.copy(); fwd[0] += 0.08
        grasp = np.concatenate([pose[0], pose[1]])
        empty = copy.copy(task.layout)
        empty.actors = {k: v for k, v in task.layout.actors.items() if k == "robot"}
        d = grasp[:3] - cur[:3]
        row["d"] = np.round(d, 3).tolist()
        cases = [("full_world", grasp, links, task.layout), ("palm_world", grasp, {}, task.layout),
                 ("full_empty", grasp, links, empty), ("palm_empty", grasp, {}, empty)]
        row["cur"] = np.round(cur, 3).tolist()
        for name, goal, link_poses, world in cases:
            try:
                planner.batch_plan_for_approach_bodex(
                    [goal], [g["grasp_qpos"]], [g["squeeze_qpos"]], [g["lift_qpos"]],
                    link_poses=deepcopy(link_poses), debug_grasp=g, world_layout=world,
                    lock_links_names=LOCK, current_joint_qpos=robot.get_robot_qpos())
                row[name] = "ok"
            except Exception as e:
                row[name] = str(e)[:120]
        out["cases"].append(row)
        with open(a.out + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
env.close()
