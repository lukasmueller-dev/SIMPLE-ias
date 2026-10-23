# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Dex3 side grasps for the pour bottle (pour:900): hand upright, palm facing the bottle from its right.

    .venv/bin/python scripts/make_pour_grasps.py [--pre-close 0.2]

The palm frame of the right Dex3 is the robot frame at the ready pose (x along the fingers, y out of
the palm, z up), so an upright hand is easy for the arm. Finger joints and the hand-to-object offset
come from graspnet's chips can grasp 0 (BODex), with the axis made vertical and the offset shrunk to
the bottle radius. Written in BODex's world frame, object at [0.4, 0, stable z], to
data/assets/graspnet/dex_grasp/dex3_right/900/.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import transforms3d as t3d

from simple.utils import resolve_data_path

SRC, DST, HAND = "10", "900", "dex3_right"
CAN_R, BOTTLE_R, BOTTLE_H = 0.0402, 0.0325, 0.215
ALONG, OUT = 0.055, 0.061          # bottle axis in the can grasp's palm frame: along the fingers, out of the palm
GRASP_Z = 0.090                    # palm height above the bottle base
PRE_GAP = 0.015                    # the pregrasp palm stands this much further out
YAWS = np.deg2rad([-20, -10, 0, 10, 20, 30])   # about the bottle axis; 0 = fingers straight ahead
CLOSE_DIR = np.array([0.0, -1, -1, 1, 1, 1, 1])  # BODex joint order; SIMPLE's squeeze_dir, reordered


def main(pre_close):
    src = Path(resolve_data_path(f"assets/graspnet/dex_grasp/{HAND}/{SRC}"))
    dst = Path(resolve_data_path(f"assets/graspnet/dex_grasp/{HAND}")) / DST
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    joints = {k: torch.load(src / f"{SRC}__0{k}.pt", map_location="cpu").numpy()[7:] + pre_close * CLOSE_DIR
              for k in ("_pregrasp", "")}
    meta = np.load(src / f"{SRC}__0.npy", allow_pickle=True).item()
    meta["robot_pose"] = meta["robot_pose"].copy()
    meta["robot_pose"][..., -7:] += pre_close * CLOSE_DIR
    axis = np.array([0.4, 0.0, GRASP_Z])                          # bottle axis at palm height, BODex world
    out_dist = OUT - (CAN_R - BOTTLE_R)
    for i, yaw in enumerate(YAWS):
        R = t3d.euler.euler2mat(0, 0, yaw)
        for k, gap in (("_pregrasp", PRE_GAP), ("", 0.0)):
            p = axis - R @ np.array([ALONG, out_dist + gap, 0.0])
            x = np.concatenate([p, t3d.quaternions.mat2quat(R), joints[k]]).astype(np.float32)
            t = torch.from_numpy(x)
            torch.save(t.cuda() if torch.cuda.is_available() else t, dst / f"{DST}__{i}{k}.pt")
        np.save(dst / f"{DST}__{i}.npy", meta, allow_pickle=True)
        print(f"{DST}__{i}: yaw {np.rad2deg(yaw):+.0f}, palm {np.round(axis - R @ [ALONG, out_dist, 0], 3).tolist()}")
    print(f"{len(YAWS)} grasps -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-close", type=float, default=0.2)
    main(ap.parse_args().pre_close)
