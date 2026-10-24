# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Pour-task objects in the graspnet file layout, made by scripts/make_pour_assets.py."""

import os
from typing import Any

import numpy as np

from simple.assets.graspnet import Graspnet1BAsset
from simple.utils import resolve_data_path

from .asset_manager import AssetManager

Pour_Object_Names = {
    900: "coke zero bottle",
    901: "orange cup",
    # look variants (scripts/make_pour_assets.py): the shape, physics and grasps of 900 or 901, another USD
    902: "coke zero bottle empty", 920: "coke zero bottle full",
    910: "cola bottle full", 911: "cola bottle empty", 912: "orange soda bottle full", 913: "orange soda bottle empty",
    914: "lemon soda bottle", 915: "water bottle",
    916: "striped bottle a", 917: "striped bottle b", 918: "striped bottle c", 919: "striped bottle d",
    931: "red cup", 932: "blue cup", 933: "white cup", 934: "green cup", 935: "clear cup",
}
CUPS = {901, 931, 932, 933, 934, 935}


def shape_uid(uid: int) -> int:
    return 901 if uid in CUPS else 900


class PourAsset(Graspnet1BAsset):
    symmetric_z = True   # Bodex.load_cached_grasps: grasps ignore the yaw the scene draws
    friction = (1.5, 0.3, 0.01)   # MuJoCo sliding, torsional, rolling: Dex3 rubber on PET; the default 0.05 lets a held bottle pivot out

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "res_id": "pour"}


@AssetManager.register("pour")
class PourAssetManager(AssetManager):
    src_dir: str = "assets/pour"

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir

    def load(self, asset_id: str) -> PourAsset:
        uid = int(asset_id)
        assert uid in Pour_Object_Names, f"Asset ID {asset_id} not in {sorted(Pour_Object_Names)}"
        shape = shape_uid(uid)
        pieces = resolve_data_path(f"{self.src_dir}/models/{shape:03d}/coacd_0.05/")
        name = Pour_Object_Names[uid]
        asset = PourAsset(
            uid=str(uid),
            label="_".join(name.split(" ")),
            name=name,
            usd_path=resolve_data_path(f"{self.src_dir}/ruled_models/{uid:03d}_ruled.usd"),
            collision_mesh_curobo=resolve_data_path(f"{self.src_dir}/collision_models_mujoco/{shape:03d}/{shape:03d}_32x8.obj"),
            collision_meshes_mujoco=sorted(os.path.join(pieces, f) for f in os.listdir(pieces)
                                           if f.startswith("convex_piece")),
            description=name,
            stable_poses=np.load(resolve_data_path(f"{self.src_dir}/stable/{shape}_stable.npy")),
            canonical_grasps=[],
        )
        asset.grasp_uid = str(shape)   # simple.grasps.bodex: the shape's cached grasps
        return asset

    def sample(self, exclude: list[str] | None = None) -> PourAsset:
        import random
        ids = [str(i) for i in Pour_Object_Names if str(i) not in (exclude or [])]
        return self.load(random.choice(ids))

    def __len__(self) -> int:
        return len(Pour_Object_Names)

    def __iter__(self):
        return (self.load(str(i)) for i in Pour_Object_Names)
