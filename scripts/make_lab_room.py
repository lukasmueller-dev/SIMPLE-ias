# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""The IAS lab room for the pour task: terrazzo patch with a rubber-mat band, light grey floor, white walls and
cabinets, white desks, black office chairs, tripod, plywood box, white cloth behind the board. Visual only (Isaac).

    uv run --with usd-core --with pillow --with numpy python scripts/make_lab_room.py <floor photo> <data>/scenes/lab

Writes lab<k>/lab<k>.usd (+ textures/) for k in VARIANTS, in the HSSD scene layout (`/World/surface` marks the
board; `/World/ceilings`), registered as hssd:lab<k> in resources/hssd-scenes/config.yaml. Z up, metres, the
board centred at the origin, the robot on -x facing +x. Variants differ in floor tint and texture scale, wall
tint, the mat band and where the props stand.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

VARIANTS = 6
TILE_M = 0.5                  # floor texture tile edge, metres (chips ~1 cm, as in the photos)
CROP = (1900, 3300, 800)      # x, y, size in the photo (IMG_6130 at 5712 x 4284): no cable, little perspective
ROOM_X, ROOM_Y, ROOM_H = (-3.5, 3.6), (-4.0, 4.0), 2.9
PATCH_X, PATCH_Y = (-2.6, 2.1), (-2.4, 2.4)   # terrazzo around the board
BAND = 0.18                                    # grey interlocking mat band around the patch


def seamless_tile(photo, out):
    x, y, s = CROP
    tile = np.asarray(Image.open(photo).convert("RGB"), dtype=np.float32)[y:y + s, x:x + s]
    rolled = np.roll(tile, (s // 2, s // 2), axis=(0, 1))   # its edges are the crop's middle: they wrap
    d = np.minimum.outer(np.minimum(np.arange(s), np.arange(s)[::-1]), np.minimum(np.arange(s), np.arange(s)[::-1]))
    m = np.clip(d / (0.25 * s), 0.0, 1.0)[..., None]
    Image.fromarray((tile * m + rolled * (1 - m)).astype(np.uint8)).save(out)


def material(stage, name, color=None, texture=None, rough=0.7, scale=1.0):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/{name}")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if color is not None:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if texture is not None:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(scale, scale, scale))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def box(stage, path, center, size, mat, rot_z=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateZOp().Set(float(rot_z))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return cube


def quad(stage, path, x, y, z, mat, tile=None):
    """Horizontal rectangle x[0]..x[1] by y[0]..y[1] at height z; UVs repeat every `tile` metres."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [(x[0], y[0], z), (x[1], y[0], z), (x[1], y[1], z), (x[0], y[1], z)]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    if tile:
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        uv.Set(Vt.Vec2fArray([Gf.Vec2f(p[0] / tile, p[1] / tile) for p in pts]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def chair(stage, path, x, y, yaw, black):
    c, s = np.cos(yaw), np.sin(yaw)
    at = lambda dx, dy, z: (x + c * dx - s * dy, y + s * dx + c * dy, z)   # noqa: E731
    deg = float(np.degrees(yaw))
    box(stage, f"{path}/seat", at(0, 0, 0.48), (0.50, 0.50, 0.07), black, deg)
    box(stage, f"{path}/back", at(-0.24, 0, 0.82), (0.05, 0.46, 0.58), black, deg)
    box(stage, f"{path}/post", at(0, 0, 0.27), (0.05, 0.05, 0.40), black, deg)
    for k in range(5):
        a = yaw + k * 2 * np.pi / 5
        box(stage, f"{path}/leg{k}", (x + 0.15 * np.cos(a), y + 0.15 * np.sin(a), 0.06), (0.30, 0.04, 0.04),
            black, float(np.degrees(a)))
    for side in (-1, 1):
        box(stage, f"{path}/arm{side + 1}", at(0, side * 0.27, 0.66), (0.30, 0.04, 0.04), black, deg)


def desk(stage, path, x, y, yaw, white, dark):
    c, s = np.cos(yaw), np.sin(yaw)
    deg = float(np.degrees(yaw))
    box(stage, f"{path}/top", (x, y, 0.735), (1.60, 0.80, 0.03), white, deg)
    for i, (dx, dy) in enumerate([(-0.75, -0.35), (0.75, -0.35), (-0.75, 0.35), (0.75, 0.35)]):
        box(stage, f"{path}/leg{i}", (x + c * dx - s * dy, y + s * dx + c * dy, 0.36), (0.05, 0.05, 0.72), dark, deg)
    box(stage, f"{path}/monitor", (x + s * 0.25, y - c * 0.25 + 0.0, 0.98), (0.60, 0.04, 0.36), dark, deg)


def build(out_dir, k, tile_png):
    rng = np.random.default_rng(100 + k)
    name = f"lab{k}"
    d = Path(out_dir) / name
    (d / "textures").mkdir(parents=True, exist_ok=True)
    Image.open(tile_png).save(d / "textures" / "terrazzo.png")
    stage = Usd.Stage.CreateNew(str(d / f"{name}.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for group in ("floors", "walls", "ceilings", "props"):   # Isaac sets ceilings' visibility: typed prims
        UsdGeom.Xform.Define(stage, f"/World/{group}")

    wall_grey = 1.0 - rng.uniform(0.0, 0.12)
    terrazzo = material(stage, "terrazzo", texture="./textures/terrazzo.png", rough=0.45,
                        scale=float(rng.uniform(0.85, 1.15)))
    grey_floor = material(stage, "grey_floor", color=(0.62, 0.62, 0.60), rough=0.6)
    mat_band = material(stage, "mat_band", color=(0.20, 0.20, 0.21), rough=0.85)
    wall = material(stage, "wall", color=(wall_grey, wall_grey, wall_grey * 0.98), rough=0.9)
    white = material(stage, "white", color=(0.92, 0.92, 0.90), rough=0.5)
    cloth = material(stage, "cloth", color=(0.95, 0.95, 0.95), rough=0.95)
    dark = material(stage, "dark", color=(0.12, 0.12, 0.13), rough=0.6)
    black = material(stage, "black", color=(0.04, 0.04, 0.045), rough=0.7)
    plywood = material(stage, "plywood", color=(0.78, 0.62, 0.42), rough=0.8)
    beige = material(stage, "beige", color=(0.72, 0.64, 0.52), rough=0.95)

    tile = TILE_M * rng.uniform(0.8, 1.2)
    sx, sy = rng.uniform(-0.25, 0.25, 2)                      # where the patch sits around the board
    px, py = (PATCH_X[0] + sx, PATCH_X[1] + sx), (PATCH_Y[0] + sy, PATCH_Y[1] + sy)
    quad(stage, "/World/floors/terrazzo", px, py, 0.004, terrazzo, tile)
    bx, by = (px[0] - BAND, px[1] + BAND), (py[0] - BAND, py[1] + BAND)
    quad(stage, "/World/floors/band", bx, by, 0.003, mat_band)
    quad(stage, "/World/floors/grey", ROOM_X, ROOM_Y, 0.002, grey_floor)

    (x0, x1), (y0, y1) = ROOM_X, ROOM_Y
    for i, (c, s) in enumerate([((x1, 0, ROOM_H / 2), (0.05, y1 - y0, ROOM_H)), ((x0, 0, ROOM_H / 2), (0.05, y1 - y0, ROOM_H)),
                                ((0, y0, ROOM_H / 2), (x1 - x0, 0.05, ROOM_H)), ((0, y1, ROOM_H / 2), (x1 - x0, 0.05, ROOM_H))]):
        box(stage, f"/World/walls/wall{i}", (c[0] + (x0 + x1) / 2 * (i > 1), c[1], c[2]), s, wall)
    box(stage, "/World/ceilings/ceiling", ((x0 + x1) / 2, 0, ROOM_H), (x1 - x0, y1 - y0, 0.04), wall)

    # behind the board (+x): cabinets on the far wall, a black curtain, a leaning OSB board, stacked mats
    cab0 = rng.uniform(-2.6, -1.2)
    for i in range(4):
        box(stage, f"/World/props/cabinet{i}", (x1 - 0.32, cab0 + 0.82 * i, 1.1), (0.6, 0.8, 2.2), white)
        box(stage, f"/World/props/cabinet{i}_gap", (x1 - 0.615, cab0 + 0.82 * i + 0.41, 1.1), (0.01, 0.02, 2.2), dark)
    box(stage, "/World/props/curtain", (x1 - 0.1, cab0 - 0.75, 1.15), (0.05, 1.0, 2.3), black)
    box(stage, "/World/props/osb_leaning", (x1 - 0.25, cab0 + 3.6, 0.9), (0.03, 0.62, 1.2), plywood, 0.0)
    box(stage, "/World/props/mat_stack", (x1 - 0.6, rng.uniform(1.8, 3.0), 0.35), (0.9, 0.9, 0.7), dark)
    box(stage, "/World/props/carpet", (rng.uniform(1.6, 2.4), rng.uniform(-0.3, 0.6), 0.006), (1.5, 1.4, 0.004), beige)
    chair(stage, "/World/props/chair0", rng.uniform(1.8, 2.6), rng.uniform(-0.2, 0.7), rng.uniform(np.pi * 0.7, np.pi * 1.3), black)
    # white cloth over a bench right behind the board, as in the lab photos; a plywood box at its right end
    box(stage, "/World/props/cloth", (0.50, rng.uniform(-0.45, -0.15), 0.275), (0.32, 0.9, 0.55), cloth)   # below the board top
    box(stage, "/World/props/plywood_box", (0.5, -0.82, 0.15), (0.40, 0.50, 0.30), plywood)
    # desks along the sides, chairs at them, a tripod
    for i in range(3):
        side = -1 if i < 2 else 1
        dx = rng.uniform(0.2, 1.8) + (1.7 * i if i < 2 else 0)
        desk(stage, f"/World/props/desk{i}", dx - 0.8, side * (y1 - 0.5), 0.0, white, dark)
        chair(stage, f"/World/props/desk_chair{i}", dx - 0.8, side * (y1 - 1.2), side * np.pi / 2 + rng.uniform(-0.4, 0.4), black)
    tx, ty = rng.uniform(1.0, 2.0), rng.uniform(-2.2, -1.4)
    for i in range(3):
        a = i * 2 * np.pi / 3
        box(stage, f"/World/props/tripod_leg{i}", (tx + 0.18 * np.cos(a), ty + 0.18 * np.sin(a), 0.6), (0.02, 0.02, 1.2), dark)
    box(stage, "/World/props/tripod_pole", (tx, ty, 1.3), (0.03, 0.03, 1.0), dark)
    box(stage, "/World/props/tripod_head", (tx, ty, 1.82), (0.10, 0.06, 0.06), black)

    surface = box(stage, "/World/surface", (0, 0, 0.31), (0.6, 1.2, 0.62), white)   # hidden: Isaac centres the room here
    surface.CreatePurposeAttr(UsdGeom.Tokens.default_)
    stage.GetRootLayer().Save()
    return name


def main(photo, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tile_png = out / "terrazzo_tile.png"
    seamless_tile(photo, tile_png)
    for k in range(VARIANTS):
        print("wrote", build(out, k, tile_png))


if __name__ == "__main__":
    main(*sys.argv[1:3])
