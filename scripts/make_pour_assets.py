# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Pour-task assets: a Coke Zero 0.5 L bottle (900) and an orange plastic cup (901), as revolved meshes,
and their look variants (LOOKS: same shapes, other materials and labels).

    uv run --with usd-core --with trimesh --with pillow python scripts/make_pour_assets.py <data>/assets/pour

Writes the graspnet layout that simple.assets.pour loads: ruled_models/<id>_ruled.usd (+ textures/),
collision_models_mujoco/<id>/<id>_32x8.obj (cuRobo), models/<id>/coacd_0.05/convex_piece_*.obj (MuJoCo),
stable/<id>_stable.npy. Profiles are nominal until measured: (z, r) in metres from the base.
"""

import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

# Coke Zero 0.5 L PET, contour bottle, cap on
BOTTLE = [(0.000, 0.024), (0.004, 0.030), (0.012, 0.0325), (0.040, 0.0325), (0.055, 0.0300),
          (0.070, 0.0325), (0.125, 0.0325), (0.140, 0.0300), (0.160, 0.0215), (0.176, 0.0145),
          (0.186, 0.0135), (0.188, 0.0160), (0.191, 0.0160), (0.192, 0.0148), (0.215, 0.0148)]
BOTTLE_LABEL = (0.072, 0.122)      # black label band
BOTTLE_CAP = 0.192                 # black cap above
BOTTLE_PIECES = [0.0, 0.012, 0.055, 0.125, 0.160, 0.186, 0.215]   # convex slices for MuJoCo

# small orange plastic cup: height, bottom and top radius, wall
CUP_H, CUP_R0, CUP_R1, CUP_WALL = 0.090, 0.027, 0.037, 0.002

SEGMENTS = 48


def revolve(profile, segments=SEGMENTS, close_bottom=True, close_top=True):
    """Vertices, triangles and (u, v) of a surface of revolution about z, centred on its bbox."""
    zs = np.array([p[0] for p in profile])
    rs = np.array([p[1] for p in profile])
    h = zs.max() - zs.min()
    th = np.linspace(0, 2 * np.pi, segments + 1)   # last column repeats the first for the uv seam
    verts, uvs = [], []
    for z, r in zip(zs, rs):
        for t in th:
            verts.append((r * np.cos(t), r * np.sin(t), z))
            uvs.append((t / (2 * np.pi), (z - zs.min()) / h))
    n = segments + 1
    tris = []
    for i in range(len(zs) - 1):
        for j in range(segments):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    for close, row, z, flip in ((close_bottom, 0, zs[0], True), (close_top, len(zs) - 1, zs[-1], False)):
        if close:
            centre = len(verts)
            verts.append((0.0, 0.0, z))
            uvs.append((0.5, (z - zs.min()) / h))
            for j in range(segments):
                a, b = row * n + j, row * n + j + 1
                tris.append((centre, b, a) if flip else (centre, a, b))
    verts = np.array(verts)
    verts[:, 2] -= zs.min() + 0.5 * h
    return verts, np.array(tris), np.array(uvs)


def frustum(z0, r0, z1, r1, zc, segments=24):
    """A closed convex frustum (one MuJoCo convex piece), z shifted by -zc."""
    v, f, _ = revolve([(z0, r0), (z1, r1)], segments)
    v[:, 2] += 0.5 * (z0 + z1) - zc   # revolve centred it on its own bbox
    return v, f


def radius_at(profile, z):
    zs, rs = zip(*profile)
    return float(np.interp(z, zs, rs))


def write_obj(path, verts, tris):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def bottle_texture(path):
    """u around, v up (row 0 = top). Clear PET with water up to half, black label with a red disc, black cap."""
    w, h = 512, 1024
    H = BOTTLE[-1][0]
    img = Image.new("RGB", (w, h), (200, 206, 212))
    d = ImageDraw.Draw(img)
    row = lambda z: int(round((1 - z / H) * (h - 1)))
    d.rectangle([0, row(0.105), w, h], fill=(168, 180, 190))              # water, half full
    d.rectangle([0, row(BOTTLE_LABEL[1]), w, row(BOTTLE_LABEL[0])], fill=(18, 18, 18))
    cy, rr = (row(BOTTLE_LABEL[1]) + row(BOTTLE_LABEL[0])) // 2, (row(BOTTLE_LABEL[0]) - row(BOTTLE_LABEL[1])) // 3
    for cx in (w // 4, 3 * w // 4):                                       # red disc, white script bar
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(200, 16, 30))
        d.rectangle([cx - 2 * rr, cy - rr // 5, cx + 2 * rr, cy + rr // 5], fill=(235, 235, 235))
    d.rectangle([0, 0, w, row(BOTTLE_CAP)], fill=(12, 12, 12))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def cup_texture(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (240, 110, 15)).save(path)


def write_usd(path, verts, tris, uvs, texture_rel):
    """Same prim layout as the graspnet ruled models: /World/{Looks,Meshes/{visual,collision}}, OmniPBR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for xf in (world, UsdGeom.Xform.Define(stage, "/World/Meshes")):
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
        xf.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    UsdGeom.Scope.Define(stage, "/World/Looks")
    mat = UsdShade.Material.Define(stage, "/World/Looks/material_0")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/material_0/material_0")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_rel))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    for name in ("visual", "collision"):
        m = UsdGeom.Mesh.Define(stage, f"/World/Meshes/{name}")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in tris.flatten()]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateExtentAttr([Gf.Vec3f(*map(float, verts.min(0))), Gf.Vec3f(*map(float, verts.max(0)))])
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        if name == "collision":
            UsdGeom.Imageable(m).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().Save()


def cup_visual():
    """Hollow cup: outer wall, rim, inner wall, inner floor."""
    o = [(0.0, CUP_R0), (CUP_H, CUP_R1)]
    i = [(CUP_WALL * 2, CUP_R0 - CUP_WALL), (CUP_H, CUP_R1 - CUP_WALL)]
    vo, fo, uo = revolve(o, close_bottom=True, close_top=False)
    vi, fi, ui = revolve(i, close_bottom=True, close_top=False)
    vi[:, 2] += (vo[:, 2].max() - vi[:, 2].max())                 # both centred on their own bbox: align tops
    fi = fi[:, ::-1] + len(vo)                                    # inner faces point inwards
    n = SEGMENTS + 1
    rim = []
    top_o, top_i = 1 * n, 1 * n + len(vo)                         # row 1 is the top ring of both
    for j in range(SEGMENTS):
        a, b, c, d = top_o + j, top_o + j + 1, top_i + j, top_i + j + 1
        rim += [(a, d, b), (a, c, d)]
    return np.vstack([vo, vi]), np.vstack([fo, fi, np.array(rim)]), np.vstack([uo, ui])


# look variants: (uid, shape, body, label, cap). body: ("glass", rgb) clear or tinted PET / cup, or ("pbr", rgb) a
# full bottle's liquid or an opaque cup; label: a brand drawing or None; cap: rgb. Physics and grasps stay 900/901's.
COLA, ORANGE, LEMON = (40, 18, 10), (245, 120, 10), (200, 230, 150)
CLEAR, GREEN_PET = (235, 240, 240), (150, 210, 160)
LOOKS = [
    (902, 900, ("glass", CLEAR), "zero", (12, 12, 12)),
    (920, 900, ("pbr", COLA), "zero", (12, 12, 12)),
    (910, 900, ("pbr", COLA), "cola", (200, 16, 30)),
    (911, 900, ("glass", CLEAR), "cola", (200, 16, 30)),
    (912, 900, ("pbr", ORANGE), "orange", (20, 60, 160)),
    (913, 900, ("glass", CLEAR), "orange", (20, 60, 160)),
    (914, 900, ("glass", GREEN_PET), "lemon", (30, 140, 50)),
    (915, 900, ("glass", CLEAR), "water", (40, 110, 220)),
    (916, 900, ("glass", CLEAR), "stripes0", (230, 230, 230)),
    (917, 900, ("pbr", (90, 30, 120)), "stripes1", (240, 200, 20)),
    (918, 900, ("glass", GREEN_PET), "stripes2", (200, 40, 40)),
    (919, 900, ("pbr", (230, 200, 60)), "stripes3", (20, 20, 20)),
    (931, 901, ("pbr", (200, 25, 25)), None, None),
    (932, 901, ("pbr", (30, 80, 200)), None, None),
    (933, 901, ("pbr", (235, 235, 230)), None, None),
    (934, 901, ("pbr", (40, 160, 60)), None, None),
    (935, 901, ("glass", (230, 235, 240)), None, None),
]


def label_texture(path, brand):
    """The label band only (u around, v over the band, row 0 = top)."""
    w, h = 512, 256
    rng = np.random.default_rng(zlib.crc32(brand.encode()))
    base = {"zero": (18, 18, 18), "cola": (200, 16, 30), "orange": (20, 60, 160), "lemon": (30, 140, 50),
            "water": (200, 225, 245)}.get(brand, tuple(int(c) for c in rng.integers(30, 230, 3)))
    img = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(img)
    if brand in ("zero", "cola"):
        for cx in (w // 4, 3 * w // 4):
            if brand == "zero":
                d.ellipse([cx - 45, h // 2 - 45, cx + 45, h // 2 + 45], fill=(200, 16, 30))
            d.rectangle([cx - 90, h // 2 - 9, cx + 90, h // 2 + 9], fill=(235, 235, 235))
    elif brand == "orange":
        for cx in (w // 4, 3 * w // 4):
            d.ellipse([cx - 60, h // 2 - 60, cx + 60, h // 2 + 60], fill=(245, 120, 10))
    elif brand == "lemon":
        d.rectangle([0, h // 2 - 20, w, h // 2 + 20], fill=(230, 230, 60))
    elif brand == "water":
        d.rectangle([0, h // 3, w, 2 * h // 3], fill=(40, 110, 220))
    else:                                         # generic: random stripes
        for _ in range(int(rng.integers(2, 6))):
            y0 = int(rng.integers(0, h - 20))
            d.rectangle([0, y0, w, y0 + int(rng.integers(8, 60))], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _shader(stage, name, kind, rgb=None, texture=None, rough=0.3):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/{name}")
    col = Gf.Vec3f(*(c / 255 for c in rgb)) if rgb is not None else None
    if kind == "glass":
        sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(col)
        sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.45)
        sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.05)
    else:
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        if texture is not None:
            sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        else:
            sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(col)
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def write_variant_usd(path, verts, tris, uvs, parts):
    """write_usd's prim layout; `parts` [(name, face indices, material kwargs)] become GeomSubsets of the visual mesh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for xf in (world, UsdGeom.Xform.Define(stage, "/World/Meshes")):
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
        xf.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    UsdGeom.Scope.Define(stage, "/World/Looks")
    mats = [(name, faces, _shader(stage, f"material_{k}", **kw)) for k, (name, faces, kw) in enumerate(parts)]
    for name in ("visual", "collision"):
        m = UsdGeom.Mesh.Define(stage, f"/World/Meshes/{name}")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in tris.flatten()]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateExtentAttr([Gf.Vec3f(*map(float, verts.min(0))), Gf.Vec3f(*map(float, verts.max(0)))])
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))
        if name == "collision":
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats[0][2])
            UsdGeom.Imageable(m).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            continue
        for part, faces, mat in mats:
            sub = UsdGeom.Subset.CreateGeomSubset(m, part, UsdGeom.Tokens.face, Vt.IntArray([int(i) for i in faces]),
                                                  familyName="materialBind")
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()


def write_looks(out):
    v, f, uv = revolve(BOTTLE)
    H = BOTTLE[-1][0]
    zc = v[f].mean(1)[:, 2] + 0.5 * H                        # face centroid height from the base
    cap = np.where(zc >= BOTTLE_CAP)[0]
    band = np.where((zc >= BOTTLE_LABEL[0]) & (zc <= BOTTLE_LABEL[1]))[0]
    body = np.setdiff1d(np.arange(len(f)), np.concatenate([cap, band]))
    cv, cf, cuv = cup_visual()
    for uid, shape, (kind, rgb), brand, cap_rgb in LOOKS:
        usd = out / "ruled_models" / f"{uid}_ruled.usd"
        if shape == 901:
            write_variant_usd(usd, cv, cf, cuv, [("body", np.arange(len(cf)), dict(kind=kind, rgb=rgb, rough=0.4))])
            continue
        # the band's own uv: v over the label only
        luv = uv.copy()
        luv[:, 1] = (luv[:, 1] * H - BOTTLE_LABEL[0]) / (BOTTLE_LABEL[1] - BOTTLE_LABEL[0])
        tex = f"./textures/{uid}_label.png"
        label_texture(out / "ruled_models" / "textures" / f"{uid}_label.png", brand)
        write_variant_usd(usd, v, f, luv, [("body", body, dict(kind=kind, rgb=rgb, rough=0.15)),
                                          ("label", band, dict(kind="pbr", texture=tex, rough=0.35)),
                                          ("cap", cap, dict(kind="pbr", rgb=cap_rgb, rough=0.4))])
    print(f"{len(LOOKS)} look variants")


def main(out):
    out = Path(out)
    # bottle
    v, f, uv = revolve(BOTTLE)
    H = BOTTLE[-1][0]
    bottle_texture(out / "ruled_models" / "textures" / "900_texture.png")
    write_usd(out / "ruled_models" / "900_ruled.usd", v, f, uv, "./textures/900_texture.png")
    v32, f32, _ = revolve(BOTTLE, segments=32)
    write_obj(out / "collision_models_mujoco" / "900" / "900_32x8.obj", v32, f32)
    for k, (z0, z1) in enumerate(zip(BOTTLE_PIECES[:-1], BOTTLE_PIECES[1:])):
        pv, pf = frustum(z0, radius_at(BOTTLE, z0), z1, radius_at(BOTTLE, z1), 0.5 * H)
        write_obj(out / "models" / "900" / "coacd_0.05" / f"convex_piece_{k}.obj", pv, pf)
    (out / "stable").mkdir(parents=True, exist_ok=True)
    np.save(out / "stable" / "900_stable.npy", np.array([[0.5, 0.0, 0.5 * H, 1.0, 0.0, 0.0, 0.0]]))
    # cup: hollow to the camera, solid to the physics (the bottle never enters it)
    v, f, uv = cup_visual()
    cup_texture(out / "ruled_models" / "textures" / "901_texture.png")
    write_usd(out / "ruled_models" / "901_ruled.usd", v, f, uv, "./textures/901_texture.png")
    sv, sf = frustum(0.0, CUP_R0, CUP_H, CUP_R1, 0.5 * CUP_H, segments=32)
    write_obj(out / "collision_models_mujoco" / "901" / "901_32x8.obj", sv, sf)
    write_obj(out / "models" / "901" / "coacd_0.05" / "convex_piece_0.obj", sv, sf)
    np.save(out / "stable" / "901_stable.npy", np.array([[0.5, 0.0, 0.5 * CUP_H, 1.0, 0.0, 0.0, 0.0]]))
    write_looks(out)
    print(f"pour assets -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
