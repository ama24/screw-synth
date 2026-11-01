"""Everything that needs bpy / BlenderProc. Only imported inside `blenderproc run`.

World units: blender_units = mm * 0.001 * world_scale (Bullet is unstable at millimetre scale).
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import blenderproc as bproc  # noqa: F401  (must be importable: we run inside blenderproc)
import bpy
import numpy as np
from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects

from . import backgrounds as bg
from . import dirt as dirt_mod
from .camera import RenderCamera

log = logging.getLogger(__name__)
SLOTS = ("body", "insert")


def _input(node: bpy.types.Node, names: Sequence[str]):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def _mix_rgba(nt: bpy.types.NodeTree):
    """Mix node in colour mode: (factor, A, B, result) sockets picked by type, since Blender 4 has
    several sockets called A/B/Result (float, vector, colour) on the same node."""
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
    b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
    fac = next(s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE")
    out = next(s for s in mix.outputs if s.name == "Result" and s.type == "RGBA")
    return mix, fac, a, b, out


def _set(node: bpy.types.Node, names: Sequence[str], value: Any) -> None:
    sock = _input(node, names)
    if sock is not None:
        sock.default_value = value


# ------------------------------------------------------------------------------------ renderer


def init_renderer(cfg: Mapping[str, Any]) -> None:
    bproc.renderer.set_render_devices(desired_gpu_device_type=[cfg.get("device", "OPTIX"), "CUDA"])
    bproc.renderer.set_max_amount_of_samples(int(cfg.get("samples", 64)))
    bproc.renderer.set_noise_threshold(float(cfg.get("noise_threshold", 0.02)))
    den = cfg.get("denoiser", "OPTIX")
    bproc.renderer.set_denoiser(None if den in (None, "none", "NONE") else den)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=int(cfg.get("diffuse_bounces", 3)),
        glossy_bounces=int(cfg.get("glossy_bounces", 4)),
        max_bounces=int(cfg.get("max_bounces", 6)),
        transmission_bounces=int(cfg.get("transmission_bounces", 2)),
        transparent_max_bounces=4,
    )
    scene = bpy.context.scene
    # must stay off: with persistent data Cycles keeps the shaders of the colour pass and
    # BlenderProc's segmentation pass renders an empty (all-background) instance map
    scene.render.use_persistent_data = False
    scene.view_settings.view_transform = (
        "Standard"  # no filmic tone curve: linear -> sRGB like a camera
    )
    scene.view_settings.look = "None"
    scene.render.film_transparent = False
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id"], default_values={"category_id": 0}
    )
    if cfg.get("depth", False):
        bproc.renderer.enable_depth_output(activate_antialiasing=False)


def set_camera(cam: RenderCamera, world_scale: float) -> None:
    w, h = cam.resolution
    bproc.camera.set_intrinsics_from_K_matrix(cam.K, w, h, clip_start=0.01, clip_end=100.0)
    bproc.camera.add_camera_pose(cam.cam2world_blender(world_scale), frame=0)


def check_camera(cam: RenderCamera, world_scale: float, tol_px: float = 0.1) -> float:
    """Project plate points with Blender's camera and with K/T in numpy; they must agree."""
    x0, y0, x1, y1 = cam.visible_plate_rect(-5.0)
    pts = np.array(
        [
            [x, y, z]
            for x in np.linspace(x0, x1, 5)
            for y in np.linspace(y0, y1, 5)
            for z in (0.0, 8.0)
        ]
    )
    px_np = cam.project(pts)
    px_bl = bproc.camera.project_points(pts * 0.001 * world_scale, frame=0)
    err = float(np.abs(px_np - px_bl).max())
    if err > tol_px:
        raise RuntimeError(f"Blender camera disagrees with K/T by {err:.3f} px (> {tol_px})")
    log.info("camera check: Blender vs numpy projection max error %.4f px", err)
    return err


# --------------------------------------------------------------------------------- materials


def make_material(name: str, p: Mapping[str, Any], bu_per_mm: float = 0.01) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    _set(bsdf, ["Base Color"], (*p["base_color"], 1.0))
    _set(bsdf, ["Metallic"], p["metallic"])
    _set(bsdf, ["Roughness"], p["roughness"])
    _set(bsdf, ["Anisotropic"], p.get("anisotropic", 0.0))
    _set(bsdf, ["Anisotropic Rotation"], p.get("anisotropic_rotation", 0.0))
    _set(bsdf, ["Coat Weight", "Clearcoat"], p.get("coat", 0.0))
    _set(bsdf, ["Subsurface Weight", "Subsurface"], p.get("subsurface", 0.0))
    if p.get("iridescence", 0.0) > 0:
        _set(bsdf, ["Thin Film Thickness"], 300.0 * p["iridescence"])
    coord = nt.nodes.new("ShaderNodeTexCoord")
    seed_offset = (p.get("noise_seed", 0) % 1000) * 0.137
    rough_in = _input(bsdf, ["Roughness"])
    if p.get("scratches") or p.get("fingerprints"):
        noise = nt.nodes.new("ShaderNodeTexNoise")
        _set(noise, ["Scale"], 400.0 if p.get("scratches") else 60.0)
        _set(noise, ["Detail"], 8.0)
        _set(noise, ["W"], seed_offset)
        noise.noise_dimensions = "4D"
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeMapRange")
        _set(ramp, ["From Min"], 0.45)
        _set(ramp, ["From Max"], 0.75)
        _set(ramp, ["To Min"], p["roughness"])
        _set(ramp, ["To Max"], min(1.0, p["roughness"] + (0.35 if p.get("fingerprints") else 0.2)))
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs["Result"], rough_in)
    if p.get("grime"):
        gnoise = nt.nodes.new("ShaderNodeTexNoise")
        _set(gnoise, ["Scale"], 25.0)
        _set(gnoise, ["W"], seed_offset + 3.1)
        gnoise.noise_dimensions = "4D"
        nt.links.new(coord.outputs["Object"], gnoise.inputs["Vector"])
        mix, m_fac, m_a, m_b, m_out = _mix_rgba(nt)
        mix.blend_type = "MULTIPLY"
        m_a.default_value = (*p["base_color"], 1.0)
        m_b.default_value = (0.55, 0.5, 0.42, 1.0)
        fac = nt.nodes.new("ShaderNodeMapRange")
        _set(fac, ["From Min"], 0.5)
        _set(fac, ["From Max"], 0.8)
        nt.links.new(gnoise.outputs["Fac"], fac.inputs["Value"])
        nt.links.new(fac.outputs["Result"], m_fac)
        nt.links.new(m_out, _input(bsdf, ["Base Color"]))
    d = p.get("dirt") or {}
    if d.get("dirty"):
        w = (d.get("seed", 0) % 1000) * 0.173
        vec = coord.outputs["Object"]
        if d.get("oil", 0) > 0:  # smears of a few mm on parts a few mm thick
            m = _oil_mask(nt, vec, d["oil"], 2.5 * bu_per_mm, w)
            _overlay(nt, bsdf, ["Base Color"], m, (*OIL_RGB, 1.0))
            _overlay(nt, bsdf, ["Roughness"], m, OIL_ROUGHNESS)
            _overlay(nt, bsdf, ["Metallic"], m, 0.2)
        if d.get("dust", 0) > 0:
            m = _dust_mask(nt, vec, d["dust"], d["dust_size_mm"], bu_per_mm, w + 5.3)
            _overlay(nt, bsdf, ["Base Color"], m, (*DUST_RGB, 1.0))
            _overlay(nt, bsdf, ["Roughness"], m, 1.0)
            _overlay(nt, bsdf, ["Metallic"], m, 0.0)
    return mat


# ------------------------------------------------------------------------------------- dirt
# Dark oil (glossy, nearly black-brown) and dust (matt, light grey-beige specks), layered over a
# material's own colour, roughness and metallic. Amounts come from scenegen.dirt (dirt.py).

OIL_RGB = (0.012, 0.009, 0.006)
OIL_ROUGHNESS = 0.4  # a satin film: glossier than the plate, but no mirror of the light panel
DUST_RGB = (0.55, 0.52, 0.47)


def _mix_float(nt: bpy.types.NodeTree):
    """Mix node in float mode: (factor, A, B, result) sockets."""
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    a = next(s for s in mix.inputs if s.name == "A" and s.type == "VALUE")
    b = next(s for s in mix.inputs if s.name == "B" and s.type == "VALUE")
    fac = next(s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE")
    out = next(s for s in mix.outputs if s.name == "Result" and s.type == "VALUE")
    return mix, fac, a, b, out


def _overlay(nt: bpy.types.NodeTree, node: bpy.types.Node, names: Sequence[str], mask, value):
    """Blend an input of `node` towards `value` where `mask` is 1, keeping whatever feeds it now."""
    sock = _input(node, names)
    if sock is None:
        return
    colour = sock.type == "RGBA"
    mix, fac, a, b, out = _mix_rgba(nt) if colour else _mix_float(nt)
    if sock.is_linked:
        nt.links.new(sock.links[0].from_socket, a)
    else:
        a.default_value = tuple(sock.default_value) if colour else sock.default_value
    b.default_value = value
    nt.links.new(mask, fac)
    nt.links.new(out, sock)
    return mix


def _oil_mask(nt: bpy.types.NodeTree, vector, coverage: float, feature_bu: float, w: float):
    """Blotchy 0..1 mask covering about `coverage` of the surface, blobs of about feature_bu."""
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    _set(noise, ["Scale"], 1.0 / max(feature_bu, 1e-6))
    _set(noise, ["Detail"], 6.0)
    _set(noise, ["Roughness"], 0.6)
    _set(noise, ["W"], w)
    nt.links.new(vector, noise.inputs["Vector"])
    t = dirt_mod.oil_threshold(coverage)
    ramp = nt.nodes.new("ShaderNodeMapRange")
    _set(ramp, ["From Min"], t)
    _set(ramp, ["From Max"], t + 0.12)  # soft edges: a film thinning out, not a painted patch
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    return ramp.outputs["Result"]


def _dust_cell_mm(size_mm: float) -> float:
    """Spacing of the dust grid: a speck takes at most a quarter of its cell."""
    return max(4.0 * size_mm, 0.3)


def _dust_mask(
    nt: bpy.types.NodeTree, vector, amount: float, size_mm: float, bu_per_mm: float, w: float
):
    """Round specks of size_mm; `amount` = share of the grid cells that hold one (0..1)."""
    cell = _dust_cell_mm(size_mm)
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.voronoi_dimensions = "4D"
    _set(vor, ["Scale"], 1.0 / (cell * bu_per_mm))
    _set(vor, ["W"], w)
    nt.links.new(vector, vor.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(vor.outputs["Color"], sep.inputs["Color"])
    keep = nt.nodes.new("ShaderNodeMath")
    keep.operation = "LESS_THAN"  # 1 in the cells whose random value is below the amount
    nt.links.new(sep.outputs[0], keep.inputs[0])
    keep.inputs[1].default_value = amount
    r = size_mm / 2 / cell  # speck radius in cell units (Voronoi distance is in cell units)
    spot = nt.nodes.new("ShaderNodeMapRange")
    _set(spot, ["From Min"], r)
    _set(spot, ["From Max"], 0.5 * r)  # 1 in the centre, soft edge
    nt.links.new(vor.outputs["Distance"], spot.inputs["Value"])
    both = nt.nodes.new("ShaderNodeMath")
    both.operation = "MULTIPLY"
    nt.links.new(keep.outputs[0], both.inputs[0])
    nt.links.new(spot.outputs["Result"], both.inputs[1])
    return both.outputs[0]


# ------------------------------------------------------------------------------------- assets


class AssetLibrary:
    """Loads each part once (hidden prototype) and hands out duplicates."""

    def __init__(self, assets_dir: Path, world_scale: float) -> None:
        self.dir = Path(assets_dir)
        self.s = 0.001 * world_scale
        self.protos: dict[str, tuple[Any, list[Any], dict]] = {}

    def _load(self, part_id: str, meta: dict) -> tuple[Any, list[Any], dict]:
        folder = self.dir / part_id
        obj = bproc.loader.load_obj(str(folder / "render.ply"), forward_axis="Y", up_axis="Z")[0]
        obj.set_scale([self.s] * 3)
        obj.persist_transformation_into_mesh()
        obj.set_shading_mode("AUTO", 35)
        mesh = obj.get_mesh()
        slots = meta["mesh"].get("material_slots", {"body": [0, len(mesh.polygons)]})
        # the PLY loader adds a default material; drop it (and the datablock) so slots follow SLOTS
        loader_mats = [m.name for m in mesh.materials if m is not None]
        mesh.materials.clear()
        for name in loader_mats:
            m = bpy.data.materials.get(name)
            if m is not None and m.users == 0:
                bpy.data.materials.remove(m)
        for _ in SLOTS[: max(1, len(slots))]:
            mesh.materials.append(None)
        if "insert" in slots:
            a, b = slots["insert"]
            idx = np.zeros(len(mesh.polygons), dtype=np.int32)
            idx[a:b] = 1
            mesh.polygons.foreach_set("material_index", idx)
        pieces = bproc.loader.load_obj(str(folder / "collision.obj"), forward_axis="Y", up_axis="Z")
        for p in pieces:
            p.set_scale([self.s] * 3)
            p.persist_transformation_into_mesh()
            p.set_parent(obj)
            p.hide(True)
        obj.hide(True)
        obj.set_location([0, 0, -1000])  # prototypes stay far below the scene
        return obj, pieces, meta

    def instance(self, part_id: str, meta: dict) -> tuple[Any, list[Any]]:
        if part_id not in self.protos:
            self.protos[part_id] = self._load(part_id, meta)
        proto = self.protos[part_id][0]
        # not linked: BlenderProc's physics moves each object's origin to its centre of mass, which
        # would corrupt mesh data shared between instances
        dup = proto.duplicate(duplicate_children=True, linked=False)
        dup.hide(False)
        children = dup.get_children()
        for c in children:
            c.hide(True)
        return dup, children


# ------------------------------------------------------------------------------- static scene


class StaticScene:
    """Plate (with the real empty-plate texture), bed, collider, dark shroud and the light panel."""

    def __init__(
        self,
        cam: RenderCamera,
        cfg: Mapping[str, Any],
        calib_dir: Path,
        world_scale: float,
        plate_dir: Path | None = None,
        envelope: Mapping[str, Any] | None = None,
        open_room: bool = False,
    ) -> None:
        """envelope: what the varied camera views can see (variation.view_envelope), or None for
        the rig camera only; the plate and the shroud then grow to cover every view. open_room:
        no shroud (an HDRI environment lights the scene instead)."""
        self.cfg, self.cam, self.ws = cfg, cam, world_scale
        self.s = 0.001 * world_scale
        pc = cfg.get("plate", {}) or {}
        x0, y0, x1, y1 = cam.visible_plate_rect(30.0)
        self.centre = ((x0 + x1) / 2, (y0 + y1) / 2)  # the rig view's centre: lights stay over it
        if envelope is not None:
            ex0, ey0, ex1, ey1 = envelope["rect"]
            x0, y0 = min(x0, ex0 - 30), min(y0, ey0 - 30)
            x1, y1 = max(x1, ex1 + 30), max(y1, ey1 + 30)
        self.rect = (x0, y0, x1, y1)
        self.plate = self._plate_mesh(x0, y0, x1, y1, int(pc.get("grid", 24)))
        self.texture = None
        tex_path = Path(calib_dir) / "plate_empty.png"
        if cam.source == "calibration" and tex_path.is_file():
            self.texture = bpy.data.images.load(str(tex_path))
        # surface photos (craft mat, table...) from the plate dir, laid at true scale per scene
        self.plate_dir = Path(plate_dir) if plate_dir is not None else None
        self.images = (
            bg.usable_images(self.plate_dir, bg.load_library(self.plate_dir))
            if self.plate_dir
            else []
        )
        self._bg_cache: dict[str, str] = {}  # file -> name of the loaded bpy image
        plate_mat = self._plate_material()
        self.plate_mat_name = plate_mat.name
        self.set_plate_dirt(dirt_mod.CLEAN)
        self.plate.blender_obj.data.materials.append(plate_mat)
        self.plate.set_cp("category_id", 0)
        th = float(pc.get("thickness_mm", 1.75))
        self.collider = bproc.object.create_primitive("CUBE")
        self.collider.set_scale([(x1 - x0) / 2 * self.s, (y1 - y0) / 2 * self.s, 5 * self.s])
        self.collider.set_location([(x0 + x1) / 2 * self.s, (y0 + y1) / 2 * self.s, -5 * self.s])
        self.collider.enable_rigidbody(
            False,
            collision_shape="BOX",
            collision_margin=float(cfg.get("physics", {}).get("collision_margin_mm", 0.05))
            * self.s,
            friction=float(cfg.get("physics", {}).get("friction", 0.6)),
        )
        self.collider.hide(True)
        self.collider.set_cp("category_id", 0)
        bed = bproc.object.create_primitive("CUBE")
        bed.set_scale([(x1 - x0) / 2 * self.s * 1.3, (y1 - y0) / 2 * self.s * 1.3, 1.5 * self.s])
        bed.set_location(
            [(x0 + x1) / 2 * self.s, (y0 + y1) / 2 * self.s, (-th - 1.5 - 0.01) * self.s]
        )
        bed_mat = bproc.material.create("bed")
        bed_mat.set_principled_shader_value("Base Color", [0.25, 0.25, 0.26, 1])
        bed_mat.set_principled_shader_value("Metallic", 0.8)
        bed_mat.set_principled_shader_value("Roughness", 0.5)
        bed.replace_materials(bed_mat)
        bed.set_cp("category_id", 0)
        sc = cfg.get("shroud", {}) or {}
        hw = float(sc.get("half_width_mm", 220.0))
        hh = float(sc.get("height_mm", 420.0))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if envelope is not None:  # every camera position and the whole plate inside the shroud
            reach = max(abs(x0 - cx), abs(x1 - cx), abs(y0 - cy), abs(y1 - cy))
            hw = max(hw, reach + 10.0, float(envelope["max_offset_mm"]) + 60.0)
            hh = max(hh, float(envelope["max_height_mm"]) + 80.0)
        self.shroud = bproc.object.create_primitive("CUBE")
        self.shroud.set_scale([hw * self.s, hw * self.s, hh / 2 * self.s])
        self.shroud.set_location([cx * self.s, cy * self.s, (hh / 2 - 20) * self.s])
        smat = bproc.material.create("shroud")
        smat.set_principled_shader_value("Base Color", [*sc.get("colour", [0.02, 0.02, 0.02]), 1])
        smat.set_principled_shader_value("Roughness", 0.9)
        self.shroud.replace_materials(smat)
        self.shroud.set_cp("category_id", 0)
        if open_room:
            self.shroud.hide(True)
        self.light = bproc.types.Light("AREA", name="panel")
        self.light.blender_obj.data.shape = "RECTANGLE"
        self.world_strength = 0.0
        bproc.renderer.set_world_background([1.0, 1.0, 1.0], 0.0)

    def add_lip(self, bounds: Sequence[float], height_mm: float, thickness_mm: float) -> None:
        """Invisible walls just outside the frame: the lip around the plate that stops parts."""
        x0, y0, x1, y1 = bounds
        t, h = thickness_mm, height_mm
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        walls = [
            ((x0 - t / 2, cy), (t / 2, (y1 - y0) / 2 + t)),
            ((x1 + t / 2, cy), (t / 2, (y1 - y0) / 2 + t)),
            ((cx, y0 - t / 2), ((x1 - x0) / 2 + t, t / 2)),
            ((cx, y1 + t / 2), ((x1 - x0) / 2 + t, t / 2)),
        ]
        self.lip = []
        for (wx, wy), (hx, hy) in walls:
            wall = bproc.object.create_primitive("CUBE")
            wall.set_scale([hx * self.s, hy * self.s, h / 2 * self.s])
            wall.set_location([wx * self.s, wy * self.s, h / 2 * self.s])
            wall.enable_rigidbody(False, collision_shape="BOX", friction=0.6)
            wall.hide(True)
            wall.set_cp("category_id", 0)
            self.lip.append(wall)

    def _plate_mesh(self, x0: float, y0: float, x1: float, y1: float, n: int):
        """Grid plane at z = 0 whose UVs map plate points to the undistorted ROI (plate_empty.png)."""
        xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
        X, Y = np.meshgrid(xs, ys)
        verts = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
        faces = []
        for j in range(n):
            for i in range(n):
                a = j * (n + 1) + i
                faces.append((a, a + 1, a + n + 2, a + n + 1))
        mesh = bpy.data.meshes.new("plate")
        mesh.from_pydata((verts * self.s).tolist(), [], faces)
        uv = mesh.uv_layers.new(name="UVMap")
        px = self.cam.project(verts)
        w, h = self.cam.resolution
        uvs = np.c_[(px[:, 0] + 0.5) / w, 1.0 - (px[:, 1] + 0.5) / h]
        loop_uv = np.array([uvs[v] for poly in mesh.polygons for v in poly.vertices])
        uv.data.foreach_set("uv", loop_uv.ravel())
        mesh.update()
        obj = bpy.data.objects.new("plate", mesh)
        bpy.context.collection.objects.link(obj)
        return bproc.types.MeshObject(obj)

    def _plate_material(self) -> bpy.types.Material:
        mat = bpy.data.materials.new("plate")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        bsdf.name = "plate_bsdf"
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.name = "plate_rgb"
        coord = nt.nodes.new("ShaderNodeTexCoord")
        # surface photo in plate millimetres: rotate first, then scale to the photo's real size
        # (one Mapping node scales before rotating, which would skew non-square photos)
        rot = nt.nodes.new("ShaderNodeMapping")
        rot.name = "plate_bg_rot"
        scale = nt.nodes.new("ShaderNodeMapping")
        scale.name = "plate_bg_scale"
        nt.links.new(coord.outputs["Object"], rot.inputs["Vector"])
        nt.links.new(rot.outputs["Vector"], scale.inputs["Vector"])
        photo = nt.nodes.new("ShaderNodeTexImage")
        photo.name = "plate_bg"
        photo.interpolation = "Cubic"
        photo.extension = "MIRROR"  # tiles without seams when the photo is smaller than the plate
        nt.links.new(scale.outputs["Vector"], photo.inputs["Vector"])
        # calibrated empty-plate photo (camera-projected UVs), or the solid colour without one
        if self.texture is not None:
            img = nt.nodes.new("ShaderNodeTexImage")
            img.image = self.texture
            img.interpolation = "Cubic"
            first = img.outputs["Color"]
        else:
            first = rgb.outputs["Color"]
        mix_img, _, a1, b1, out1 = _mix_rgba(nt)
        mix_img.name = "plate_mix_img"  # 0: calibrated photo, 1: surface photo
        nt.links.new(first, a1)
        nt.links.new(photo.outputs["Color"], b1)
        mix, _, a2, b2, out2 = _mix_rgba(nt)
        mix.name = "plate_mix"  # 1: solid colour
        nt.links.new(out1, a2)
        nt.links.new(rgb.outputs["Color"], b2)
        nt.links.new(out2, _input(bsdf, ["Base Color"]))
        rough = nt.nodes.new("ShaderNodeValue")
        rough.name = "plate_rough"  # set per render by jitter()
        rough.outputs[0].default_value = 0.75
        nt.links.new(rough.outputs[0], _input(bsdf, ["Roughness"]))
        # dirt layers (scenegen.dirt.plate), set per layout by set_plate_dirt(): oil stains of a
        # few cm, then dust specks on top
        vec = coord.outputs["Object"]
        oil = _oil_mask(nt, vec, 0.0, 15.0 * self.s, 0.0)
        oil.node.name = "plate_oil_mask"
        oil.node.inputs["Value"].links[0].from_node.name = "plate_oil_noise"
        _overlay(nt, bsdf, ["Base Color"], oil, (*OIL_RGB, 1.0))
        _overlay(nt, bsdf, ["Roughness"], oil, OIL_ROUGHNESS)
        dust = _dust_mask(nt, vec, 0.0, 0.2, self.s, 0.0)
        keep = dust.node.inputs[0].links[0].from_node
        spot = dust.node.inputs[1].links[0].from_node
        keep.name, spot.name = "plate_dust_keep", "plate_dust_spot"
        spot.inputs["Value"].links[0].from_node.name = "plate_dust_vor"
        _overlay(nt, bsdf, ["Base Color"], dust, (*DUST_RGB, 1.0))
        _overlay(nt, bsdf, ["Roughness"], dust, 1.0)
        # print lines: fine waves along x as a bump
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X"
        pitch = float((self.cfg.get("plate", {}) or {}).get("line_pitch_mm", 0.4)) * self.s
        _set(wave, ["Scale"], 1.0 / max(pitch, 1e-6) / (2 * math.pi))
        nt.links.new(coord.outputs["Object"], wave.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.name = "plate_bump"
        _set(bump, ["Distance"], 0.02 * self.s)
        nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], _input(bsdf, ["Normal"]))
        return mat

    def _show_background(self, nodes: Any, background: Mapping[str, Any]) -> None:
        """Point the plate material at one background (see scenegen.backgrounds)."""
        kind = background["kind"]
        _factor(nodes["plate_mix_img"]).default_value = 1.0 if kind == "images" else 0.0
        _factor(nodes["plate_mix"]).default_value = 1.0 if kind == "colours" else 0.0
        if kind == "colours":
            nodes["plate_rgb"].outputs["Color"].default_value = (*background["rgb"], 1.0)
        if kind != "images":
            return
        name = self._bg_cache.get(background["file"])
        if name is None or name not in bpy.data.images:
            img = bpy.data.images.load(
                str(self.plate_dir / background["file"]), check_existing=True
            )
            name = self._bg_cache[background["file"]] = img.name
        img = bpy.data.images[name]
        nodes["plate_bg"].image = img
        w_px, h_px = img.size
        m = bg.photo_mapping(
            self.view.visible_plate_rect(0.0),
            background["rotation_deg"],
            background["width_mm"],
            h_px / max(1, w_px),
            background["offset"],
        )
        background["centre_uv"] = [round(v, 4) for v in m["centre_uv"]]  # recorded in meta.json
        background["fits"] = m["fits"]
        # turn around the view centre: out = R (p - c) = R p - R c
        a = math.radians(background["rotation_deg"])
        cx, cy = (v * self.s for v in m["centre_mm"])
        rot = nodes["plate_bg_rot"]
        rot.inputs["Rotation"].default_value = (0.0, 0.0, a)
        rot.inputs["Location"].default_value = (
            -(math.cos(a) * cx - math.sin(a) * cy),
            -(math.sin(a) * cx + math.cos(a) * cy),
            0.0,
        )
        # then to photo units (0..1 across the photo), centred on the chosen spot
        scale = nodes["plate_bg_scale"]
        w_bu, h_bu = (v * self.s for v in m["size_mm"])
        scale.inputs["Scale"].default_value = (1.0 / w_bu, 1.0 / h_bu, 1.0)
        scale.inputs["Location"].default_value = (*m["centre_uv"], 0.0)

    def set_plate_dirt(self, d: Mapping[str, Any]) -> None:
        """Oil stains and dust on the plate for one layout (dirt.sample(..., "plate", ...))."""
        nodes = bpy.data.materials[self.plate_mat_name].node_tree.nodes
        w = (d.get("seed", 0) % 1000) * 0.173
        oil = float(d.get("oil", 0.0)) if d.get("dirty") else 0.0
        t = dirt_mod.oil_threshold(oil) if oil > 0 else 2.0  # 2: above every noise value = clean
        _set(nodes["plate_oil_mask"], ["From Min"], t)
        _set(nodes["plate_oil_mask"], ["From Max"], t + 0.12)
        _set(nodes["plate_oil_noise"], ["W"], w)
        dust = float(d.get("dust", 0.0)) if d.get("dirty") else 0.0
        size = float(d.get("dust_size_mm", 0.0)) or 0.2
        cell = _dust_cell_mm(size)
        nodes["plate_dust_keep"].inputs[1].default_value = dust
        _set(nodes["plate_dust_vor"], ["Scale"], 1.0 / (cell * self.s))
        _set(nodes["plate_dust_vor"], ["W"], w + 5.3)
        r = size / 2 / cell
        _set(nodes["plate_dust_spot"], ["From Min"], r)
        _set(nodes["plate_dust_spot"], ["From Max"], 0.5 * r)

    def jitter(
        self, rng: random.Random, view: RenderCamera | None = None, rig_view: bool = True
    ) -> dict[str, Any]:
        """Randomise lights and plate for one render; returns the parameters used.

        view: the camera of this layout (default: the rig camera). Away from the rig view the
        calibrated empty-plate photo does not fit (it is the rig camera's picture), so another
        background is used, and the light panel stays above the camera so it never blocks the view.
        """
        self.view = view or self.cam
        pc = self.cfg.get("plate", {}) or {}
        lc = self.cfg.get("lights", {}) or {}
        rough = rng.uniform(*pc.get("roughness", [0.6, 0.9]))
        bump = rng.uniform(*pc.get("bump_strength", [0.0, 0.15]))
        nodes = bpy.data.materials[self.plate_mat_name].node_tree.nodes
        nodes["plate_rough"].outputs[0].default_value = rough
        _set(nodes["plate_bump"], ["Strength"], bump)
        background = bg.choose(pc, self.images, self.texture is not None and rig_view, rng)
        self._show_background(nodes, background)
        size = rng.uniform(*lc.get("panel_size_mm", [250, 400]))
        height = rng.uniform(*lc.get("height_mm", [320, 380]))
        if not rig_view:
            height = max(height, float(self.view.centre_in_plate[2]) + 30.0)
        off = lc.get("offset_mm", [0, 40])
        r, a = rng.uniform(*off), rng.uniform(0, 2 * math.pi)
        cx, cy = self.centre[0] + r * math.cos(a), self.centre[1] + r * math.sin(a)
        energy = rng.uniform(*lc.get("energy_w", [40, 120]))
        temp = rng.uniform(*lc.get("colour_temperature_k", [5000, 6500]))
        col = _kelvin_rgb(temp)
        self.light.set_location([cx * self.s, cy * self.s, height * self.s])
        self.light.set_rotation_euler([0, 0, rng.uniform(0, math.pi)])
        self.light.blender_obj.data.size = size * self.s
        self.light.blender_obj.data.size_y = size * self.s * rng.uniform(0.6, 1.0)
        self.light.set_energy(energy)
        self.light.set_color(col)
        world = rng.uniform(*lc.get("fill_world_strength", [0.0, 0.05]))
        bproc.renderer.set_world_background([1.0, 1.0, 1.0], world)
        exposure = rng.uniform(*self.cfg.get("render", {}).get("exposure", [0.0, 0.0]))
        bpy.context.scene.view_settings.exposure = exposure
        return {
            "plate_roughness": rough,
            "plate_bump": bump,
            "background": background,
            "plate_solid_colour": background.get("rgb"),
            "light_size_mm": size,
            "light_height_mm": height,
            "light_xy_mm": [cx, cy],
            "light_energy_w": energy,
            "light_temperature_k": temp,
            "world_strength": world,
            "exposure_ev": exposure,
        }


def set_environment(path: Path, strength: float, rotation_deg: float) -> None:
    """HDRI environment around the scene (lights it and shows in reflections).

    One texture node, reused for every layout (BlenderProc's own helper adds a new node per call);
    the image is loaded once per file.
    """
    nt = bpy.context.scene.world.node_tree
    background = next(n for n in nt.nodes if n.type == "BACKGROUND")
    tex = nt.nodes.get("env_tex")
    if tex is None:
        tex = nt.nodes.new("ShaderNodeTexEnvironment")
        tex.name = "env_tex"
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.name = "env_rot"
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    tex.image = bpy.data.images.load(str(path), check_existing=True)
    nt.nodes["env_rot"].inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    nt.links.new(tex.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = float(strength)


def _factor(node) -> Any:
    return next(s for s in node.inputs if s.name == "Factor" and s.type == "VALUE")


def _kelvin_rgb(t: float) -> list[float]:
    """Approximate RGB of a black body (Tanner Helland), normalised to max 1."""
    t /= 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = (
        99.4708025861 * math.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    b = (
        255.0
        if t >= 66
        else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    )
    rgb = [max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b)]
    m = max(rgb)
    return [c / m for c in rgb]


# ------------------------------------------------------------------------------ per instance


def place_instance(
    obj,
    children,
    T_plate_mm: np.ndarray,
    s: float,
    mass_kg: float,
    physics: Mapping[str, Any],
    active: bool = True,
    body: bool = True,
) -> None:
    """Pose the part and give it a compound collision body: active parts are simulated, passive
    ones (lying on the plate in a rest pose) are fixed obstacles. body=False: no collision body
    at all (nothing that moves can reach it), which saves Blender's slow rigid-body setup."""
    T = np.array(T_plate_mm, dtype=float)
    T[:3, 3] *= s
    obj.set_local2world_mat(T)
    if not body:
        return
    margin = float(physics.get("collision_margin_mm", 0.05)) * s
    obj.enable_rigidbody(
        active,
        collision_shape="COMPOUND",
        collision_margin=margin,
        mass=mass_kg,
        friction=float(physics.get("friction", 0.6)),
        angular_damping=float(physics.get("angular_damping", 0.3)),
        linear_damping=float(physics.get("linear_damping", 0.1)),
    )
    for c in children:
        c.enable_rigidbody(
            active,
            collision_shape="CONVEX_HULL",
            collision_margin=margin,
            friction=float(physics.get("friction", 0.6)),
        )


def assign_materials(obj, materials: Mapping[str, bpy.types.Material]) -> None:
    # on the mesh (each instance has its own, non-linked), not on the object: BlenderProc's
    # segmentation pass swaps obj.data.materials, and an object-linked material would override it
    mesh_mats = obj.blender_obj.data.materials
    for i in range(len(mesh_mats)):
        name = SLOTS[i] if i < len(SLOTS) else "body"
        mesh_mats[i] = materials.get(name, materials["body"])


def simulate(physics: Mapping[str, Any]) -> None:
    world = bpy.context.scene.rigidbody_world
    if world is not None:
        world.enabled = True  # BlenderProc disables it after every simulation
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=float(physics.get("min_time", 0.4)),
        max_simulation_time=float(physics.get("max_time", 1.5)),
        check_object_interval=float(physics.get("check_interval", 0.2)),
        substeps_per_frame=int(physics.get("substeps", 20)),
        solver_iters=int(physics.get("solver_iters", 20)),
    )


def final_pose_mm(obj, s: float) -> np.ndarray:
    T = np.array(obj.get_local2world_mat(), dtype=float)
    T[:3, 3] /= s
    return T


def datablock_counts() -> dict[str, int]:
    """Objects, meshes and materials alive in Blender; constant from scene to scene if nothing leaks."""
    d = bpy.data
    return {"objects": len(d.objects), "meshes": len(d.meshes), "materials": len(d.materials)}


def remove(objs: Sequence[Any], material_names: Sequence[str]) -> None:
    # by name: BlenderProc may already have freed the materials, and touching a freed Python
    # reference raises (or crashes Blender)
    # BlenderProc's delete() leaves the mesh data behind; with non-linked duplicates that leaks
    # every scene's meshes until Blender runs out of memory, so free them explicitly
    every = [e.blender_obj for o in objs for e in (o, *o.get_children())]
    mesh_names = {b.data.name for b in every if b.type == "MESH" and b.data is not None}
    for b in every:
        bpy.data.objects.remove(b, do_unlink=True)
    for name in mesh_names:
        m = bpy.data.meshes.get(name)
        if m is not None and m.users == 0:
            bpy.data.meshes.remove(m)
    for name in material_names:
        m = bpy.data.materials.get(name)
        if m is not None:
            bpy.data.materials.remove(m)


def render(depth: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # BlenderProc numbers the objects (pass_index) only once, in enable_segmentation_output; parts
    # created later keep 0 and vanish into the background of the instance map. Renumber every mesh
    # object (unique, non-zero) before each render; the decoder maps pass_index -> category_id.
    for i, o in enumerate(get_all_blender_mesh_objects(), start=1):
        o.pass_index = i
    data = bproc.renderer.render()
    rgb = np.asarray(data["colors"][0])[..., :3].astype(np.uint8)
    seg = np.asarray(data["category_id_segmaps"][0]).astype(np.uint16)
    dep = np.asarray(data["depth"][0], dtype=np.float32) if depth and "depth" in data else None
    return rgb, seg, dep
