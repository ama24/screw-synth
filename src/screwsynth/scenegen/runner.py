"""Scene generation loop (runs inside `blenderproc run tools/scenegen/generate.py`)."""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from screwsynth.common.config import Config
from screwsynth.common.io import read_json, read_yaml, write_json, write_yaml
from screwsynth.common.paths import repo_root, resolve
from screwsynth.common.png import write_png
from screwsynth.common.seed import derive_seed

from . import camera as camera_mod
from . import dirt, environments
from .composition import BalancedSampler, BalanceState, parts_from_assets, plan_scene
from .groundtruth import (
    discard_reason,
    object_meta,
    silhouette_area,
    silhouette_mask,
    transform_points,
)
from .materials import pick_insert_finish, sample_material
from .placement import ClumpSettings, convex_hull_2d, place, place_clumped, separation
from .variation import (
    CameraSettings,
    EnvironmentSettings,
    layout_environment,
    layout_view,
    view_envelope,
)

log = logging.getLogger("screwsynth.scenegen")

STEEL_DENSITY_KG_MM3 = 7.85e-6
PLASTIC_DENSITY_KG_MM3 = 1.15e-6


@dataclass
class RunArgs:
    run: str
    scenes: int
    seed: int = 0
    start: int = 0
    resume: bool = False
    nominal_camera: bool = False
    preview: bool = False
    benchmark: bool = False
    # parallel renders: each worker keeps its own class-balance state and log (they would
    # overwrite one shared file); None = the single, sequential renderer
    worker: int | None = None

    def suffix(self) -> str:
        return "" if self.worker is None else f"_w{self.worker}"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Part:
    """Asset data needed per instance: part.json, rest poses, local footprints and collision vertices."""

    def __init__(self, assets_dir: Path, rest_dir: Path, part_id: str, collision_reader) -> None:
        self.meta = read_json(assets_dir / part_id / "part.json")
        self.rest = read_json(rest_dir / f"{part_id}.json")
        self.pieces = [
            np.asarray(p.vertices) for p in collision_reader(assets_dir / part_id / "collision.obj")
        ]
        self.verts = np.vstack(self.pieces)
        self.hulls = []
        self.heights = []  # top of the part in each rest pose (mm above the plate)
        for pose in self.rest["poses"]:
            v = transform_points(np.asarray(pose["transform"]), self.verts)
            self.hulls.append(convex_hull_2d(v[:, :2]))
            self.heights.append(float(v[:, 2].max() - v[:, 2].min()))

    def mass_kg(self, finish: str) -> float:
        density = PLASTIC_DENSITY_KG_MM3 if finish.startswith("nylon") else STEEL_DENSITY_KG_MM3
        return max(1e-5, float(self.meta["mesh"]["volume_mm3"]) * density)

    def sample_pose(self, rng: random.Random) -> int:
        return rng.choices(
            range(len(self.rest["poses"])), weights=[p["probability"] for p in self.rest["poses"]]
        )[0]


def _read_collision(path: Path):
    """Collision OBJ reader without trimesh (inside Blender)."""
    pieces, verts = [], None

    class _P:
        def __init__(self, v):
            self.vertices = np.asarray(v, dtype=float)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            if verts:
                pieces.append(_P(verts))
            verts = []
        elif line.startswith("v ") and verts is not None:
            verts.append([float(x) for x in line.split()[1:4]])
    if verts:
        pieces.append(_P(verts))
    return pieces


def _yaw_matrix(yaw: float, xy: tuple[float, float], lift: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3], T[2, 3] = xy[0], xy[1], lift
    return T


def _px_per_mm(cam, area) -> float:
    """Image scale at the centre of the working area (pixels per mm on the plate)."""
    cx, cy = (area[0] + area[2]) / 2, (area[1] + area[3]) / 2
    uv = cam.project(np.array([[cx, cy, 0.0], [cx + 1.0, cy, 0.0]]))
    return float(np.linalg.norm(uv[1] - uv[0]))


def _near(a, b, reach: float) -> bool:
    """Footprints within `reach` mm of each other (bounding circles)."""
    ca, cb = a.footprint.mean(axis=0), b.footprint.mean(axis=0)
    ra = float(np.linalg.norm(a.footprint - ca, axis=1).max())
    rb = float(np.linalg.norm(b.footprint - cb, axis=1).max())
    return float(np.linalg.norm(ca - cb)) < ra + rb + reach


def _lying_on(kept, parts, on_plate_mm: float) -> dict[int, list[int]]:
    """id -> ids of the parts physically resting on it, from the settled poses: a part off the
    plate whose footprint overlaps another one that reaches up to it lies on that one."""
    geo = {}
    for inst in kept:
        v = transform_points(inst["T"], parts[inst["info"].id].verts)
        geo[inst["id"]] = (convex_hull_2d(v[:, :2]), float(v[:, 2].min()), float(v[:, 2].max()))
    out: dict[int, list[int]] = {i: [] for i in geo}
    for b, (fb, zb, _) in geo.items():
        if zb <= on_plate_mm:
            continue
        for a, (fa, _, za) in geo.items():
            if a != b and za >= zb - 1.0 and separation(fa, fb) < -0.2:
                out[a].append(b)
    return out


def _settle(instances, objs, parts, area, physics, discard_cfg, dropped, bs, s):
    """Simulate the parts that can move (if any) and drop those that ended up off the plate.

    Discarded parts are deleted from the scene at once, so they can neither be rendered nor collide
    in a later settle (after a removal). Returns the parts that stay.
    """
    if any(inst["active"] for inst in instances):
        bs.simulate(physics)
    kept = []
    for inst in instances:
        if not inst["active"]:
            kept.append(inst)
            continue
        T = bs.final_pose_mm(inst["obj"], s)
        why = discard_reason(
            T,
            inst["T"],  # movement during this settle (after a removal, not since the start)
            parts[inst["info"].id].verts,
            area,
            float(discard_cfg.get("max_lift_mm", 0.3)),
            float(discard_cfg.get("max_move_mm", 30.0)),
            float(discard_cfg.get("below_plate_mm", 1.0)),
            allow_stacked=True,
            max_stack_mm=float(discard_cfg.get("max_stack_mm", 25.0)),
        )
        if why:
            dropped.append({"part_id": inst["info"].id, "reason": why})
            objs.remove(inst["obj"])
            bs.remove([inst["obj"]], [])
        else:
            inst["T"] = T
            kept.append(inst)
    return kept


def run(args: RunArgs, cfg: Config) -> dict[str, Any]:
    from . import blender_scene as bs  # bpy only from here on

    sg = dict(cfg.get("scenegen", {}) or {})
    ws = float(cfg.get("world_scale", 10))
    s = 0.001 * ws
    paths = {
        k: resolve(cfg.get(f"paths.{k}", d))
        for k, d in (
            ("assets_dir", "data/assets"),
            ("rest_poses_dir", "data/rest_poses"),
            ("calibration_dir", "data/calibration"),
            ("plate_dir", "data/plate"),
            ("hdri_dir", "data/hdri"),
            ("output_dir", "output"),
            ("classes", "configs/classes.yaml"),
            ("catalogue", "configs/catalogue.yaml"),
        )
    }
    out = paths["output_dir"] / args.run
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    from screwsynth.common.log import setup_logging

    setup_logging(cfg.get("logging.level", "INFO"), out / f"log{args.suffix()}.txt")

    render_cfg = dict(sg.get("render", {}) or {})
    scale = float(render_cfg.get("resolution_scale", 1.0))
    if args.preview:
        pv = dict(sg.get("preview", {}) or {})
        scale = float(pv.get("resolution_scale", 0.25))
        render_cfg["samples"] = int(pv.get("samples", 16))
    if args.nominal_camera:
        nc = dict(sg.get("nominal_camera", {}) or {})
        cam = camera_mod.nominal(
            nc.get("focal_length_mm", 4.74),
            nc.get("pixel_size_um", 1.4),
            tuple(nc.get("resolution", (4608, 2592))),
            nc.get("height_mm", 300.0),
            scale=scale,
        )
        area = (0.0, 0.0, 220.0, 220.0)
    else:
        calib = read_json(paths["calibration_dir"] / "calibration.json")
        cam = camera_mod.from_calibration(calib, scale)
        wa = calib["extrinsic"]["working_area_mm"]
        area = (
            wa["origin"][0],
            wa["origin"][1],
            wa["origin"][0] + wa["size"][0],
            wa["origin"][1] + wa["size"][1],
        )
    vis = cam.visible_plate_rect(-float(sg.get("area_margin_mm", 3.0)))
    area = (max(area[0], vis[0]), max(area[1], vis[1]), min(area[2], vis[2]), min(area[3], vis[3]))
    # camera views and HDRI environments per layout (defaults: the rig camera, no environment)
    views = CameraSettings.from_config(sg.get("camera"))
    target = ((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)
    envelope = None if views.is_rig_view else view_envelope(cam, views, target)
    env = EnvironmentSettings.from_config(sg.get("environment"))
    hdris = environments.usable(paths["hdri_dir"]) if env.enabled else []
    if env.enabled and not hdris:
        log.warning(
            "environment enabled but no HDRI files switched on in %s: rendering without one",
            paths["hdri_dir"],
        )

    classes_doc = read_yaml(paths["classes"])
    class_ids = {name: int(i) for i, name in classes_doc["names"].items()}
    from screwsynth.catalogue.schema import catalogue_from_config

    catalogue = catalogue_from_config(cfg)
    enabled = {p.id for p in catalogue.enabled_parts}
    infos = parts_from_assets(
        paths["assets_dir"], list(class_ids), enabled, paths["rest_poses_dir"]
    )
    missing = set(class_ids) - {p.class_name for p in infos}
    if missing:
        log.warning("classes without built assets/rest poses (not rendered): %s", sorted(missing))
    state_path = out / f"balance_state{args.suffix()}.json"
    sampler = BalancedSampler(
        infos,
        BalanceState.load(state_path),
        int(sg.get("balance_slack", 1)),
        sg.get("finish_weights"),
        float(sg.get("unknown_fraction", 0.0)),
        float(sg.get("catch_up_share", 0.25)),
    )

    run_cfg = {
        "run": args.run,
        "args": vars(args),
        "seed": args.seed,
        "git_commit": git_commit(),
        "config": cfg.to_dict(),
        "camera": cam.to_json(),
        "camera_views": views.to_json(),
        "environment": {"enabled": bool(hdris), "files": hdris},
        "area_mm": list(area),
        "classes_hash": classes_doc.get("catalogue_hash"),
        "world_scale": ws,
    }
    first = args.worker in (None, 0)  # parallel renders: one writer, the others would collide
    if first and (not (out / "run_config.yaml").is_file() or not args.resume):
        write_yaml(out / "run_config.yaml", json.loads(json.dumps(run_cfg)))

    import blenderproc as bproc

    bproc.init()
    bs.init_renderer(render_cfg)
    bs.set_camera(cam, ws)
    bs.check_camera(cam, ws)
    static = bs.StaticScene(
        cam, sg, paths["calibration_dir"], ws, paths["plate_dir"], envelope, bool(hdris)
    )
    library = bs.AssetLibrary(paths["assets_dir"], ws)
    parts: dict[str, Part] = {}
    physics = dict(sg.get("physics", {}) or {})
    discard_cfg = dict(sg.get("discard", {}) or {})
    variants = max(1, int(sg.get("variants_per_layout", 1)))
    measurable_tilt = float(cfg.get("restposes.measurable_tilt_deg", 15))
    window = int(sg.get("keypoint_window_px", 2))
    clump_st = ClumpSettings.from_config(sg.get("clumped"))
    lip = dict(sg.get("lip", {}) or {})
    # parts may reach up to the lip around the plate, just outside the frame (cut by the border)
    bounds = cam.visible_plate_rect(float(lip.get("outside_mm", 2.0)))
    if lip.get("enabled", True):
        static.add_lip(
            bounds, float(lip.get("height_mm", 6.0)), float(lip.get("thickness_mm", 5.0))
        )
    on_plate_mm = float(sg.get("on_plate_mm", 0.3))
    removal_range = tuple(int(x) for x in sg.get("removal_steps", (0, 0)))
    done_dir = out / "layouts_done"
    done_dir.mkdir(exist_ok=True)
    stats = {"done": 0, "skipped": 0, "failed": 0, "seconds": []}

    def scene_name(idx: int, r: int, v: int) -> str:
        return f"{idx:06d}" + (f"_r{r}" if r else "") + (f"_{v}" if variants > 1 else "")

    for idx in range(args.start, args.start + args.scenes):
        first = [scene_name(idx, 0, v) for v in range(variants)]
        legacy_done = removal_range == (0, 0) and all(
            (scenes_dir / n / "meta.json").is_file() for n in first
        )
        if args.resume and ((done_dir / f"{idx:06d}").is_file() or legacy_done):
            stats["skipped"] += 1
            continue
        t0 = time.perf_counter()
        timing: dict[str, float] = {}
        rng = random.Random(derive_seed(args.seed, "scene", idx))
        # one camera (and environment) per layout: the picker's photos are taken from the same spot
        view = layout_view(cam, views, target, idx, args.seed)
        vcam = view.camera
        if not view.rig_view:
            bs.set_camera(vcam, ws)
            bs.check_camera(vcam, ws)
        environment = layout_environment(hdris, env, idx, args.seed)
        px_mm = _px_per_mm(vcam, area)
        occlusion_min_px = max(1, round(float(sg.get("occlusion_min_mm2", 0.5)) * px_mm**2))
        # dirt has its own random streams: switching it on or off changes nothing else
        drng = random.Random(derive_seed(args.seed, "dirt", idx))
        plate_dirt = dirt.sample(
            sg.get("dirt"), "plate", random.Random(derive_seed(args.seed, "plate-dirt", idx))
        )
        static.set_plate_dirt(plate_dirt)  # the same plate for every photo of the layout
        rrng = random.Random(derive_seed(args.seed, "removals", idx))
        removals = rrng.randint(*removal_range) if removal_range[1] > 0 else 0
        objs, mats = [], []
        try:
            plan = plan_scene(sampler, sg, rng)
            chosen = []
            for info, finish in plan.parts:
                if info.id not in parts:
                    parts[info.id] = Part(
                        paths["assets_dir"], paths["rest_poses_dir"], info.id, _read_collision
                    )
                pose_i = parts[info.id].sample_pose(rng)
                chosen.append((info, finish, pose_i))
            hulls = [parts[i.id].hulls[p] for i, _, p in chosen]
            tries = int(sg.get("max_place_tries", 60))
            if plan.kind == "clumped":
                heights = [parts[i.id].heights[p] for i, _, p in chosen]
                placed = place_clumped(hulls, area, bounds, rng, clump_st, tries, heights)
            else:
                placed = place(hulls, area, rng, plan.gap_mm, plan.touch_probability, tries)
            timing["place"] = time.perf_counter() - t0
            instances = []
            # parts lying on the plate are already in an exact rest pose: fixed obstacles. Only
            # parts dropped on top (and crowded scenes, where footprints overlap) move, and only
            # parts within reach of a moving one need a collision body at all.
            moving = [pl for pl in placed if pl.layer > 1 or plan.kind == "crowded"]
            reach = float(physics.get("collision_reach_mm", 20.0))
            for pl in placed:
                info, finish, pose_i = chosen[pl.index]
                part = parts[info.id]
                active = pl.layer > 1 or plan.kind == "crowded"
                body = active or any(_near(pl, m, reach) for m in moving)
                if pl.layer > 1:  # dropped from just above the stack under it
                    lift = pl.base_z + float(physics.get("stack_clearance_mm", 0.5))
                else:
                    lift = float(physics.get("drop_mm", 0.05)) if active else 0.0
                T0 = _yaw_matrix(pl.yaw, pl.xy, lift) @ np.asarray(
                    part.rest["poses"][pose_i]["transform"]
                )
                obj, children = library.instance(info.id, part.meta)
                objs.append(obj)  # registered at once so a failure below still cleans it up
                bs.place_instance(obj, children, T0, s, part.mass_kg(finish), physics, active, body)
                material = {
                    "body": {
                        **sample_material(finish, sg.get("materials"), rng, sg.get("wear")),
                        "dirt": dirt.sample(sg.get("dirt"), "parts", drng),
                    }
                }
                if "insert" in part.meta["mesh"]["material_slots"]:
                    material["insert"] = sample_material(
                        pick_insert_finish(sg.get("insert_colours"), rng),
                        sg.get("materials"),
                        rng,
                        {"scratches": 0, "grime": 0, "fingerprints": 0},
                    )
                bmats = {
                    k: bs.make_material(f"{info.id}_{k}_{len(objs)}", v, s)
                    for k, v in material.items()
                }
                mats += [m.name for m in bmats.values()]
                bs.assign_materials(obj, bmats)
                instances.append(
                    {
                        "info": info,
                        "finish": finish,
                        "pose": pose_i,
                        "T0": T0,
                        "T": T0,
                        "obj": obj,
                        "material": material,
                        "active": active,
                        "layer": pl.layer,
                    }
                )
            timing["build"] = time.perf_counter() - t0 - timing["place"]
            dropped: list[dict[str, Any]] = []
            kept = _settle(instances, objs, parts, area, physics, discard_cfg, dropped, bs, s)
            timing["physics"] = time.perf_counter() - t0 - timing["place"] - timing["build"]
            for k, inst in enumerate(kept, start=1):
                inst["obj"].set_cp("category_id", k)
                inst["id"] = k
            counted = [inst["info"].class_name for inst in kept]
            removed: list[int] = []
            images = 0
            for r in range(removals + 1):
                if r > 0:  # the picker takes one free part, the rest settles, a new photo
                    free = [inst for inst in kept if inst.get("free")]
                    if not free:
                        break
                    victim = rrng.choice(free)
                    removed.append(victim["id"])
                    # by identity: list.remove() compares dicts, and their numpy arrays fail
                    kept = [inst for inst in kept if inst is not victim]
                    objs.remove(victim["obj"])
                    bs.remove([victim["obj"]], [])
                    kept = _settle(kept, objs, parts, area, physics, discard_cfg, dropped, bs, s)
                on_it = _lying_on(kept, parts, on_plate_mm)
                shapes = {}
                for inst in kept:
                    part = parts[inst["info"].id]
                    pieces = [transform_points(inst["T"], q) for q in part.pieces]
                    shapes[inst["id"]] = (
                        silhouette_area(vcam, pieces),
                        silhouette_mask(vcam, pieces),
                        float(transform_points(inst["T"], part.verts)[:, 2].min()),
                    )
                for v in range(variants):
                    name = scene_name(idx, r, v)
                    light = static.jitter(
                        random.Random(derive_seed(args.seed, "light", idx, v)),
                        vcam,
                        view.rig_view,
                    )
                    light["plate_dirt"] = dict(plate_dirt)
                    if environment is not None:
                        bs.set_environment(
                            paths["hdri_dir"] / environment["file"],
                            environment["strength"],
                            environment["rotation_deg"],
                        )
                        light["environment"] = dict(environment)
                        if not env.panel_light:
                            static.light.set_energy(0.0)
                            light["light_energy_w"] = 0.0
                    rgb, seg, dep = bs.render(bool(render_cfg.get("depth", False)))
                    objects = []
                    for inst in kept:
                        part = parts[inst["info"].id]
                        area_px, sil, zmin = shapes[inst["id"]]
                        objects.append(
                            object_meta(
                                inst_id=inst["id"],
                                part=part.meta,
                                class_id=class_ids[part.meta["class_name"]],
                                finish=inst["finish"],
                                material=inst["material"]["body"],
                                T_plate=inst["T"],
                                rest_pose_index=inst["pose"],
                                measurable_tilt_deg=measurable_tilt,
                                cam=vcam,
                                rig_view=view.rig_view,
                                instance=seg,
                                alone_area=area_px,
                                window=window,
                                silhouette=sil,
                                occlusion_min_px=occlusion_min_px,
                                on_plate_mm=on_plate_mm,
                                lowest_z_mm=zmin,
                                lying_on_it=on_it[inst["id"]],
                            )
                        )
                    if v == 0:
                        free_ids = {o["instance_id"] for o in objects if o["free"]}
                        for inst in kept:
                            inst["free"] = inst["id"] in free_ids
                    ids_in_mask = sorted(int(i) for i in np.unique(seg) if i)
                    missing_ids = sorted({o["instance_id"] for o in objects} - set(ids_in_mask))
                    d = scenes_dir / name
                    d.mkdir(parents=True, exist_ok=True)
                    write_png(d / "rgb.png", rgb)
                    write_png(d / "instance.png", seg)
                    if dep is not None:
                        np.save(d / "depth.npy", dep)
                    write_json(
                        d / "meta.json",
                        {
                            "scene_id": name,
                            "layout_id": f"{idx:06d}",
                            "variant": v,
                            "removal_step": r,
                            "removed_ids": list(removed),
                            "seed": args.seed,
                            "scene_kind": plan.kind,
                            "units": "mm",
                            "image": "rgb.png",
                            "instance_mask": "instance.png",
                            "camera": vcam.to_json(),
                            "camera_view": view.to_json(),
                            "world_scale": ws,
                            "area_mm": list(area),
                            "lighting": light,
                            "objects": objects,
                            "dropped": dropped,
                            "fully_hidden_ids": missing_ids,
                            "render": {
                                "samples": render_cfg.get("samples"),
                                "denoiser": render_cfg.get("denoiser"),
                                "resolution": list(cam.resolution),
                            },
                        },
                    )
                    images += 1
            timing["render"] = time.perf_counter() - t0 - sum(timing.values())
            sampler.record(counted)
            sampler.state.save(state_path)
            (done_dir / f"{idx:06d}").write_text(str(images), encoding="utf-8")
            dt = time.perf_counter() - t0
            stats["done"] += 1
            stats["seconds"].append(dt)
            log.info(
                "scene %06d: %d objects (%d dropped, %d simulated), %s, %d image(s), %.1f s "
                "(place %.1f, build %.1f, physics %.1f, render+gt %.1f)",
                idx,
                len(counted),
                len(dropped),
                sum(1 for i in instances if i["active"]),
                plan.kind,
                images,
                dt,
                timing["place"],
                timing["build"],
                timing["physics"],
                timing["render"],
            )
        except Exception as exc:  # one bad scene must not stop the run
            stats["failed"] += 1
            log.error("scene %06d failed: %s\n%s", idx, exc, traceback.format_exc())
            with (out / "failed.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"scene": idx, "error": str(exc)}) + "\n")
        finally:
            bs.remove(objs, mats)
            log.debug("datablocks after scene %06d: %s", idx, bs.datablock_counts())
    if stats["seconds"]:
        log.info(
            "rendered %d scenes, %.1f s/scene (mean), skipped %d, failed %d",
            stats["done"],
            float(np.mean(stats["seconds"])),
            stats["skipped"],
            stats["failed"],
        )
    return stats
