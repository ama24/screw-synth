"""Convert processed scenes into training datasets (YOLO seg/det/obb/pose, COCO, stage-2 crops)."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.common.io import read_json, write_json, write_text

from . import geometry as g

log = logging.getLogger(__name__)

FORMATS = ("seg", "det", "obb", "pose", "coco", "crops")
DEFAULT_FORMATS = ("seg", "pose", "coco")
KEYPOINTS = ("head_top", "bearing", "tip")
FLIP_IDX = [0, 1, 2]  # keypoints have no left/right twin


@dataclass
class Settings:
    imgsz: int = 1024
    min_visible_fraction: float = 0.25
    min_area_px: int = 60
    polygon_tolerance_px: float = 0.7
    polygon_min_points: int = 4
    multi_piece: str = "merge"  # merge | largest
    crop_size_px: int = 384
    crop_mode: str = "scaled"  # fixed | scaled (measure.crops)
    crop_margin: float = 1.2
    split: Mapping[str, float] = field(default_factory=lambda: {"train": 0.9, "val": 0.1})
    seed: int = 0
    min_instances_per_class: int = 3000
    image_format: str = "jpg"
    val_real: str | None = None

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None, seed: int = 0) -> Settings:
        s = dict(section or {})
        return cls(
            imgsz=int(s.get("stage1_imgsz", 1024)),
            min_visible_fraction=float(s.get("min_visible_fraction", 0.25)),
            min_area_px=int(s.get("min_area_px", 60)),
            polygon_tolerance_px=float(s.get("polygon_tolerance_px", 0.7)),
            polygon_min_points=int(s.get("polygon_min_points", 4)),
            multi_piece=str(s.get("multi_piece", "merge")),
            crop_size_px=int(s.get("crop_size_px", 384)),
            crop_mode=str(s.get("crop_mode", "scaled")),
            crop_margin=float(s.get("crop_margin", 1.2)),
            split=dict(s.get("split", {"train": 0.9, "val": 0.1})),
            seed=seed,
            min_instances_per_class=int(s.get("min_instances_per_class", 3000)),
            image_format=str(s.get("image_format", "jpg")),
            val_real=s.get("val_real"),
        )


@dataclass
class Instance:
    obj: dict[str, Any]
    mask: np.ndarray
    poly: np.ndarray  # full-res pixels
    box: tuple[float, float, float, float]
    obb: np.ndarray
    kps: np.ndarray  # (3, 3): x, y, v (COCO visibility)


# ------------------------------------------------------------------------------------- splits


def split_layouts(
    layout_ids: Sequence[str], fractions: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Deterministic split by layout (never splits one scene's objects/variants across sets)."""
    ids = sorted(set(layout_ids), key=lambda s: hashlib.sha256(f"{seed}/{s}".encode()).hexdigest())
    names = [n for n in fractions if fractions[n] > 0]
    total = sum(fractions[n] for n in names)
    counts = {n: round(len(ids) * fractions[n] / total) for n in names}
    # small runs: rounding can leave a split empty (3 layouts at 90/10 -> 3/0), and a dataset
    # without validation images cannot be trained: every split gets at least one layout, taken
    # from the largest split, as long as there are enough layouts
    if len(ids) >= len(names):
        for n in names:
            while counts[n] == 0:
                donor = max(names, key=lambda m: counts[m])
                counts[donor] -= 1
                counts[n] += 1
    counts[names[0]] += len(ids) - sum(counts.values())  # rounding remainder
    out, start = {}, 0
    for name in names:
        for lid in ids[start : start + counts[name]]:
            out[lid] = name
        start += counts[name]
    return out


# ----------------------------------------------------------------------------------- objects


def instances_of(
    meta: Mapping[str, Any], mask: np.ndarray, st: Settings
) -> tuple[list[Instance], list[dict]]:
    keep, dropped = [], []
    for obj in meta["objects"]:
        m = mask == obj["instance_id"]
        area = int(m.sum())
        why = None
        if obj.get("visible_pixel_fraction", 1.0) < st.min_visible_fraction:
            why = f"visible fraction {obj.get('visible_pixel_fraction'):.2f} < {st.min_visible_fraction}"
        elif area < st.min_area_px:
            why = f"mask area {area} px < {st.min_area_px}"
        if why:
            dropped.append(
                {
                    "scene": meta["scene_id"],
                    "instance_id": obj["instance_id"],
                    "part_id": obj["part_id"],
                    "reason": why,
                }
            )
            continue
        poly = g.polygon(m, st.polygon_tolerance_px, st.multi_piece, st.polygon_min_points)
        kps = np.zeros((3, 3))
        if obj.get("keypoints_labelled"):
            for i, k in enumerate(KEYPOINTS):
                kp = obj["keypoints"][k]
                u, v = kp["uv"]
                inside = 0 <= u <= mask.shape[1] - 1 and 0 <= v <= mask.shape[0] - 1
                if kp["visibility"] and inside:  # keypoints outside the image are not labelled
                    kps[i] = [u, v, kp["visibility"]]
        keep.append(Instance(obj, m, poly, g.bbox(m), g.obb(m), kps))
    return keep, dropped


def _fmt(vals: Sequence[float]) -> str:
    return " ".join(f"{v:.6f}" for v in vals)


def yolo_lines(insts: Sequence[Instance], w: int, h: int) -> dict[str, list[str]]:
    out = {f: [] for f in ("seg", "det", "obb", "pose")}
    norm = np.array([w, h], dtype=float)
    for ins in insts:
        c = ins.obj["class_id"]
        x0, y0, x1, y1 = ins.box
        cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
        # YOLO convention: pixel index / size (Ultralytics rasterises x * w with fillPoly)
        poly = np.clip(ins.poly / norm, 0, 1)
        out["seg"].append(f"{c} {_fmt(poly.ravel())}")
        out["det"].append(f"{c} {_fmt([cx, cy, bw, bh])}")
        out["obb"].append(f"{c} {_fmt(np.clip(ins.obb / norm, 0, 1).ravel())}")
        k = ins.kps.copy()
        k[:, 0] = np.where(k[:, 2] > 0, k[:, 0] / w, 0)
        k[:, 1] = np.where(k[:, 2] > 0, k[:, 1] / h, 0)
        out["pose"].append(
            f"{c} {_fmt([cx, cy, bw, bh])} "
            + " ".join(f"{x:.6f} {y:.6f} {int(v)}" for x, y, v in np.clip(k, 0, None))
        )
    return out


# ------------------------------------------------------------------------------------- crops


def make_crop(image: np.ndarray, ins: Instance, st: Settings) -> tuple[np.ndarray, dict[str, Any]]:
    """Full-resolution crop centred on the object; keypoints in crop coordinates."""
    from screwsynth.measure.crops import crop_window

    h, w = image.shape[:2]
    ox, oy, side, scale = crop_window(
        ins.box, (w, h), st.crop_size_px, st.crop_mode, st.crop_margin
    )
    crop = image[oy : oy + side, ox : ox + side]
    if side != st.crop_size_px:
        crop = cv2.resize(crop, (st.crop_size_px, st.crop_size_px), interpolation=cv2.INTER_AREA)
    k = ins.kps.copy()
    k[:, 0] = np.where(k[:, 2] > 0, ((k[:, 0] + 0.5) - ox) * scale - 0.5, 0)
    k[:, 1] = np.where(k[:, 2] > 0, ((k[:, 1] + 0.5) - oy) * scale - 0.5, 0)
    inside = (
        (k[:, 0] >= 0)
        & (k[:, 0] <= st.crop_size_px - 1)
        & (k[:, 1] >= 0)
        & (k[:, 1] <= st.crop_size_px - 1)
    )
    k[~inside, 2] = 0
    return crop, {"offset_x": ox, "offset_y": oy, "side_px": side, "scale": scale, "keypoints": k}


# ----------------------------------------------------------------------------------- writing


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


GROUPS_FILE = "groups.csv"


def _write_dataset_yaml(
    path: Path,
    root: Path,
    names: Mapping[int, str],
    pose: bool,
    real: str | None,
    val: str = "images/val",
    package_files: Sequence[str] = (),
) -> None:
    lines = [f"path: {root.as_posix()}", "train: images/train", f"val: {val}"]
    if real:
        lines.append(f"val_real: {real}")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in sorted(names.items())]
    if pose:
        lines += ["kpt_shape: [3, 3]", f"flip_idx: {FLIP_IDX}", f"kpt_names: {list(KEYPOINTS)}"]
    if (root / GROUPS_FILE).is_file():
        lines.append(f"groups: {GROUPS_FILE}  # image -> layout; one group never spans two splits")
    if package_files:  # the trainer copies these into the model package (a sorter reads them)
        lines.append(f"package_files: [{', '.join(package_files)}]")
    write_text(path, "\n".join(lines) + "\n")


def _write_groups(path: Path, groups: Sequence[tuple[str, str, str]]) -> None:
    """image file name, split, group (the layout: every photo of one layout shares it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "split", "group"])
        wr.writerows(groups)


def export(
    scene_dirs: Sequence[Path],
    out: Path,
    names: Mapping[int, str],
    formats: Sequence[str],
    st: Settings,
    package_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Export processed scenes to datasets under `out`.

    package_files: file name -> text (classes.yaml, parts.json), written into every format folder
    and listed in its dataset yaml, for the model package."""
    formats = [f for f in formats if f in FORMATS]
    metas = {d: read_json(d / "meta.json") for d in scene_dirs}
    splits = split_layouts(
        [m.get("layout_id", m["scene_id"]) for m in metas.values()], st.split, st.seed
    )
    staging = out / "_images"
    yolo_fmts = [f for f in formats if f in ("seg", "det", "obb", "pose")]
    coco: dict[str, dict] = {
        s: {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": i + 1,
                    "name": n,
                    "supercategory": "fastener",
                    "keypoints": list(KEYPOINTS),
                    "skeleton": [[1, 2], [2, 3]],
                }
                for i, n in sorted(names.items())
            ],
        }
        for s in st.split
    }
    stats: dict[str, Any] = {
        "images": Counter(),
        "instances": Counter(),
        "per_class": Counter(),
        "per_finish": Counter(),
        "per_size": Counter(),
        "per_length": Counter(),
        "dropped": 0,
        "views": {k: Counter() for k in VIEW_BINS},
        "view_angle_per_class": defaultdict(Counter),
    }
    dropped_all, sources, crop_rows = [], [], []
    groups: list[tuple[str, str, str]] = []
    ann_id = 1
    for img_id, (d, meta) in enumerate(
        sorted(metas.items(), key=lambda kv: kv[1]["scene_id"]), start=1
    ):
        split = splits[meta.get("layout_id", meta["scene_id"])]
        image = cv2.imread(str(d / meta["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(d / "instance.png"), cv2.IMREAD_UNCHANGED)
        h, w = mask.shape
        insts, dropped = instances_of(meta, mask, st)
        dropped_all += dropped
        name = f"{d.parent.parent.name}_{meta['scene_id']}"
        s = st.imgsz / max(w, h)
        small = (
            cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            if s < 1
            else image
        )
        img_path = staging / split / f"{name}.{st.image_format}"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(img_path), small, [cv2.IMWRITE_JPEG_QUALITY, 95] if st.image_format == "jpg" else []
        )
        groups.append((img_path.name, split, str(meta.get("layout_id", meta["scene_id"]))))
        sources.append(
            {
                "image": f"{split}/{img_path.name}",
                "scene_dir": str(d),
                "full_res_image": str(d / meta["image"]),
                "width": w,
                "height": h,
                "scale": s,
            }
        )
        lines = yolo_lines(insts, w, h)
        for f in yolo_fmts:
            _link_or_copy(img_path, out / f / "images" / split / img_path.name)
            write_text(
                out / f / "labels" / split / f"{name}.txt",
                "\n".join(lines[f]) + ("\n" if lines[f] else ""),
            )
        stats["images"][split] += 1
        for k, v in (meta.get("camera_view") or {}).items():
            if k in VIEW_BINS:
                stats["views"][k][_bin(v, VIEW_BINS[k])] += 1
        if "coco" in formats:
            cc = coco[split]
            cc["images"].append(
                {
                    "id": img_id,
                    "file_name": f"{split}/{img_path.name}",
                    "width": small.shape[1],
                    "height": small.shape[0],
                    "scene_dir": str(d),
                }
            )
            for ins in insts:
                poly = (ins.poly + 0.5) * s
                x0, y0, x1, y1 = (np.array(ins.box) * s).tolist()
                kp = ins.kps.copy()
                kp[:, :2] = np.where(kp[:, 2:3] > 0, (kp[:, :2] + 0.5) * s, 0)
                cc["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": ins.obj["class_id"] + 1,
                        "iscrowd": 0,
                        "segmentation": [np.round(poly.ravel(), 2).tolist()],
                        "area": float(ins.mask.sum() * s * s),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "keypoints": np.round(kp.ravel(), 2).tolist(),
                        "num_keypoints": int((kp[:, 2] > 0).sum()),
                        "instance_id": ins.obj["instance_id"],
                        "part_id": ins.obj["part_id"],
                        "finish": ins.obj["finish"],
                        # for the picker and the measurement (tool 08); absent in older renders
                        "attributes": {
                            k: ins.obj[k]
                            for k in ("support", "free", "lying_on_it", "measurable")
                            if k in ins.obj
                        },
                    }
                )
                ann_id += 1
        for ins in insts:
            o = ins.obj
            stats["instances"][split] += 1
            stats["per_class"][o["class_name"]] += 1
            stats["per_finish"][o["finish"]] += 1
            stats["per_size"][o["size"]] += 1
            if o.get("view_angle_deg") is not None:
                stats["view_angle_per_class"][o["class_name"]][_bin(o["view_angle_deg"], 10)] += 1
            if o.get("length") is not None:
                stats["per_length"][f"{o['class_name']}x{o['length']:g}"] += 1
            if "crops" in formats and o.get("measurable") and o.get("keypoints_labelled"):
                crop, info = make_crop(image, ins, st)
                cname = f"{name}_{o['instance_id']:03d}.png"
                cpath = out / "crops" / "images" / split / cname
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cpath), crop)
                k = info["keypoints"]
                cs = st.crop_size_px
                bx0 = ((ins.box[0] - info["offset_x"]) * info["scale"]) / cs
                by0 = ((ins.box[1] - info["offset_y"]) * info["scale"]) / cs
                bx1 = ((ins.box[2] - info["offset_x"]) * info["scale"]) / cs
                by1 = ((ins.box[3] - info["offset_y"]) * info["scale"]) / cs
                bx0, by0, bx1, by1 = np.clip([bx0, by0, bx1, by1], 0, 1)
                label = (
                    f"0 {_fmt([(bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0])} "
                    + " ".join(
                        f"{x / cs if v else 0:.6f} {y / cs if v else 0:.6f} {int(v)}"
                        for x, y, v in k
                    )
                )
                write_text(
                    out / "crops" / "labels" / split / cname.replace(".png", ".txt"), label + "\n"
                )
                crop_rows.append(
                    {
                        "crop": f"{split}/{cname}",
                        "split": split,
                        "scene_dir": str(d),
                        "instance_id": o["instance_id"],
                        "part_id": o["part_id"],
                        "class_name": o["class_name"],
                        "class_id": o["class_id"],
                        "size": o["size"],
                        "length_mm": o["length"],
                        "nominal_length_mode": o.get("nominal_length_mode"),
                        "axis_tilt_deg": o.get("axis_tilt_deg"),
                        "axis_height_head_mm": o.get("axis_height_head_mm"),
                        "axis_height_tip_mm": o.get("axis_height_tip_mm"),
                        "axis_height_head_top_mm": o.get("axis_height_head_top_mm"),
                        "offset_x": info["offset_x"],
                        "offset_y": info["offset_y"],
                        "crop_side_px": info["side_px"],
                        "scale": info["scale"],
                    }
                )
                for i, n in enumerate(KEYPOINTS):  # keypoints in crop pixels ("" when not labelled)
                    crop_rows[-1][f"{n}_x"] = float(k[i, 0]) if k[i, 2] else ""
                    crop_rows[-1][f"{n}_y"] = float(k[i, 1]) if k[i, 2] else ""
                    crop_rows[-1][f"{n}_v"] = int(k[i, 2])
    stats["dropped"] = len(dropped_all)
    # dataset files
    real = st.val_real
    extras = dict(package_files or {})
    for f in [*yolo_fmts, *(["coco"] if "coco" in formats else [])]:
        _write_groups(out / f / GROUPS_FILE, groups)
        for name, text in extras.items():
            write_text(out / f / name, text)
    for f in yolo_fmts:
        pose = f == "pose"
        _write_dataset_yaml(out / f"{f}.yaml", out / f, names, pose, real, package_files=[*extras])
        if real:
            _write_dataset_yaml(
                out / f"{f}_real.yaml", out / f, names, pose, None, real, package_files=[*extras]
            )
    if "crops" in formats:
        _write_dataset_yaml(out / "crops.yaml", out / "crops", {0: "screw"}, True, None)
        with (out / "crops.csv").open("w", newline="", encoding="utf-8") as fh:
            if crop_rows:
                wr = csv.DictWriter(fh, fieldnames=list(crop_rows[0]))
                wr.writeheader()
                wr.writerows(crop_rows)
    if "coco" in formats:
        for split, cc in coco.items():
            cc["info"] = {
                "description": "screw-synth",
                "coordinates": "exported (stage-1) image pixels",
            }
            write_json(out / "coco" / f"{split}.json", cc)
            for im in cc["images"]:
                _link_or_copy(staging / im["file_name"], out / "coco" / "images" / im["file_name"])
    shutil.rmtree(staging, ignore_errors=True)
    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(
            fh, fieldnames=["image", "scene_dir", "full_res_image", "width", "height", "scale"]
        )
        wr.writeheader()
        wr.writerows(sources)
    with (out / "dropped.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["scene", "instance_id", "part_id", "reason"])
        wr.writeheader()
        wr.writerows(dropped_all)
    summary = finalize_stats(stats, names, st)
    write_json(out / "stats.json", summary)
    write_json(
        out / "export.json",
        {
            "formats": list(formats),
            "settings": {k: v for k, v in vars(st).items()},
            "scenes": len(scene_dirs),
            "names": {int(k): v for k, v in names.items()},
        },
    )
    return summary


VIEW_BINS = {"height_mm": 25.0, "tilt_deg": 5.0, "tilt_direction_deg": 30.0, "roll_deg": 30.0}


def _bin(value: float, width: float) -> str:
    lo = math.floor(float(value) / width) * width
    return f"{lo:g}-{lo + width:g}"


def finalize_stats(
    stats: Mapping[str, Any], names: Mapping[int, str], st: Settings
) -> dict[str, Any]:
    per_class = {n: int(stats["per_class"].get(n, 0)) for _, n in sorted(names.items())}
    low = sorted(n for n, c in per_class.items() if c < st.min_instances_per_class)
    return {
        "images": dict(stats["images"]),
        "instances": dict(stats["instances"]),
        "per_class": per_class,
        "per_finish": dict(sorted(stats["per_finish"].items())),
        "per_size": dict(sorted(stats["per_size"].items())),
        "per_length": dict(sorted(stats["per_length"].items())),
        "dropped": stats["dropped"],
        # images per camera-view bin and objects per viewing-angle bin (0 = seen from above)
        "camera_views": {
            k: dict(sorted(c.items(), key=lambda kv: float(kv[0].split("-")[0])))
            for k, c in stats.get("views", {}).items()
            if c
        },
        "view_angle_per_class": {
            n: dict(sorted(c.items(), key=lambda kv: float(kv[0].split("-")[0])))
            for n, c in sorted(stats.get("view_angle_per_class", {}).items())
        },
        "min_instances_per_class": st.min_instances_per_class,
        "classes_below_min": low,
    }


def load_names(classes_yaml: Path) -> dict[int, str]:
    from screwsynth.common.io import read_yaml

    doc = read_yaml(classes_yaml)
    return {int(k): str(v) for k, v in doc["names"].items()}


def merge(datasets: Sequence[Path], out: Path) -> dict[str, Any]:
    """Combine exported datasets (same classes) into one; files are hard-linked when possible."""
    names = None
    totals: dict[str, Counter] = defaultdict(Counter)
    for ds in datasets:
        exp = read_json(ds / "export.json")
        n = {int(k): v for k, v in exp["names"].items()}
        if names is not None and n != names:
            raise ValueError(f"{ds}: class names differ from the first dataset")
        names = n
        for f in ("seg", "det", "obb", "pose", "crops", "coco"):
            src = ds / f
            if not src.is_dir():
                continue
            for p in src.rglob("*"):
                if p.is_file() and (p.suffix != ".json" or p.name == "parts.json"):
                    _link_or_copy(p, out / f / p.relative_to(src))
        st = read_json(ds / "stats.json")
        for key in ("per_class", "per_finish", "per_size", "per_length", "images", "instances"):
            totals[key].update(st.get(key, {}))
        for y in ds.glob("*.yaml"):
            text = y.read_text(encoding="utf-8").replace(ds.as_posix(), out.as_posix())
            write_text(out / y.name, text)
    write_json(out / "stats.json", {k: dict(v) for k, v in totals.items()})
    write_json(out / "export.json", {"merged_from": [str(d) for d in datasets], "names": names})
    for csv_name in ("crops.csv", "sources.csv"):
        rows = []
        for ds in datasets:
            p = ds / csv_name
            if p.is_file():
                rows += list(csv.DictReader(p.open(encoding="utf-8")))
        if rows:
            with (out / csv_name).open("w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    log.info("merged %d datasets into %s", len(datasets), out)
    return json.loads(json.dumps({k: dict(v) for k, v in totals.items()}))
