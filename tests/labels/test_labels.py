import csv
import math

import cv2
import numpy as np
import pytest

from fake_scenes import PX_PER_MM, make_run, rotated_rect
from screwsynth.common.io import read_json
from screwsynth.labels import geometry as g
from screwsynth.labels import inspect as insp
from screwsynth.labels.cli import main
from screwsynth.labels.export import Settings, export, split_layouts

NAMES = {i: f"C{i}" for i in range(4)}
NAMES.update({0: "DIN912_M3", 1: "DIN912_M4", 2: "DIN912_M5", 3: "DIN934_M6"})


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


@pytest.mark.parametrize("angle", [0, 17, 45, 73, 120])
@pytest.mark.parametrize("width_mm", [2.0, 6.0])
def test_polygon_round_trip(angle, width_mm):
    """Rasterised polygon vs source mask, including a thin 2 mm object at 11.7 px/mm."""
    mask = np.zeros((500, 500), np.uint8)
    poly = rotated_rect(250, 250, 30 * PX_PER_MM, width_mm * PX_PER_MM, angle)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    p = g.polygon(mask.astype(bool), tol_px=0.7)
    assert len(p) >= 4
    assert _iou(g.rasterize(p, mask.shape), mask.astype(bool)) > 0.97


def test_polygon_round_trip_threaded_outline():
    """Saw-tooth outline (thread crests) keeps enough points."""
    mask = np.zeros((300, 600), np.uint8)
    pts = [(50, 150)]
    for k, x in enumerate(range(60, 540, 8)):
        pts.append((x, 125 if k % 2 else 130))
    pts += [(540, 150)]
    pts += [(x, 170 if k % 2 else 175) for k, x in reversed(list(enumerate(range(60, 540, 8))))]
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 1)
    p = g.polygon(mask.astype(bool), tol_px=0.7)
    assert _iou(g.rasterize(p, mask.shape), mask.astype(bool)) > 0.97


def test_split_mask_merged_with_bridges():
    mask = np.zeros((200, 300), bool)
    mask[50:70, 20:120] = True
    mask[50:70, 160:280] = True  # occluder split the object in two
    merged = g.polygon(mask, multi="merge")
    largest = g.polygon(mask, multi="largest")
    assert _iou(g.rasterize(merged, mask.shape), mask) > 0.95
    assert _iou(g.rasterize(largest, mask.shape), mask) < 0.6


def test_obb_and_bbox():
    mask = np.zeros((400, 400), np.uint8)
    poly = rotated_rect(200, 200, 200, 30, 30)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    box = g.obb(mask.astype(bool))
    sides = sorted(np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1))
    assert sides[2] == pytest.approx(200, abs=3) and sides[0] == pytest.approx(30, abs=3)
    x0, y0, x1, y1 = g.bbox(mask.astype(bool))
    ys, xs = np.nonzero(mask)
    assert (x0, y0, x1, y1) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def test_rle_round_trip():
    mask = np.random.default_rng(0).random((37, 51)) > 0.6
    assert np.array_equal(g.rle_decode(g.rle_encode(mask)), mask)


def test_split_never_leaks_and_is_deterministic():
    layouts = [f"{i:06d}" for i in range(200)]
    s1 = split_layouts(layouts * 3, {"train": 0.9, "val": 0.1}, seed=1)
    assert s1 == split_layouts(list(reversed(layouts)), {"train": 0.9, "val": 0.1}, seed=1)
    assert sum(v == "val" for v in s1.values()) == 20
    assert s1 != split_layouts(layouts, {"train": 0.9, "val": 0.1}, seed=2)


def test_export_all_formats(tmp_path):
    scenes = make_run(tmp_path / "out", "r1", n=10, variants=2)
    st = Settings(imgsz=320, split={"train": 0.8, "val": 0.2}, min_instances_per_class=50)
    summary = export(
        scenes, tmp_path / "ds", NAMES, ["seg", "det", "obb", "pose", "coco", "crops"], st
    )
    ds = tmp_path / "ds"
    assert summary["images"]["train"] + summary["images"]["val"] == 20
    # variants of one layout never end up in different splits
    split_of = {}
    for split in ("train", "val"):
        for p in (ds / "seg" / "images" / split).iterdir():
            layout = p.stem.split("_")[1]
            assert split_of.setdefault(layout, split) == split
    # groups.csv tells a trainer which images belong together (for leakage checks and re-splits)
    groups = list(csv.DictReader((ds / "seg" / "groups.csv").open(encoding="utf-8")))
    assert len(groups) == 20 and (ds / "coco" / "groups.csv").is_file()
    assert all(split_of[g["image"].split("_")[1]] == g["split"] for g in groups)
    assert len({g["group"] for g in groups}) == 10
    assert "groups: groups.csv" in (ds / "seg.yaml").read_text(encoding="utf-8")
    for f in ("seg", "det", "obb", "pose"):
        assert (ds / f"{f}.yaml").is_file()
        img = next((ds / f / "images" / "train").iterdir())
        assert max(cv2.imread(str(img)).shape[:2]) == 320
        assert (ds / f / "labels" / "train" / (img.stem + ".txt")).is_file()
    pose_yaml = (ds / "pose.yaml").read_text(encoding="utf-8")
    assert "kpt_shape: [3, 3]" in pose_yaml and "flip_idx: [0, 1, 2]" in pose_yaml
    line = (
        (ds / "pose" / "labels" / "train" / (img.stem + ".txt")).read_text().splitlines()[0].split()
    )
    assert len(line) == 14
    coco = read_json(ds / "coco" / "train.json")
    assert coco["annotations"] and coco["categories"][0]["keypoints"] == [
        "head_top",
        "bearing",
        "tip",
    ]
    rows = list(csv.DictReader((ds / "crops.csv").open(encoding="utf-8")))
    assert rows and all(r["class_name"].startswith("DIN912") for r in rows)
    crop = cv2.imread(str(ds / "crops" / "images" / rows[0]["crop"]))
    assert crop.shape[:2] == (384, 384) or crop.shape[0] == crop.shape[1]
    assert summary["classes_below_min"]  # 20 images x <=4 instances < 50
    assert insp.check(ds) == []


@pytest.mark.parametrize("mode", ["fixed", "scaled"])
def test_crop_keypoints_land_on_features(tmp_path, mode):
    scenes = make_run(tmp_path / "out", "r1", n=3)
    export(scenes, tmp_path / "ds", NAMES, ["crops"], Settings(crop_size_px=200, crop_mode=mode))
    scales = []
    for r in csv.DictReader((tmp_path / "ds" / "crops.csv").open(encoding="utf-8")):
        meta = read_json(f"{r['scene_dir']}/meta.json")
        obj = next(o for o in meta["objects"] if o["instance_id"] == int(r["instance_id"]))
        s = float(r["scale"])
        scales.append(s)
        for k in ("head_top", "bearing", "tip"):
            if r[f"{k}_v"] != "0":
                u, v = obj["keypoints"][k]["uv"]
                x = (float(r[f"{k}_x"]) + 0.5) / s - 0.5 + int(r["offset_x"])
                y = (float(r[f"{k}_y"]) + 0.5) / s - 0.5 + int(r["offset_y"])
                assert x == pytest.approx(u, abs=1e-6)
                assert y == pytest.approx(v, abs=1e-6)
    assert scales and (mode == "scaled" or all(s == 1 for s in scales))


def test_low_visibility_dropped(tmp_path):
    scenes = make_run(tmp_path / "out", "r1", n=2)
    meta_path = scenes[0] / "meta.json"
    meta = read_json(meta_path)
    meta["objects"][0]["visible_pixel_fraction"] = 0.1
    from screwsynth.common.io import write_json

    write_json(meta_path, meta)
    summary = export(scenes, tmp_path / "ds", NAMES, ["seg"], Settings())
    assert summary["dropped"] == 1
    rows = list(csv.DictReader((tmp_path / "ds" / "dropped.csv").open(encoding="utf-8")))
    assert "visible fraction" in rows[0]["reason"]


def test_check_detects_bad_labels(tmp_path):
    scenes = make_run(tmp_path / "out", "r1", n=2)
    export(scenes, tmp_path / "ds", NAMES, ["seg", "pose"], Settings())
    lab = next((tmp_path / "ds" / "pose" / "labels").rglob("*.txt"))
    lab.write_text("7 0.5 0.5 1.2 0.1 0 0 0 0 0 0 0 0 0\nabc\n", encoding="utf-8")
    problems = insp.check(tmp_path / "ds")
    assert any("unknown class" in p for p in problems)
    assert any("outside [0, 1]" in p for p in problems)
    assert any("unparsable" in p for p in problems)


def test_cli_export_preview_stats_merge(tmp_path, capsys):
    out = tmp_path / "output"
    make_run(out, "r1", n=4)
    make_run(out, "r2", n=4)
    classes = tmp_path / "classes.yaml"
    classes.write_text(
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()), encoding="utf-8"
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"seed: 0\npaths: {{output_dir: {out.as_posix()}, classes: {classes.as_posix()}}}\n"
        "labels: {stage1_imgsz: 256, min_instances_per_class: 10}\n",
        encoding="utf-8",
    )
    c = ["--config", str(cfg)]
    assert (
        main(
            [
                "export",
                "--run",
                "r1",
                "--out",
                str(tmp_path / "d1"),
                "--formats",
                "seg,det,pose,coco",
                *c,
            ]
        )
        == 0
    )
    assert main(["export", "--run", "r2", "--out", str(tmp_path / "d2"), *c]) == 0
    assert (
        main(["preview", "--dataset", str(tmp_path / "d1"), "--n", "4", "--split", "train", *c])
        == 0
    )
    assert (tmp_path / "d1" / "preview_train.jpg").is_file()
    assert main(["check", "--dataset", str(tmp_path / "d1"), *c]) == 0
    assert main(["stats", "--dataset", str(tmp_path / "d1"), *c]) == 0
    assert (
        main(
            ["merge", str(tmp_path / "d1"), str(tmp_path / "d2"), "--out", str(tmp_path / "m"), *c]
        )
        == 0
    )
    n1 = len(list((tmp_path / "d1" / "seg" / "images").rglob("*.jpg")))
    n2 = len(list((tmp_path / "d2" / "seg" / "images").rglob("*.jpg")))
    assert len(list((tmp_path / "m" / "seg" / "images").rglob("*.jpg"))) == n1 + n2
    assert str(tmp_path / "m").replace("\\", "/") in (tmp_path / "m" / "seg.yaml").read_text(
        encoding="utf-8"
    )
    # the class list and the parts table travel with every dataset, and from there into the model
    # package (a sorter reads both)
    seg_yaml = (tmp_path / "d1" / "seg.yaml").read_text(encoding="utf-8")
    assert "package_files: [classes.yaml, parts.json]" in seg_yaml
    assert (
        (tmp_path / "d1" / "seg" / "classes.yaml").read_text(encoding="utf-8").startswith("names:")
    )
    assert read_json(tmp_path / "d1" / "seg" / "parts.json")["format"] == "screwsynth-parts"
    assert (tmp_path / "m" / "seg" / "parts.json").is_file()
    assert (tmp_path / "m" / "seg" / "classes.yaml").is_file()


@pytest.mark.slow
def test_ultralytics_loads_pose_dataset(tmp_path):
    ultra = pytest.importorskip("ultralytics")
    from ultralytics.data.utils import check_det_dataset

    scenes = make_run(tmp_path / "out", "r1", n=4)
    export(
        scenes,
        tmp_path / "ds",
        NAMES,
        ["pose"],
        Settings(imgsz=256, split={"train": 0.5, "val": 0.5}),
    )
    data = check_det_dataset(str(tmp_path / "ds" / "pose.yaml"))
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    cfg = get_cfg(overrides={"task": "pose", "imgsz": 256})
    ds = build_yolo_dataset(cfg, data["train"], batch=2, data=data, mode="val")
    item = ds[0]
    assert item["keypoints"].shape[1:] == (3, 3)
    assert ultra is not None and math.isfinite(float(item["bboxes"].sum()))


def test_check_compares_washers_with_filled_area():
    """A washer's polygon has no hole; the checker must compare it with the filled mask area."""
    ring = np.zeros((200, 200), np.uint8)
    cv2.circle(ring, (100, 100), 60, 1, -1)
    cv2.circle(ring, (100, 100), 33, 0, -1)
    disc = math.pi * 60**2
    assert abs(insp._outer_area(ring > 0) - disc) / disc < 0.02
    assert ring.sum() / disc < 0.75  # the raw mask would fail the 0.8-1.2 band


def test_split_never_leaves_validation_empty():
    from screwsynth.labels.export import split_layouts

    for n in (2, 3, 5, 9):
        out = split_layouts([f"L{i}" for i in range(n)], {"train": 0.9, "val": 0.1}, seed=0)
        assert len(out) == n and list(out.values()).count("val") == 1, n
    one = split_layouts(["L0"], {"train": 0.9, "val": 0.1}, seed=0)
    assert one == {"L0": "train"}  # a single layout cannot be split
    big = split_layouts([f"L{i}" for i in range(1000)], {"train": 0.9, "val": 0.1}, seed=0)
    assert list(big.values()).count("val") == 100
