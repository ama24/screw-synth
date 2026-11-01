import math
import random
import shutil
import subprocess
from collections import Counter

import cv2
import numpy as np
import pytest

from screwsynth.calibration.geometry import Camera
from screwsynth.common.paths import repo_root
from screwsynth.common.png import read_png, write_png
from screwsynth.scenegen import camera as cammod
from screwsynth.scenegen.composition import BalancedSampler, BalanceState, PartInfo, plan_scene
from screwsynth.scenegen.groundtruth import (
    OCCLUDED,
    VISIBLE,
    discard_reason,
    keypoint_visibility,
    object_meta,
    occluders,
    silhouette_area,
    silhouette_mask,
)
from screwsynth.scenegen.materials import DEFAULT_PRESETS, pick_insert_finish, sample_material
from screwsynth.scenegen.placement import (
    ClumpSettings,
    convex_hull_2d,
    inside_area,
    nearest_gaps,
    place,
    place_clumped,
    separation,
)

# ----------------------------------------------------------------------------------- PNG


def test_png_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    mask = rng.integers(0, 65535, (37, 53), dtype=np.uint16)
    gray = rng.integers(0, 256, (10, 7), dtype=np.uint8)
    write_png(tmp_path / "rgb.png", rgb)
    write_png(tmp_path / "mask.png", mask)
    write_png(tmp_path / "gray.png", gray)
    assert np.array_equal(read_png(tmp_path / "rgb.png"), rgb)
    assert np.array_equal(read_png(tmp_path / "mask.png"), mask)
    assert np.array_equal(read_png(tmp_path / "gray.png"), gray)
    # OpenCV reads them too (BGR order, 16-bit preserved)
    assert np.array_equal(cv2.imread(str(tmp_path / "rgb.png"))[..., ::-1], rgb)
    assert np.array_equal(cv2.imread(str(tmp_path / "mask.png"), cv2.IMREAD_UNCHANGED), mask)


def test_png_reads_opencv_files(tmp_path):
    img = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "cv.png"), img)  # OpenCV uses adaptive filters
    assert np.array_equal(read_png(tmp_path / "cv.png"), img[..., ::-1])


# -------------------------------------------------------------------------------- camera


def test_nominal_camera_geometry():
    cam = cammod.nominal()
    assert cam.resolution == (2560, 2560)
    assert cam.K[0, 2] == pytest.approx((4608 - 1) / 2 - cam.roi["x0"])
    centre = cam.project(np.array([[110.0, 110.0, 0.0]]))[0]
    assert np.allclose(centre, [(2560 - 1) / 2, (2560 - 1) / 2], atol=1e-6)
    # plate x to the right, plate y up in the image
    p = cam.project(np.array([[120.0, 110, 0], [110, 120, 0]]))
    assert p[0, 0] > centre[0] and p[1, 1] < centre[1]
    assert cam.visible_plate_rect()[2] - cam.visible_plate_rect()[0] == pytest.approx(
        2560 / (3385.7 / 300), rel=1e-3
    )


def test_camera_matches_calibration_geometry():
    """The render camera (ROI crop) projects like the full calibrated camera minus the ROI offset."""
    full = Camera(
        np.array([[3400.0, 0, 2310], [0, 3405, 1290], [0, 0, 1]]), np.zeros(5), (4608, 2592)
    )
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([math.pi + 0.01, 0.02, 0.3]))[0]
    T[:3, 3] = [-100.0, 110.0, 301.0]
    full.T_cam_plate = T
    calib = {
        "camera_matrix": full.K.tolist(),
        "dist_coeffs": [0] * 5,
        "image_size": [4608, 2592],
        "roi": {"x0": 1000, "y0": 16, "w": 2560, "h": 2560},
        "extrinsic": {"T_cam_plate": T.tolist()},
    }
    cam = cammod.from_calibration(calib)
    pts = np.array([[10.0, -20.0, 0.0], [150.0, -100.0, 5.0], [60.0, -60.0, 2.5]])
    assert np.allclose(cam.project(pts), full.project(pts, distort=False) - [1000, 16], atol=1e-9)
    half = cammod.from_calibration(calib, scale=0.5)
    assert np.allclose(half.project(pts), (cam.project(pts) + 0.5) * 0.5 - 0.5, atol=1e-9)


def test_blender_pose_convention():
    """cam2world (Blender: camera looks along -z) maps the optical axis onto the camera's view ray."""
    cam = cammod.nominal()
    M = cam.cam2world_blender(world_scale=10)
    look = M[:3, :3] @ np.array([0, 0, -1.0])
    assert np.allclose(look, [0, 0, -1])  # straight down
    assert np.allclose(M[:3, 3], np.array([110, 110, 300]) * 0.01)
    up_image = M[:3, :3] @ np.array([0, 1.0, 0])  # Blender camera +y = image up = plate +y here
    assert np.allclose(up_image, [0, 1, 0])


# ------------------------------------------------------------------------- composition


def _parts(n_classes=20, lengths=4):
    out = []
    for c in range(n_classes):
        for k in range(lengths if c % 2 == 0 else 1):
            out.append(
                PartInfo(
                    f"C{c:02d}_{k}",
                    f"C{c:02d}",
                    "screw",
                    ("zinc_clear", "black_oxide"),
                    float(k + 1),
                )
            )
    return out


def test_sampler_balances_classes():
    sampler = BalancedSampler(_parts(), slack=1)
    rng = random.Random(0)
    for _ in range(1000):
        part, finish = sampler.draw(rng)
        sampler.record([part.class_name])
    counts = list(sampler.state.counts.values())
    assert sum(counts) == 1000
    assert max(counts) - min(counts) <= 2  # 50 per class, spread <= slack + 1


def test_sampler_recovers_from_imbalance_and_lengths_uniform():
    state = BalanceState(counts={"C00": 300})
    sampler = BalancedSampler(_parts(), state=state)
    rng = random.Random(1)
    drawn = Counter()
    lengths = Counter()
    for _ in range(800):
        part, _ = sampler.draw(rng)
        sampler.record([part.class_name])
        drawn[part.class_name] += 1
        if part.class_name == "C02":
            lengths[part.length] += 1
    assert drawn["C00"] < 10  # the over-represented class is avoided
    assert set(lengths) == {1.0, 2.0, 3.0, 4.0}


def test_new_class_catches_up_inside_mixed_scenes():
    # 20 classes at 800, then C20 is added to the catalogue (count 0)
    state = BalanceState(counts={f"C{c:02d}": 800 for c in range(20)})
    sampler = BalancedSampler(_parts(21), state=state, catch_up_share=0.25)
    assert sampler.behind() == ["C20"]
    cfg = {"few_fraction": 0, "crowded_fraction": 0, "objects": [60, 60]}
    rng = random.Random(3)
    new_per_scene = []
    for _ in range(200):
        plan = plan_scene(sampler, cfg, rng)
        names = Counter(p.class_name for p, _ in plan.parts)
        if sampler.behind():
            new_per_scene.append(names["C20"])
            assert len(names) >= 15  # the other classes stay mixed in
        sampler.record([p.class_name for p, _ in plan.parts])
    assert not sampler.behind()  # caught up
    # about a quarter of every catch-up scene, never alone; ~800 instances at ~15 per scene
    assert min(new_per_scene[:-1]) >= 2 and max(new_per_scene) <= 35
    assert 12 <= sum(new_per_scene) / len(new_per_scene) <= 18
    assert 45 <= len(new_per_scene) <= 75


def test_catch_up_never_triggers_in_a_fresh_run():
    sampler = BalancedSampler(_parts(), slack=1)
    rng = random.Random(5)
    for _ in range(500):
        assert not sampler.behind()
        part, _ = sampler.draw(rng)
        sampler.record([part.class_name])


def test_catch_up_off_keeps_pure_deficit_sampling():
    state = BalanceState(counts={f"C{c:02d}": 800 for c in range(20)})
    sampler = BalancedSampler(_parts(21), state=state, catch_up_share=0)
    assert not sampler.behind()
    rng = random.Random(0)
    assert {sampler.draw(rng)[0].class_name for _ in range(30)} == {"C20"}


def test_balance_state_persists(tmp_path):
    s = BalanceState(counts={"a": 3}, scenes=2)
    s.save(tmp_path / "b.json")
    assert BalanceState.load(tmp_path / "b.json") == s
    assert BalanceState.load(tmp_path / "missing.json") == BalanceState()


def test_plan_scene_kinds_and_determinism():
    cfg = {
        "objects": [15, 40],
        "few_objects": [1, 5],
        "few_fraction": 0.2,
        "crowded_fraction": 0.3,
        "crowded_objects": [30, 50],
    }
    kinds = Counter()
    for i in range(300):
        plan = plan_scene(BalancedSampler(_parts()), cfg, random.Random(i))
        kinds[plan.kind] += 1
        lo, hi = {"few": (1, 5), "crowded": (30, 50), "normal": (15, 40)}[plan.kind]
        assert lo <= len(plan.parts) <= hi
    assert 30 < kinds["few"] < 90 and 60 < kinds["crowded"] < 130
    a = plan_scene(BalancedSampler(_parts()), cfg, random.Random(7))
    b = plan_scene(BalancedSampler(_parts()), cfg, random.Random(7))
    assert [(p.id, f) for p, f in a.parts] == [(p.id, f) for p, f in b.parts]


# ---------------------------------------------------------------------------- placement


def _rect(w, h):
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])


def test_separation():
    a = _rect(2, 2)
    assert separation(a, a + [5, 0]) == pytest.approx(3)
    assert separation(a, a + [1.5, 0]) == pytest.approx(-0.5)
    tri = convex_hull_2d(np.array([[0, 0], [1, 0], [0, 1], [0.2, 0.2]]))
    assert len(tri) == 3


def test_place_no_overlap_inside_area():
    rng = random.Random(0)
    hulls = [_rect(rng.uniform(4, 40), rng.uniform(3, 8)) for _ in range(40)]
    area = (0, 0, 220, 220)
    placed = place(hulls, area, rng, gap_mm=1.0)
    assert len(placed) >= 35
    for i, a in enumerate(placed):
        assert inside_area(a.footprint, area)
        for b in placed[i + 1 :]:
            assert separation(a.footprint, b.footprint) >= 1.0 - 1e-9


def test_place_touching():
    rng = random.Random(3)
    hulls = [_rect(20, 4) for _ in range(20)]
    placed = place(hulls, (0, 0, 120, 120), rng, gap_mm=-0.3, touch_probability=1.0)
    seps = [min(separation(a.footprint, b.footprint) for b in placed if b is not a) for a in placed]
    assert len(placed) >= 15
    assert (
        sum(abs(s + 0.3) < 0.05 for s in seps) >= len(placed) // 2
    )  # most parts touch a neighbour
    assert min(seps) >= -0.3 - 1e-6


def test_discard_rules():
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 2, 2.0]])
    T = np.eye(4)
    T[:3, 3] = [50, 50, 0]
    area = (0, 0, 100, 100)
    assert discard_reason(T, T, verts, area) is None
    lifted = T.copy()
    lifted[2, 3] = 2.0
    assert "stacked" in discard_reason(lifted, T, verts, area)
    moved = T.copy()
    moved[0, 3] += 30
    assert "moved" in discard_reason(moved, T, verts, area)
    off = T.copy()
    off[0, 3] = 150
    assert "left" in discard_reason(off, T, verts, area)
    sunk = T.copy()
    sunk[2, 3] = -5
    assert "fell" in discard_reason(sunk, T, verts, area)


# ----------------------------------------------------------------------- ground truth


def test_keypoint_visibility():
    inst = np.zeros((20, 20), np.uint16)
    inst[5:10, 5:15] = 3
    inst[5:10, 12:15] = 4  # part 4 covers the right end of part 3
    uv = np.array([[7.0, 7.0], [13.0, 7.0], [4.0, 7.0], [30.0, 7.0]])
    assert keypoint_visibility(uv, inst, 3) == [VISIBLE, OCCLUDED, VISIBLE, 0]


def test_silhouette_area_matches_rasterised_box():
    cam = cammod.nominal(scale=0.25)
    box = np.array([[x, y, z] for x in (100, 120) for y in (105, 109) for z in (0, 4.0)])
    total, inside = silhouette_area(cam, [box])
    px = cam.project(box)
    w = px[:, 0].max() - px[:, 0].min()
    h = px[:, 1].max() - px[:, 1].min()
    assert total == pytest.approx(w * h, rel=0.12)
    assert inside == total
    far = box + [400, 0, 0]
    t2, i2 = silhouette_area(cam, [far])
    assert t2 > 0 and i2 == 0


# --------------------------------------------------------------------------- materials


def test_materials_sampled_in_range():
    rng = random.Random(0)
    for finish, preset in DEFAULT_PRESETS.items():
        m = sample_material(finish, None, rng)
        lo, hi = (
            preset["roughness"]
            if isinstance(preset["roughness"], list)
            else (preset["roughness"],) * 2
        )
        assert lo <= m["roughness"] <= hi
        assert all(0 <= c <= 1 for c in m["base_color"])
    override = sample_material("brass", {"brass": {"base_color": [1, 0, 0], "roughness": 0.5}}, rng)
    assert override["roughness"] == 0.5 and override["base_color"][0] > 0.9
    with pytest.raises(KeyError):
        sample_material("anodised_red", None, rng)


def test_generate_script_imports_blenderproc_first():
    """`blenderproc run` refuses scripts whose first non-comment line is not the blenderproc import."""
    text = (repo_root() / "tools" / "scenegen" / "generate.py").read_text(encoding="utf-8")
    lines = [s.strip() for s in text.splitlines()]
    first = next(s for s in lines if s and not s.startswith("#"))
    assert first.startswith("import blenderproc")


def test_insert_colour_weights():
    rng = random.Random(0)
    picks = Counter(pick_insert_finish(None, rng) for _ in range(2000))
    assert set(picks) == {"insert_blue", "insert_white"}
    assert 0.65 < picks["insert_blue"] / 2000 < 0.75
    assert {pick_insert_finish({"insert_white": 1, "insert_blue": 0}, rng) for _ in range(50)} == {
        "insert_white"
    }
    a = [pick_insert_finish(None, random.Random(5)) for _ in range(3)]
    assert a == [pick_insert_finish(None, random.Random(5)) for _ in range(3)]
    with pytest.raises(ValueError):
        pick_insert_finish({"insert_blue": 0}, rng)


# ------------------------------------------------------------------------- Blender smoke


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("blenderproc") is None, reason="blenderproc not installed")
def test_blender_smoke(tmp_path):
    """One tiny scene with the nominal camera; instance ids in meta.json match instance.png."""
    import json

    cfg = tmp_path / "config.yaml"
    base = (repo_root() / "configs" / "config.yaml").read_text(encoding="utf-8")
    cfg.write_text(
        # replace the key itself: a second output_dir would lose (in YAML the last one wins)
        base.replace("  output_dir: output\n", f"  output_dir: {tmp_path.as_posix()}\n", 1),
        encoding="utf-8",
    )
    r = subprocess.run(
        [
            "blenderproc",
            "run",
            str(repo_root() / "tools/scenegen/generate.py"),
            "--run",
            "smoke",
            "--preview",
            "--nominal-camera",
            "--config",
            str(cfg),
        ],
        cwd=repo_root(),
        timeout=1800,
    )
    assert r.returncode == 0
    scene = sorted((tmp_path / "smoke" / "scenes").iterdir())[0]
    meta = json.loads((scene / "meta.json").read_text())
    mask = read_png(scene / "instance.png")
    ids = {o["instance_id"] for o in meta["objects"]}
    assert set(np.unique(mask)) - {0} <= ids


# ----------------------------------------------------------------------- clumped layouts


def _mixed_hulls(n, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.55:
            d = rng.choice([2, 3, 4, 5, 6])
            out.append(_rect(rng.choice([8, 12, 16, 20, 30]) + d, d))
        else:
            a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            r = rng.choice([2.5, 3.5, 4.5, 6.0])
            out.append(convex_hull_2d(np.c_[r * np.cos(a), r * np.sin(a)]))
    return out


AREA, BOUNDS = (3.0, 3.0, 217.0, 217.0), (-2.0, -2.0, 222.0, 222.0)


def test_clumped_layers_and_bounds():
    placed = place_clumped(_mixed_hulls(80), AREA, BOUNDS, random.Random(3))
    assert len(placed) >= 70
    layer1 = [p for p in placed if p.layer == 1]
    layer2 = [p for p in placed if p.layer == 2]
    assert layer2, "some parts land on top of others"
    for i, a in enumerate(layer1):  # layer 1: never overlapping (touching allowed)
        for b in layer1[i + 1 :]:
            assert separation(a.footprint, b.footprint) >= -1e-6
    for i, a in enumerate(layer2):  # at most two layers: tops never overlap each other
        assert any(separation(a.footprint, b.footprint) < 0 for b in layer1)
        for b in layer2[i + 1 :]:
            assert separation(a.footprint, b.footprint) >= -1e-6
    for p in placed:
        assert inside_area(p.footprint, BOUNDS)
        c = p.footprint.mean(axis=0)
        assert AREA[0] <= c[0] <= AREA[2] and AREA[1] <= c[1] <= AREA[3]


def test_clumped_is_denser_than_scattered_and_deterministic():
    hulls = _mixed_hulls(60)
    clumped = place_clumped(hulls, AREA, BOUNDS, random.Random(5))
    scattered = place(hulls, AREA, random.Random(5), 1.0)
    g_clump = nearest_gaps([p for p in clumped if p.layer == 1])
    g_scat = nearest_gaps(scattered)
    # in clumps the nearest neighbour is under 1 mm away (often touching), scattered ~5 mm
    assert np.median(g_clump) < 1.0 and np.median(g_scat) > 4 * np.median(g_clump)
    assert np.mean(g_clump < 0.3) > 0.1
    again = place_clumped(hulls, AREA, BOUNDS, random.Random(5))
    assert [(p.index, p.xy, p.layer) for p in again] == [(p.index, p.xy, p.layer) for p in clumped]


def test_clump_settings_from_config():
    st = ClumpSettings.from_config({"clumps": [2, 2], "spread_mm": [5, 5], "loose_fraction": 0})
    assert st.clumps == (2, 2) and st.spread_mm == (5.0, 5.0) and st.loose_fraction == 0.0
    assert st.overlap_probability == ClumpSettings().overlap_probability


def test_plan_scene_clumped_kind():
    cfg = {
        "clumped_fraction": 1.0,
        "few_fraction": 0,
        "crowded_fraction": 0,
        "clumped_objects": [40, 60],
    }
    plan = plan_scene(BalancedSampler(_parts()), cfg, random.Random(1))
    assert plan.kind == "clumped" and 40 <= len(plan.parts) <= 60


def test_occluders_free_and_measurable():
    cam = cammod.nominal(scale=0.25)
    lower = np.array([[x, y, z] for x in (100, 130) for y in (100, 104) for z in (0, 4.0)])
    sil = silhouette_mask(cam, [lower])
    total, inside = silhouette_area(cam, [lower])
    instance = np.zeros((cam.resolution[1], cam.resolution[0]), np.uint16)
    mask, x0, y0 = sil
    ys, xs = np.nonzero(mask)
    instance[ys + y0, xs + x0] = 1
    # part 2 lies across the middle of part 1
    instance[ys.min() + y0 : ys.max() + y0 + 1, xs.min() + x0 + 10 : xs.min() + x0 + 16] = 2
    assert occluders(sil, instance, 1, min_px=5) == {2: int((instance == 2).sum())}
    assert occluders(sil, instance, 1, min_px=10_000) == {}  # a sliver below the threshold
    part = {
        "id": "DIN912_M4x30",
        "class_name": "DIN912_M4",
        "kind": "screw",
        "family": "DIN912",
        "size": "M4",
        "length": 30,
        # on the part, clear of part 2 (which covers x ~ 104-106 or ~124-126, by image orientation)
        "keypoints": {
            "head_top": {"xyz": [120.0, 102.0, 2.0], "labelled": True},
            "bearing": {"xyz": [117.0, 102.0, 2.0], "labelled": True},
            "tip": {"xyz": [110.0, 102.0, 2.0], "labelled": True},
        },
    }
    common = {
        "inst_id": 1,
        "part": part,
        "class_id": 0,
        "finish": "zinc_clear",
        "material": {},
        "T_plate": np.eye(4) @ np.diag([1, 1, 1, 1]),
        "rest_pose_index": 0,
        "measurable_tilt_deg": 90,
        "cam": cam,
        "alone_area": (total, inside),
        "silhouette": sil,
        "occlusion_min_px": 5,
    }
    covered = object_meta(instance=instance, **common)
    assert covered["occluded_by"] == [2] and not covered["free"] and not covered["measurable"]
    clear = object_meta(instance=(instance == 1).astype(np.uint16), **common)
    assert clear["free"] and clear["measurable"] and clear["support"] == "plate"
    assert 0 <= clear["view_angle_deg"] < 30
    # the same part seen from any other camera view: never measured (needs the rig view)
    other_view = object_meta(instance=(instance == 1).astype(np.uint16), rig_view=False, **common)
    assert other_view["free"] and not other_view["measurable"]
    on_parts = object_meta(instance=(instance == 1).astype(np.uint16), lowest_z_mm=3.0, **common)
    assert on_parts["support"] == "parts" and not on_parts["measurable"]
    # physically covered (from the 3D poses) is not free, even if nothing shows over it
    held = object_meta(instance=(instance == 1).astype(np.uint16), lying_on_it=[7], **common)
    assert held["lying_on_it"] == [7] and not held["free"] and not held["measurable"]
    # a neighbour only leaning over it in the image does not make it un-free
    leaning = object_meta(instance=instance, lying_on_it=[], **common)
    assert leaning["free"] and leaning["occluded_by"] == [2]


def test_discard_allows_stacking():
    verts = np.array([[x, y, z] for x in (0, 10) for y in (0, 4) for z in (0, 4)], float)
    T = np.eye(4)
    T[:3, 3] = [100, 100, 3.0]  # resting 3 mm up, on another part
    assert "stacked" in discard_reason(T, T, verts, AREA)
    assert discard_reason(T, T, verts, AREA, allow_stacked=True) is None


def test_clumped_multilayer_stacks():
    hulls = _mixed_hulls(90)
    heights = [3.0 + (i % 4) for i in range(len(hulls))]  # every part low enough to carry others
    st = ClumpSettings(overlap_probability=(0.6, 0.6), max_layers=3)
    placed = place_clumped(hulls, AREA, BOUNDS, random.Random(2), st, heights=heights)
    by_index = {p.index: p for p in placed}
    layers = Counter(p.layer for p in placed)
    assert layers[3] > 0 and max(layers) == 3
    top = {}
    for p in placed:  # placement order: parts below are always placed first
        if p.layer == 1:
            assert p.base_z == 0.0
        else:
            below = [
                q
                for q in placed
                if q is not p and q.index in top and separation(p.footprint, q.footprint) < -0.5
            ]
            assert max(q.layer for q in below) == p.layer - 1
            assert p.base_z == pytest.approx(max(top[q.index] for q in below))
        top[p.index] = p.base_z + heights[p.index]
    assert all(by_index[i].layer >= 1 for i in by_index)


def test_clumped_single_layer_when_max_layers_is_one():
    st = ClumpSettings(overlap_probability=(0.9, 0.9), max_layers=1)
    placed = place_clumped(_mixed_hulls(60), AREA, BOUNDS, random.Random(4), st)
    assert {p.layer for p in placed} == {1}
