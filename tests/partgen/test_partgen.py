import math

import numpy as np
import pytest
import trimesh

from screwsynth.catalogue.schema import load_catalogue, parse_catalogue
from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, read_yaml
from screwsynth.common.paths import repo_root
from screwsynth.partgen import assets, checks, collision
from screwsynth.partgen.builders import Lod, build, rod_end_styles
from screwsynth.partgen.cli import contact_sheet_parts, main
from screwsynth.partgen.keypoints import nominal_length
from screwsynth.partgen.profiles import (
    EXT_DEPTH,
    EXT_ROOT,
    RodSpec,
    rod_mesh,
    thread_depth_fraction,
)

LOW = Lod(thread_segments=24, axial_per_pitch=8, round_segments=32)

# a sample from every family, including extremes (smallest/largest, partially threaded)
SAMPLE = [
    "DIN912_M2x3",
    "DIN912_M4x16",
    "DIN912_M10x80",  # partially threaded
    "DIN933_M2x4",
    "DIN933_M8x40",
    "ISO7380_M3x6",
    "ISO7380_M10x30",
    "DIN7991_M3x10",
    "DIN7991_M8x60",  # partially threaded
    "DIN934_M2",
    "DIN934_M10",
    "DIN985_M3",
    "DIN985_M10",
    "DIN125A_M2",
    "DIN9021_M10",
    "TNUT_SLIDE_S6_M3",
    "TNUT_SLIDE_S8_M6",
    "TNUT_DROP_S6_M5",
    "TNUT_DROP_S8_M4",
    "DIN913_M2x2",
    "DIN913_M10x30",
    "DIN975_M2x12",
    "DIN975_M8x50",
    "STANDOFF_MF_M2x3",  # shortest: the female thread stops short of the stud
    "STANDOFF_MF_M3x10",
    "STANDOFF_FF_M2.5x3",  # tapped through
    "STANDOFF_FF_M5x30",
    "STANDOFF_MM_M3x10",
    "BRACKET_2020",
    "BRACKET_4040",
]


@pytest.fixture(scope="module")
def catalogue():
    root = repo_root()
    return load_catalogue(root / "configs" / "catalogue.yaml", max_length_mm=80)


@pytest.fixture(scope="module")
def built(catalogue):
    return {pid: build(catalogue.part(pid), LOW) for pid in SAMPLE}


# ---------------------------------------------------------------------------------- profiles


def test_thread_profile_shape():
    u = np.linspace(0, 1, 1001, endpoint=False)
    f = thread_depth_fraction(u, 1 / 8, EXT_ROOT)
    assert f.min() == 0 and f.max() == 1
    assert np.all(f[u < 1 / 8] == 0)
    # 60 deg flanks: the descending flank spans depth * tan(30) of the pitch
    flank = (1 - 1 / 8 - EXT_ROOT) / 2
    assert flank == pytest.approx(EXT_DEPTH * math.tan(math.radians(30)))


def test_rod_mesh_is_closed_and_hits_major_diameter():
    p = 0.7
    spec = RodSpec(radius=2.0, pitch=p, z0=-10, z1=0, depth=EXT_DEPTH * p, tip_radius=1.5)
    v, f = rod_mesh(spec, 24, 8)
    mesh = trimesh.Trimesh(v, f, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    r = np.hypot(v[:-2, 0], v[:-2, 1])
    assert r.max() == pytest.approx(2.0)
    assert r[v[:-2, 2] > -8].min() == pytest.approx(2.0 - EXT_DEPTH * p, abs=1e-9)


def test_rod_runout_reaches_plain_radius():
    spec = RodSpec(radius=3, pitch=0.5, z0=-20, z1=0, depth=0.3, thread_end=-10, runout=1.0)
    v, _ = rod_mesh(spec, 24, 8)
    upper = v[:-2][v[:-2, 2] > -9]
    assert np.hypot(upper[:, 0], upper[:, 1]) == pytest.approx(3.0)


# ----------------------------------------------------------------------------------- builders


@pytest.mark.parametrize("pid", SAMPLE)
def test_part_geometry(catalogue, built, pid):
    part = catalogue.part(pid)
    b = built[pid]
    problems = checks.check_mesh(b.mesh, 200_000)
    problems += checks.check_dimensions(part, b.mesh, 0.01)
    problems += checks.check_keypoints(part, b.keypoints, 0.01)
    assert problems == []


@pytest.mark.parametrize("pid", SAMPLE)
def test_collision_proxy(built, pid):
    b = built[pid]
    assert collision.total_triangles(b.collision) < 500
    assert all(p.is_convex for p in b.collision)
    assert collision.contained_fraction(b.mesh, b.collision) >= 0.95


def test_keypoints(catalogue, built):
    for pid in ("DIN912_M4x16", "DIN933_M8x40", "ISO7380_M3x6"):
        part, b = catalogue.part(pid), built[pid]
        assert b.labelled
        assert b.keypoints["bearing"] == [0, 0, 0]
        assert nominal_length(part, b.keypoints) == pytest.approx(part.length)
        assert b.keypoints["head_top"][2] == pytest.approx(part.dims["k"])
    kp = built["DIN7991_M3x10"].keypoints
    assert kp["head_top"][2] - kp["tip"][2] == pytest.approx(10)  # head included
    assert not built["DIN934_M2"].labelled


def test_partial_thread(catalogue, built):
    """DIN912 M10x80 is partially threaded: the upper shank is plain at radius d/2."""
    part = catalogue.part("DIN912_M10x80")
    assert part.fully_threaded is False
    v = built["DIN912_M10x80"].mesh.vertices
    tip = -80
    plain = v[(v[:, 2] > tip + part.dims["b"] + 4) & (v[:, 2] < -1)]
    threaded = v[(v[:, 2] > tip + 5) & (v[:, 2] < tip + part.dims["b"] - 4)]
    assert np.hypot(plain[:, 0], plain[:, 1]).min() == pytest.approx(5.0, abs=1e-3)
    assert np.hypot(threaded[:, 0], threaded[:, 1]).min() < 5.0 - 0.5


def test_nyloc_has_insert_slot(built):
    b = built["DIN985_M10"]
    assert set(b.slots) == {"body", "insert"}
    a, z = b.slots["insert"]
    assert 0 < z - a < len(b.mesh.faces)
    assert b.slots["body"] == (0, a)  # faces sorted by slot
    insert_z = b.mesh.triangles_center[a:z, 2]
    assert insert_z.min() > b.mesh.bounds[0][2] + 5  # insert sits in the upper collar


def test_hex_head_orientation(built):
    """Flats are perpendicular to y: max |y| equals s/2, corners lie on x."""
    v = built["DIN933_M8x40"].mesh.vertices
    head = v[v[:, 2] > 0.5]
    assert np.abs(head[:, 1]).max() == pytest.approx(6.5, abs=1e-3)
    assert np.abs(head[:, 0]).max() > 7.0


def test_triangle_budget(catalogue):
    part = catalogue.part("DIN912_M10x80")
    high = Lod(64, 16, 96)
    assert len(build(part, high, max_triangles=200_000).mesh.faces) < 200_000
    small = build(part, high, max_triangles=40_000)
    assert len(small.mesh.faces) < 40_000


# ------------------------------------------------------------------------------------- assets


def _settings():
    s = assets.Settings.from_config(load_config(), "low")
    return assets.Settings(s.lod_name, s.lod, s.max_triangles, s.collision_segments, 0.01, 120)


def test_build_asset_files_and_incremental(catalogue, tmp_path):
    part = catalogue.part("DIN912_M3x8")
    st = _settings()
    r = assets.build_asset(part, tmp_path, st)
    assert r.status == "built", r
    folder = tmp_path / part.id
    for f in ("render.ply", "collision.obj", "part.json", "preview.png"):
        assert (folder / f).is_file()
    meta = read_json(folder / "part.json")
    assert meta["catalogue_hash"] == part.hash and meta["checks"]["passed"]
    assert meta["nominal_length_mm"] == pytest.approx(8)
    assert list(meta["keypoints"]) == ["head_top", "bearing", "tip"]
    assert meta["mesh"]["watertight"] and meta["mesh"]["material_slots"]["body"][0] == 0
    assert meta["axis_height_when_lying"]["head"] == pytest.approx(part.dims["dk"] / 2)

    mesh = trimesh.load_mesh(folder / "render.ply", process=False)
    assert mesh.is_watertight and len(mesh.faces) == meta["mesh"]["triangles"]
    pieces = collision.read_obj(folder / "collision.obj")
    assert len(pieces) == meta["collision"]["pieces"] and all(p.is_convex for p in pieces)

    assert assets.build_asset(part, tmp_path, st).status == "skipped"
    assert assets.build_asset(part, tmp_path, st, force=True).status == "built"
    assert assets.recheck_asset(part, tmp_path, st).status == "built"


def test_catalogue_change_triggers_rebuild(tmp_path):
    data = read_yaml(repo_root() / "configs" / "catalogue.yaml")
    st = _settings()
    part = parse_catalogue(data).part("DIN934_M4")
    assert assets.build_asset(part, tmp_path, st).status == "built"
    data["families"]["DIN934"]["sizes"]["M4"]["dims"]["m"] = 3.0
    changed = parse_catalogue(data).part("DIN934_M4")
    assert not assets.is_up_to_date(changed, tmp_path / part.id, st)
    assert assets.recheck_asset(changed, tmp_path, st).status == "failed"
    assert assets.build_asset(changed, tmp_path, st).status == "built"


def test_cli_build_check_clean(tmp_path, capsys):
    args = ["--assets-dir", str(tmp_path)]
    assert (
        main(
            [
                "build",
                "--id",
                "DIN125A_M3",
                "TNUT_DROP_S6_M4",
                "--lod",
                "low",
                "--workers",
                "1",
                *args,
            ]
        )
        == 0
    )
    assert "built: 2" in capsys.readouterr().out
    assert main(["build", "--id", "DIN125A_M3", "--lod", "low", *args]) == 0
    assert "skipped: 1" in capsys.readouterr().out
    assert main(["check", "--id", "DIN125A_M3", *args]) == 0
    assert main(["preview", "--id", "DIN125A_M3", *args]) == 0
    assert main(["clean", *args]) == 0
    assert not any(tmp_path.iterdir())


def test_contact_sheet_parts(catalogue):
    chosen = contact_sheet_parts(catalogue)
    real = [p for p in chosen if p.kind != "unknown"]
    assert len(real) == len(catalogue.families)
    assert {p.family for p in real} == set(catalogue.families)
    assert all(p.size == "M4" for p in real if p.kind != "bracket")  # brackets: 2020...
    junk = [p for p in chosen if p.kind == "unknown"]  # one of each unknown type
    assert len(junk) == len({p.shape for p in catalogue.parts if p.kind == "unknown"})


# ---------------------------------------------------------------------------------------- STEP


def test_step_route_with_preconverted_stl(tmp_path, monkeypatch):
    """A STEP part without cadquery falls back to an STL next to it and is aligned by keypoints."""
    head = trimesh.creation.cylinder(radius=3.5, height=4, sections=32)
    head.apply_translation([0, 0, 2])
    shank = trimesh.creation.cylinder(radius=2, height=12, sections=32)
    shank.apply_translation([0, 0, -6])
    m = trimesh.boolean.union([head, shank], engine="manifold")
    # put it in an arbitrary CAD frame: axis along +x, offset, in metres
    T = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
    T[:3, 3] = [10, 5, 0]
    m.apply_transform(T)
    m.apply_scale(0.001)
    (tmp_path / "screw.step").write_text("dummy", encoding="utf-8")
    m.export(tmp_path / "screw.stl")
    kp_local = {"head_top": [0, 0, 4], "bearing": [0, 0, 0], "tip": [0, 0, -12]}
    kp_src = {
        k: ((T[:3, :3] @ np.array(v) + T[:3, 3]) * 0.001).tolist() for k, v in kp_local.items()
    }

    data = read_yaml(repo_root() / "tools" / "catalogue" / "templates" / "family_template.yaml")
    fam = data["families"]["EXAMPLESTEP"]
    fam["enabled"] = True
    fam["sizes"]["M4"]["step"][12] = {
        "path": str(tmp_path / "screw.step"),
        "units": "m",
        "keypoints": kp_src,
    }
    cat = parse_catalogue(
        {"finishes": {"zinc_clear": {"label": "z"}}, "families": data["families"]}
    )
    b = build(cat.part("EXAMPLESTEP_M4x12"), LOW)
    lo, hi = b.mesh.bounds
    assert lo[2] == pytest.approx(-12, abs=1e-3) and hi[2] == pytest.approx(4, abs=1e-3)
    assert b.keypoints["bearing"] == pytest.approx([0, 0, 0], abs=1e-6)
    assert np.hypot(*b.mesh.vertices[:, :2].T).max() == pytest.approx(3.5, abs=1e-3)
    assert collision.contained_fraction(b.mesh, b.collision) > 0.95


def test_step_without_converter_explains(tmp_path):
    from screwsynth.partgen.builders import BuildError
    from screwsynth.partgen.step import load_source_mesh

    (tmp_path / "x.step").write_text("dummy", encoding="utf-8")
    with pytest.raises(BuildError, match="cadquery|STL"):
        load_source_mesh(tmp_path / "x.step")


def test_headless_and_standoff_keypoints(catalogue, built):
    """No head: head_top = bearing at z = 0, the nominal length is the overall (body) length."""
    for pid in ("DIN913_M10x30", "DIN975_M8x50", "STANDOFF_MF_M3x10", "STANDOFF_MM_M3x10"):
        part, b = catalogue.part(pid), built[pid]
        assert b.labelled
        assert b.keypoints["head_top"] == b.keypoints["bearing"] == [0, 0, 0]
        assert nominal_length(part, b.keypoints) == pytest.approx(part.length)
    assert not built["BRACKET_2020"].labelled


def test_set_screw_socket_and_rod_ends(built):
    v = built["DIN913_M10x30"].mesh.vertices
    top = v[v[:, 2] > -0.01]
    assert np.hypot(top[:, 0], top[:, 1]).min() < 3.0  # socket corners (s = 5) in the top face
    for pid in ("DIN975_M2x12", "DIN975_M8x50"):
        b, L = built[pid], float(pid.split("x")[-1])
        rod = b.mesh.vertices
        tip_style, top_style = rod_end_styles(pid)
        assert b.info["end_styles"] == [tip_style, top_style]
        for z, style in ((-L, tip_style), (0.0, top_style)):
            end = rod[np.abs(rod[:, 2] - z) < 0.01]
            r = np.hypot(end[:, 0], end[:, 1]).max()
            d = 2 if "M2x" in pid else 8
            if style == "chamfer":
                assert r < d / 2 - 0.1  # chamfered below the root
            else:
                assert r == pytest.approx(d / 2, abs=1e-3)  # sawn off: thread up to the edge


def test_rod_end_styles_vary():
    from collections import Counter

    cat = load_catalogue(repo_root() / "configs" / "catalogue.yaml", max_length_mm=80)
    styles = Counter(rod_end_styles(p.id) for p in cat.parts if p.family == "DIN975")
    assert len(styles) == 4 and min(styles.values()) >= 5  # every combination, none rare
    assert rod_end_styles("DIN975_M5x40") == rod_end_styles("DIN975_M5x40")  # deterministic


def test_standoff_studs(catalogue, built):
    for pid, studs in (
        ("STANDOFF_MF_M3x10", 1),
        ("STANDOFF_FF_M5x30", 0),
        ("STANDOFF_MM_M3x10", 2),
    ):
        b, d = built[pid], catalogue.part(pid).dims
        lo, hi = b.mesh.bounds
        assert (hi[2] - lo[2]) == pytest.approx(
            catalogue.part(pid).length + studs * d["ls"], abs=1e-3
        )
        assert len(b.collision) == 1 + studs
