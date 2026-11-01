"""The unknown class: junk parts the machine rejects (naming, catalogue, meshes, sampling)."""

import copy
import random
from collections import Counter

import pytest

from screwsynth.catalogue.schema import UNKNOWN_TYPES, CatalogueError, parse_catalogue
from screwsynth.common.io import read_yaml
from screwsynth.common.naming import UNKNOWN, class_name, parse_part_id, unknown_id
from screwsynth.common.paths import repo_root

SHIPPED = repo_root() / "configs" / "catalogue.yaml"


@pytest.fixture(scope="module")
def shipped_data():
    return read_yaml(SHIPPED)


def test_unknown_ids():
    pid = unknown_id("wood_screw", 3)
    assert pid == "UNKNOWN_WOODSCREW_03"
    parsed = parse_part_id(pid)
    assert parsed.is_unknown and parsed.part_id == pid and parsed.family == UNKNOWN
    assert class_name(pid) == UNKNOWN
    assert class_name("DIN912_M4x16") == "DIN912_M4"  # real parts unchanged
    with pytest.raises(ValueError):
        unknown_id("3d", 1)


def test_unknown_parts_are_deterministic_and_last_class(shipped_data):
    a = parse_catalogue(copy.deepcopy(shipped_data))
    b = parse_catalogue(copy.deepcopy(shipped_data))
    ua = [p for p in a.parts if p.kind == "unknown"]
    ub = [p for p in b.parts if p.kind == "unknown"]
    spec = a.unknown
    assert len(ua) == spec.variants * len(spec.types)
    assert [(p.id, dict(p.dims)) for p in ua] == [(p.id, dict(p.dims)) for p in ub]
    assert {p.class_name for p in ua} == {UNKNOWN}
    assert a.classes()[-1] == UNKNOWN  # sorted last: the real classes keep their ids
    for p in ua:
        name = p.shape.removeprefix("unknown_")
        for key, (lo, hi) in spec.types[name].items():
            assert lo <= p.dims[key] <= hi
        assert p.finishes


def test_unknown_rules_and_switches(shipped_data):
    data = copy.deepcopy(shipped_data)
    data["rules"] = [{"match": "UNKNOWN", "enabled": False}]
    assert UNKNOWN not in parse_catalogue(data).classes()
    data["rules"] = [{"match": "UNKNOWN_BLOB_*", "enabled": False}]
    cat = parse_catalogue(data)
    assert not any(p.enabled for p in cat.parts if p.shape == "unknown_blob")
    assert any(p.enabled for p in cat.parts if p.shape == "unknown_pin")
    data = copy.deepcopy(shipped_data)
    data["unknown"]["enabled"] = False
    assert UNKNOWN not in parse_catalogue(data).classes()
    data = copy.deepcopy(shipped_data)
    del data["unknown"]
    assert not any(p.kind == "unknown" for p in parse_catalogue(data).parts)


@pytest.mark.parametrize(
    "change",
    [
        {"types": {"spaceship": {}}},
        {"types": {"pin": {"colour": [1, 2]}}},
        {"types": {"pin": {"d": [5, 2]}}},
        {"finishes": ["anodised_red"]},
        {"variants": 0},
    ],
)
def test_unknown_section_validation(shipped_data, change):
    data = copy.deepcopy(shipped_data)
    data["unknown"].update(change)
    with pytest.raises(CatalogueError):
        parse_catalogue(data)


def test_every_unknown_type_builds(shipped_data):
    pytest.importorskip("manifold3d")
    from screwsynth.partgen.builders import Lod, build
    from screwsynth.partgen.checks import check_collision, check_mesh

    data = copy.deepcopy(shipped_data)
    for t in UNKNOWN_TYPES:  # also the types the shipped catalogue leaves out (rod, standoff)
        data["unknown"]["types"].setdefault(t, {})
    cat = parse_catalogue(data)
    lod = Lod(24, 8, 32)
    first = {}
    for p in cat.parts:
        if p.kind == "unknown":
            first.setdefault(p.shape, p)
    assert set(first) == {f"unknown_{t}" for t in UNKNOWN_TYPES}
    for p in first.values():
        built = build(p, lod)
        assert built.mesh.is_watertight, p.id
        assert check_mesh(built.mesh, 200_000) == [], p.id
        assert check_collision(built.mesh, built.collision) == [], p.id
        assert not built.labelled  # no keypoints for junk


def test_sampler_unknown_fraction():
    from screwsynth.scenegen.composition import BalancedSampler, PartInfo

    real = [PartInfo(f"C{c}_M4", f"C{c}_M4", "screw", ("zinc_clear",), None) for c in range(10)]
    junk = [PartInfo(f"UNKNOWN_PIN_0{i}", UNKNOWN, "unknown", ("brass",), None) for i in range(3)]
    sampler = BalancedSampler(real + junk, unknown_fraction=0.2)
    rng = random.Random(0)
    drawn = Counter()
    for _ in range(3000):
        part, _ = sampler.draw(rng)
        drawn[part.class_name] += 1
        sampler.record([part.class_name])
    assert 0.17 < drawn[UNKNOWN] / 3000 < 0.23
    counts = [drawn[f"C{c}_M4"] for c in range(10)]
    assert max(counts) - min(counts) <= 2  # the real classes stay balanced
    assert UNKNOWN not in sampler.classes
    none = BalancedSampler(real + junk)  # default: no junk unless asked for
    assert all(none.draw(rng)[0].class_name != UNKNOWN for _ in range(200))
