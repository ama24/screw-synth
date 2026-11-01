import pytest

from conftest import SHIPPED
from screwsynth.catalogue.schema import CatalogueError, load_catalogue, parse_catalogue
from screwsynth.catalogue.validate import errors, validate
from screwsynth.common.config import load_config
from screwsynth.common.naming import parse_part_id, size_value


@pytest.fixture(scope="module")
def shipped():
    return load_catalogue(SHIPPED, max_length_mm=load_config().catalogue.max_length_mm)


def test_shipped_catalogue_validates(shipped):
    issues = validate(shipped)
    assert errors(issues) == [], "\n".join(map(str, issues))
    assert len(shipped.parts) > 300


def test_every_enabled_class_has_an_enabled_part(shipped):
    for cname in shipped.classes():
        assert any(p.enabled and p.class_name == cname for p in shipped.parts), cname


def test_shipped_ids_follow_naming(shipped):
    for p in shipped.parts:
        assert parse_part_id(p.id).class_name == p.class_name


def test_shipped_scope(shipped):
    fams = shipped.families
    assert {
        "DIN912",
        "DIN933",
        "ISO7380",
        "DIN7991",
        "DIN934",
        "DIN985",
        "DIN125A",
        "DIN9021",
    } <= set(fams)
    assert fams["DIN985"].enabled and fams["DIN9021"].enabled  # user decision 2026-10-28
    assert fams["DIN7991"].nominal_length == "head_top_to_tip"
    assert "TNUT_DROP_S6_M4" in shipped.classes()
    assert "TNUT_SLIDE_S8_M6" in shipped.classes()
    assert all(
        f.dims_source == "vendor-typical, measure your own"
        for n, f in fams.items()
        if n.startswith("TNUT")
    )
    assert not any(p.verified for p in shipped.parts)
    # aluminium only natural or black anodised (user decision 2026-11-01), no coloured anodising
    assert {f for f in shipped.finishes if "aluminium" in f} == {"aluminium", "aluminium_black"}
    assert max(p.length for p in shipped.parts if p.length) <= 80


def test_max_length_is_a_parameter():
    short = load_catalogue(SHIPPED, max_length_mm=20)
    unlimited = load_catalogue(SHIPPED, max_length_mm=None)
    assert max(p.length for p in short.parts if p.length) == 20
    assert max(p.length for p in unlimited.parts if p.length) == 100
    assert len(unlimited.parts) > len(short.parts)


def test_expand_minimal(minimal_data):
    cat = parse_catalogue(minimal_data)
    ids = [p.id for p in cat.parts]
    assert ids == [
        "DIN912_M3x6",
        "DIN912_M3x10",
        "DIN912_M3x30",
        "DIN912_M4x8",
        "DIN912_M4x16",
        "DIN934_M3",
        "DIN125A_M3",
        "TNUT_DROP_S6_M4",
    ]
    p = cat.part("DIN912_M3x30")
    assert not p.common
    assert p.fully_threaded is False
    assert cat.part("DIN912_M3x10").fully_threaded is True
    assert p.dims["d"] == 3 and p.dims["ds"] == 3
    assert p.finishes == ("zinc_clear", "nylon_white")
    assert cat.part("DIN912_M4x8").finishes == ("zinc_clear",)
    assert cat.part("TNUT_DROP_S6_M4").slot == 6
    assert "d" not in cat.part("DIN125A_M3").dims


def test_class_order_is_numeric(minimal_data):
    minimal_data["families"]["DIN912"]["sizes"]["M10"] = {
        "pitch": 1.5,
        "dims": {"dk": 16, "k": 10, "s": 8, "t": 5, "b": 32},
        "lengths": [20],
    }
    cat = parse_catalogue(minimal_data)
    assert [c for c in cat.classes() if c.startswith("DIN912")] == [
        "DIN912_M3",
        "DIN912_M4",
        "DIN912_M10",
    ]


def test_rules_last_match_wins(minimal_data):
    minimal_data["rules"] = [
        {"match": "DIN912", "enabled": False},
        {"match": "DIN912_M4x16", "enabled": True},
    ]
    cat = parse_catalogue(minimal_data)
    enabled = {p.id for p in cat.enabled_parts if p.family == "DIN912"}
    assert enabled == {"DIN912_M4x16"}
    minimal_data["rules"] = [{"match": "M3", "enabled": False}]
    cat = parse_catalogue(minimal_data)
    assert all(p.size != "M3" for p in cat.enabled_parts)


def test_part_hash_tracks_geometry_only(minimal_data):
    a = parse_catalogue(minimal_data).part("DIN912_M3x10").hash
    minimal_data["families"]["DIN912"]["finishes"]["zinc_clear"] = ["M4"]
    assert parse_catalogue(minimal_data).part("DIN912_M3x10").hash == a
    minimal_data["families"]["DIN912"]["sizes"]["M3"]["dims"]["dk"] = 5.6
    assert parse_catalogue(minimal_data).part("DIN912_M3x10").hash != a


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update(colour="red"), "unknown key"),
        (lambda d: d["families"]["DIN912"].pop("kind"), "missing required key"),
        (lambda d: d["families"]["DIN912"].update(kind="bolt"), "kind"),
        (lambda d: d["families"]["DIN912"].update(shape="torx"), "shape"),
        (
            lambda d: d["families"]["DIN912"]["sizes"].update(M5x={"pitch": 0.8}),
            "invalid thread size",
        ),
        (
            lambda d: d["families"]["DIN912"]["sizes"].update({"m5": {"pitch": 0.8}}),
            "write the size",
        ),
        (lambda d: d["families"]["TNUT_DROP"]["sizes"].update(M4={"pitch": 0.7}), "S6_M4"),
        (lambda d: d["families"]["DIN912"]["sizes"]["M3"]["dims"].update(dk="big"), "number"),
        (lambda d: d["families"]["DIN912"]["finishes"].update(zinc_clear="M3"), "'all' or a list"),
        (lambda d: d.update(rules=[{"match": "X"}]), "match"),
        (lambda d: d["families"].update(din999=d["families"].pop("DIN934")), "family names"),
        (lambda d: d["families"].update(TNUT=d["families"].pop("TNUT_DROP")), "family names"),
        (lambda d: d["families"]["TNUT_DROP"].pop("tnut_kind"), "tnut_kind"),
        (lambda d: d["families"]["DIN912"].update(tnut_kind="DROP"), "tnut_kind"),
        (lambda d: d["families"]["TNUT_DROP"]["sizes"].update(S0_M4={"pitch": 0.7}), "S6_M4"),
    ],
)
def test_structural_errors(minimal_data, mutate, match):
    mutate(minimal_data)
    with pytest.raises(CatalogueError, match=match):
        parse_catalogue(minimal_data)


def test_shipped_expansion_2026_11(shipped):
    """DIN 913, DIN 975, the three standoff types and the extrusion brackets (2026-11-01)."""
    classes = set(shipped.classes())
    sizes = ["M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10"]
    assert {f"DIN913_{s}" for s in sizes} | {f"DIN975_{s}" for s in sizes} <= classes
    for t in ("MF", "FF", "MM"):
        assert {f"STANDOFF_{t}_{s}" for s in sizes[:6]} <= classes
        assert shipped.families[f"STANDOFF_{t}"].kind == "standoff"
    assert {"BRACKET_2020", "BRACKET_3030", "BRACKET_4040"} <= classes
    assert set(shipped.families["BRACKET"].finishes) == {"aluminium", "aluminium_black"}
    steel_look = {"black_oxide", "stainless_a2", "stainless_a4", "plain_steel"}
    assert set(shipped.families["DIN913"].finishes) == steel_look
    assert shipped.part("STANDOFF_MF_M3x10").length == 10
    assert shipped.part("BRACKET_2020").length is None
    # threaded rod and standoffs are classes now, not unknown junk
    assert not {"threaded_rod", "standoff"} & set(shipped.unknown.types)


def test_set_screw_and_rod_lengths_never_cross(shipped):
    """Lying on their side they look alike, so rod pieces start at twice the longest set screw."""
    for size in shipped.families["DIN913"].sizes:
        grub = [p.length for p in shipped.parts if p.class_name == f"DIN913_{size}"]
        rod = [p.length for p in shipped.parts if p.class_name == f"DIN975_{size}"]
        assert max(grub) < 3.5 * size_value(size)  # about 3 x d (the next standard length)
        assert rod and min(rod) >= 2 * max(grub), size


def test_threaded_rod_is_cut_to_any_length():
    """No standard lengths: min_length per size, length_step, maximum from the config."""
    from screwsynth.common.config import load_config

    cfg = load_config()
    default = load_catalogue(SHIPPED, cfg.catalogue.max_length_mm, cfg.catalogue.rod_max_length_mm)
    m3 = [p.length for p in default.parts if p.class_name == "DIN975_M3"]
    assert m3 == list(range(20, int(cfg.catalogue.rod_max_length_mm) + 1, 5))
    assert all(p.cut_to_length for p in default.parts if p.family == "DIN975")
    assert not any(p.cut_to_length for p in default.parts if p.family != "DIN975")
    long = load_catalogue(SHIPPED, max_length_mm=80, cut_max_length_mm=150)
    assert max(p.length for p in long.parts if p.family == "DIN975") == 150
    assert max(p.length for p in long.parts if p.family == "DIN912") <= 80  # screws unchanged
    m2 = [p.length for p in long.parts if p.class_name == "DIN975_M2"]
    assert m2[:3] == [12, 15, 20]  # min_length, then the multiples of the step
