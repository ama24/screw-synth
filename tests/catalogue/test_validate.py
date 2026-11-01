"""One test per validation rule: mutate the minimal fixture and expect exactly that issue code."""

import pytest

from screwsynth.catalogue.schema import parse_catalogue
from screwsynth.catalogue.validate import errors, validate


def _fam(d, name="DIN912"):
    return d["families"][name]


def _m3(d):
    return _fam(d)["sizes"]["M3"]


def test_minimal_is_valid(minimal_data):
    assert validate(parse_catalogue(minimal_data)) == []


CASES = {
    "duplicate-id": lambda d: _m3(d).update(lengths=[6, 10, 10, 30]),
    "missing-dim": lambda d: _m3(d)["dims"].pop("t"),
    "dim-positive": lambda d: _m3(d)["dims"].update(t=0),
    "d-mismatch": lambda d: _m3(d)["dims"].update(d=3.5),
    "dim-sanity": lambda d: _m3(d)["dims"].update(dk=2.5),  # head smaller than thread
    "pitch": lambda d: _m3(d).pop("pitch"),
    "no-lengths": lambda d: _m3(d).update(lengths=[], uncommon=[]),
    "uncommon": lambda d: _m3(d).update(uncommon=[12]),
    "length": lambda d: d["families"]["DIN934"]["sizes"]["M3"].update(lengths=[5]),
    "thread-length": lambda d: _m3(d)["dims"].update(b=40),  # M3x30 partially threaded, b > 30
    "unknown-finish": lambda d: _fam(d)["finishes"].update(gold="all"),
    "finish-size": lambda d: _fam(d)["finishes"].update(nylon_white=["M5"]),
    "no-finishes": lambda d: _fam(d).update(finishes={}),
    "finish-material": lambda d: d["finishes"]["zinc_clear"].update(material="wood"),
    "nominal-length": lambda d: _fam(d).pop("nominal_length"),
    "shape-kind": lambda d: _fam(d).update(shape="hex_nut"),
    "no-sizes": lambda d: d["families"]["DIN934"].update(sizes={}),
    "step-source": lambda d: _fam(d).update(source="step"),
}


@pytest.mark.parametrize("code", sorted(CASES))
def test_rule(minimal_data, code):
    CASES[code](minimal_data)
    issues = validate(parse_catalogue(minimal_data))
    codes = {i.code for i in errors(issues)}
    assert code in codes, [str(i) for i in issues]


def test_dim_sanity_per_shape(minimal_data):
    minimal_data["families"]["DIN934"]["sizes"]["M3"]["dims"]["e"] = 7.5  # e > s / cos30
    minimal_data["families"]["DIN125A"]["sizes"]["M3"]["dims"]["d2"] = 3.0  # d2 < d1
    minimal_data["families"]["TNUT_DROP"]["sizes"]["S6_M4"]["dims"]["neck_height"] = 5  # > height
    issues = errors(validate(parse_catalogue(minimal_data)))
    where = {i.where for i in issues if i.code == "dim-sanity"}
    assert where == {
        "families.DIN934.sizes.M3",
        "families.DIN125A.sizes.M3",
        "families.TNUT_DROP.sizes.S6_M4",
    }


def test_class_mismatch(minimal_data):
    # a T-nut family whose tnut_kind collides with another family's class names
    minimal_data["families"]["TNUT_SLIDE"] = dict(minimal_data["families"]["TNUT_DROP"])
    minimal_data["families"]["TNUT_SLIDE"]["shape"] = "tnut_slide"
    issues = errors(validate(parse_catalogue(minimal_data)))
    assert {"duplicate-id", "class-mismatch"} <= {i.code for i in issues}


def test_step_source_valid(minimal_data):
    fam = _fam(minimal_data)
    fam["source"] = "step"
    kps = {"head_top": [0, 0, 3], "bearing": [0, 0, 0], "tip": [0, 0, -6]}
    for size in fam["sizes"].values():
        size["step"] = {
            L: {"path": f"data/step/x{L}.step", "keypoints": kps} for L in size["lengths"]
        }
    cat = parse_catalogue(minimal_data)
    assert errors(validate(cat)) == []
    assert cat.part("DIN912_M3x6").step["path"] == "data/step/x6.step"


def test_step_source_needs_keypoints(minimal_data):
    fam = _fam(minimal_data)
    fam["source"] = "step"
    for size in fam["sizes"].values():
        size["step"] = {L: {"path": "x.step"} for L in size["lengths"]}
    issues = errors(validate(parse_catalogue(minimal_data)))
    assert any(i.code == "step-source" and "keypoints" in i.message for i in issues)


def test_unmatched_rule_is_warning(minimal_data):
    minimal_data["rules"] = [{"match": "DIN999*", "enabled": False}]
    issues = validate(parse_catalogue(minimal_data))
    assert [i.code for i in issues] == ["rule-unmatched"]
    assert errors(issues) == []


def test_cut_to_length_rules(minimal_data):
    import copy

    from screwsynth.catalogue.schema import parse_catalogue
    from screwsynth.catalogue.validate import errors, validate

    data = copy.deepcopy(minimal_data)
    fam = next(f for f in data["families"].values() if f.get("kind") == "screw")
    fam["length_step"] = 5
    for spec in fam["sizes"].values():
        spec.pop("lengths", None)
        spec.pop("uncommon", None)
        spec["min_length"] = 10
    cat = parse_catalogue(data, cut_max_length_mm=30)
    assert errors(validate(cat)) == []
    assert sorted({p.length for p in cat.parts if p.kind == "screw"}) == [10, 15, 20, 25, 30]
    for spec in fam["sizes"].values():
        spec["lengths"] = [12]  # a list on a cut-to-length size is an error
    assert any(i.code == "length" for i in errors(validate(parse_catalogue(data))))
