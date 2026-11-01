import pytest

from screwsynth.common.naming import (
    KEYPOINTS,
    class_name,
    format_size,
    parse_part_id,
    plain_id,
    screw_id,
    size_value,
    tnut_id,
)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("M4", "M4"),
        ("m4", "M4"),
        (4, "M4"),
        ("4", "M4"),
        (2.5, "M2.5"),
        ("M2.5", "M2.5"),
        (10.0, "M10"),
    ],
)
def test_format_size(size, expected):
    assert format_size(size) == expected


@pytest.mark.parametrize("bad", ["", "M", "Mx", "4mm", "M-3", 0])
def test_format_size_invalid(bad):
    with pytest.raises(ValueError):
        format_size(bad)


def test_size_value():
    assert size_value("M2.5") == 2.5


def test_builders():
    assert screw_id("DIN912", "M4", 16) == "DIN912_M4x16"
    assert screw_id("DIN7991", 2.5, 8.0) == "DIN7991_M2.5x8"
    assert plain_id("DIN934", "M4") == "DIN934_M4"
    assert plain_id("DIN125A", 10) == "DIN125A_M10"
    assert tnut_id("drop", 6, "M4") == "TNUT_DROP_S6_M4"
    assert tnut_id("SLIDE", 8, 5) == "TNUT_SLIDE_S8_M5"


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (screw_id, ("din912", "M4", 16)),
        (screw_id, ("DIN912", "M4", 0)),
        (screw_id, ("TNUT", "M4", 16)),
        (plain_id, ("DIN 934", "M4")),
        (tnut_id, ("HAMMER", 6, "M4")),
        (tnut_id, ("DROP", 0, "M4")),
    ],
)
def test_builders_invalid(call, args):
    with pytest.raises(ValueError):
        call(*args)


@pytest.mark.parametrize(
    "part_id",
    [
        "DIN912_M4x16",
        "DIN933_M10x80",
        "ISO7380_M2.5x6",
        "DIN7991_M3x12",
        "DIN912_M2x3.5",
        "DIN934_M4",
        "DIN125A_M2.5",
        "TNUT_DROP_S6_M4",
        "TNUT_SLIDE_S8_M6",
    ],
)
def test_round_trip(part_id):
    assert parse_part_id(part_id).part_id == part_id


def test_parse_screw():
    p = parse_part_id("DIN912_M2.5x12")
    assert (p.family, p.size, p.length) == ("DIN912", "M2.5", 12.0)
    assert p.is_screw and not p.is_tnut
    assert p.class_name == "DIN912_M2.5"


def test_parse_nut_and_washer():
    p = parse_part_id("DIN125A_M4")
    assert (p.family, p.size, p.length) == ("DIN125A", "M4", None)
    assert not p.is_screw
    assert p.class_name == "DIN125A_M4"


def test_parse_tnut():
    p = parse_part_id("TNUT_DROP_S6_M4")
    assert (p.family, p.tnut_kind, p.slot, p.size) == ("TNUT", "DROP", 6, "M4")
    assert p.is_tnut and not p.is_screw
    assert p.class_name == "TNUT_DROP_S6_M4"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "DIN912",
        "DIN912_M4x",
        "DIN912_4x16",
        "din912_M4x16",
        "DIN912_M4X16",
        "DIN912-M4x16",
        "TNUT_HAMMER_S6_M4",
        "TNUT_DROP_6_M4",
        "TNUT_M4",
        "TNUT_M4x10",
        "DIN912_M4x16 ",
    ],
)
def test_parse_invalid(bad):
    with pytest.raises(ValueError):
        parse_part_id(bad)


@pytest.mark.parametrize(
    ("part_id", "expected"),
    [
        ("DIN912_M4x16", "DIN912_M4"),
        ("DIN7991_M2.5x10", "DIN7991_M2.5"),
        ("DIN934_M4", "DIN934_M4"),
        ("TNUT_SLIDE_S8_M5", "TNUT_SLIDE_S8_M5"),
    ],
)
def test_class_name(part_id, expected):
    assert class_name(part_id) == expected


def test_keypoint_order():
    assert KEYPOINTS == ("head_top", "bearing", "tip")


def test_profile_parts_and_multiword_families():
    from screwsynth.common.naming import profile_id

    assert profile_id("BRACKET", 2020) == "BRACKET_2020"
    assert profile_id("BRACKET", 2020, "cast") == "BRACKET_2020_CAST"
    styled = parse_part_id("BRACKET_2020_CAST")
    assert (styled.style, styled.size, styled.class_name) == ("CAST", "2020", "BRACKET_2020")
    assert styled.part_id == "BRACKET_2020_CAST"
    p = parse_part_id("BRACKET_3030")
    assert (p.family, p.size, p.profile, p.class_name) == (
        "BRACKET",
        "3030",
        "3030",
        "BRACKET_3030",
    )
    s = parse_part_id("STANDOFF_MF_M2.5x12")
    assert (s.family, s.size, s.length, s.class_name) == (
        "STANDOFF_MF",
        "M2.5",
        12.0,
        "STANDOFF_MF_M2.5",
    )
    assert parse_part_id("STANDOFF_FF_M3").class_name == "STANDOFF_FF_M3"
    with pytest.raises(ValueError):
        profile_id("BRACKET", "20x20")
    with pytest.raises(ValueError):
        screw_id("TNUT_X", "M3", 5)  # T-nut and UNKNOWN prefixes are reserved
    with pytest.raises(ValueError):
        plain_id("UNKNOWN_FOO", "M3")
