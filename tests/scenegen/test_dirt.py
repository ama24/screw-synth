import random

import pytest

from screwsynth.common.config import load_config
from screwsynth.scenegen import dirt


def _on(target, **kw):
    return {target: {**dirt.DEFAULTS[target], "enabled": True, "share": 1.0, **kw}}


def test_dirt_is_off_by_default():
    section = load_config().get("scenegen.dirt")
    for target in ("parts", "plate"):
        assert dirt.settings(section, target)["enabled"] is False
        assert dirt.sample(section, target, random.Random(1)) == dirt.CLEAN


@pytest.mark.parametrize("target", ["parts", "plate"])
def test_dirty_values_come_from_the_ranges(target):
    sec = _on(target, oil=[0.1, 0.2], dust=0.3, dust_size_mm=[0.1, 0.2])
    for seed in range(50):
        d = dirt.sample(sec, target, random.Random(seed))
        assert d["dirty"] and 0.1 <= d["oil"] <= 0.2 and d["dust"] == 0.3
        assert 0.1 <= d["dust_size_mm"] <= 0.2


def test_parts_and_plate_switch_separately():
    sec = {**_on("parts"), "plate": {"enabled": False}}
    assert dirt.sample(sec, "parts", random.Random(3))["dirty"]
    assert not dirt.sample(sec, "plate", random.Random(3))["dirty"]


def test_share_and_reproducibility():
    sec = _on("parts", share=0.3)
    rng = random.Random(7)
    dirty = sum(dirt.sample(sec, "parts", rng)["dirty"] for _ in range(4000))
    assert 0.27 < dirty / 4000 < 0.33
    assert dirt.sample(sec, "parts", random.Random(5)) == dirt.sample(
        sec, "parts", random.Random(5)
    )


def test_switching_dirt_does_not_change_other_random_choices():
    """The rng is advanced by the same amount whether dirt is on or off."""
    a, b = random.Random(11), random.Random(11)
    dirt.sample({"parts": {"enabled": False}}, "parts", a)
    dirt.sample(_on("parts"), "parts", b)
    assert a.random() == b.random()


def test_bad_settings():
    with pytest.raises(ValueError):
        dirt.settings({"parts": {"olive_oil": 1}}, "parts")
    with pytest.raises(ValueError):
        dirt.settings({}, "floor")
    with pytest.raises(ValueError):
        dirt.sample(_on("plate", oil=[0.1, 0.2, 0.3]), "plate", random.Random(0))


def test_oil_threshold_follows_coverage():
    t = [dirt.oil_threshold(c) for c in (0.0, 0.25, 0.5, 1.0)]
    assert t == sorted(t, reverse=True)  # more coverage: a lower threshold
    assert dirt.oil_threshold(-1) == dirt.oil_threshold(0) and dirt.oil_threshold(2) == t[-1]
