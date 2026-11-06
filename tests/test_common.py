import json
import random

import pytest

from screwsynth.common.config import Config, clear_config_cache, load_config
from screwsynth.common.io import read_json, read_yaml, write_json, write_yaml
from screwsynth.common.paths import repo_root, resolve
from screwsynth.common.seed import derive_seed, rng

TOOLS = [
    "catalogue",
    "partgen",
    "calibration",
    "restposes",
    "scenegen",
    "postprocess",
    "labels",
    "measure",
]


def test_repo_root_and_resolve(tmp_path):
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert resolve("configs/config.yaml") == root / "configs" / "config.yaml"
    assert resolve(tmp_path) == tmp_path


def test_shipped_config_loads_and_is_cached():
    clear_config_cache()
    cfg = load_config()
    assert cfg is load_config()
    assert cfg.world_scale == 10
    assert cfg.get("catalogue.max_length_mm") == 80
    assert cfg.scenegen.objects == [15, 40]
    assert cfg.get("does.not.exist", "x") == "x"
    for tool in TOOLS:
        assert tool in cfg, f"config.yaml has no section for {tool}"


def test_config_env_override(tmp_path, monkeypatch):
    path = tmp_path / "alt.yaml"
    path.write_text("world_scale: 5\nnested: {a: {b: 1}}\n", encoding="utf-8")
    monkeypatch.setenv("SCREWSYNTH_CONFIG", str(path))
    clear_config_cache()
    try:
        cfg = load_config()
        assert cfg.world_scale == 5
        assert cfg.nested.a.b == 1
        assert cfg.require("nested.a.b") == 1
        with pytest.raises(KeyError):
            cfg.require("nested.a.c")
        with pytest.raises(AttributeError):
            _ = cfg.missing
    finally:
        clear_config_cache()


def test_config_mapping():
    cfg = Config({"a": 1, "b": {"c": 2}})
    assert dict(cfg)["a"] == 1
    assert len(cfg) == 2
    assert isinstance(cfg["b"], Config)


def test_json_round_trip(tmp_path):
    data = {"id": "DIN912_M4x16", "k": [1.0, 2.5], "nested": {"z": None}}
    path = tmp_path / "sub" / "meta.json"
    write_json(path, data)
    assert read_json(path) == data
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert list(path.parent.iterdir()) == [path]  # no temp files left behind


def test_json_writes_config_sections(tmp_path):
    """Config hands out nested sections as Config objects; they must serialise like dicts."""
    cfg = Config({"postprocess": {"blur": {"sigma_px": [0.3, 0.9]}}})
    write_json(tmp_path / "m.json", {"blur": cfg.get("postprocess.blur")})
    assert read_json(tmp_path / "m.json") == {"blur": {"sigma_px": [0.3, 0.9]}}


def test_yaml_round_trip(tmp_path):
    data = {"b": [1, 2], "a": {"x": "M2.5"}}
    path = tmp_path / "x.yaml"
    write_yaml(path, data)
    assert read_yaml(path) == data
    assert path.read_text(encoding="utf-8").startswith("b:")  # key order kept


def test_seeds_are_deterministic():
    assert derive_seed(0, "scene", 1) == derive_seed(0, "scene", 1)
    assert derive_seed(0, "scene", 1) != derive_seed(0, "scene", 2)
    assert rng(3).random() == random.Random(3).random()


def test_keep_awake_is_a_harmless_context():
    from screwsynth.common.awake import keep_awake

    with keep_awake("test") as active:
        assert active in (True, False)
