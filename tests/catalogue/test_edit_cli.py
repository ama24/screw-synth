import csv
import difflib
import shutil

import pytest

from conftest import MINIMAL, SHIPPED
from screwsynth.catalogue import edit
from screwsynth.catalogue.classes import build_classes, classes_up_to_date
from screwsynth.catalogue.cli import main
from screwsynth.catalogue.schema import CatalogueError, load_catalogue
from screwsynth.common.io import read_yaml


def _diff(a: str, b: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


@pytest.mark.parametrize("path", [MINIMAL, SHIPPED])
def test_round_trip_is_identical(path):
    text = path.read_text(encoding="utf-8")
    assert edit.dump_rt(edit.load_rt(path)) == text


def test_add_length_changes_one_line(minimal_file):
    before = minimal_file.read_text(encoding="utf-8")
    doc = edit.load_rt(minimal_file)
    assert edit.add_length(doc, "DIN912", "m3", 8)
    assert not edit.add_length(doc, "DIN912", "M3", 8)  # already there
    edit.save_rt(doc, minimal_file)
    diff = _diff(before, minimal_file.read_text(encoding="utf-8"))
    # only that line changes; ruamel keeps the comment (it may normalise its spacing)
    assert len(diff) == 2
    assert diff[0] == "-        lengths: [6, 10, 30]  # keep sorted"
    assert diff[1].startswith("+        lengths: [6, 8, 10, 30]")
    assert diff[1].endswith("# keep sorted")


def test_add_length_uncommon_and_new_list(minimal_file):
    doc = edit.load_rt(minimal_file)
    edit.add_length(doc, "DIN912", "M4", 12, uncommon=True)
    edit.save_rt(doc, minimal_file)
    text = minimal_file.read_text(encoding="utf-8")
    assert "lengths: [8, 12, 16]" in text
    assert "uncommon: [12]" in text
    assert "# nylon only in M3" in text


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("DIN999", "M3", 8), "unknown family"),
        (("DIN934", "M3", 8), "has no lengths"),
        (("DIN912", "M5", 8), "not in the family"),
        (("DIN912", "M3", -1), "> 0"),
    ],
)
def test_add_length_errors(minimal_file, args, match):
    with pytest.raises(CatalogueError, match=match):
        edit.add_length(edit.load_rt(minimal_file), *args)


def test_enable_disable_round_trip(minimal_file):
    before = minimal_file.read_text(encoding="utf-8")
    doc = edit.load_rt(minimal_file)
    edit.set_enabled(doc, "DIN912_M3", False)
    edit.save_rt(doc, minimal_file)
    cat = load_catalogue(minimal_file)
    assert not any(p.enabled for p in cat.parts if p.class_name == "DIN912_M3")
    text = minimal_file.read_text(encoding="utf-8")
    assert "- {match: DIN912_M3, enabled: false}" in text
    assert "# overrides" in text and "# keep sorted" in text

    doc = edit.load_rt(minimal_file)
    edit.set_enabled(doc, "DIN912_M3", True)  # replaces the earlier rule for the same pattern
    edit.save_rt(doc, minimal_file)
    after = minimal_file.read_text(encoding="utf-8")
    assert after.count("match: DIN912_M3") == 1
    assert all(p.enabled for p in load_catalogue(minimal_file).parts)
    changed = _diff(before, after)
    assert changed == [
        "-rules: []",
        "+rules:",
        "+  - {match: DIN912_M3, enabled: true}",
    ]


def test_cli_validate_list_stats(cli_env, capsys):
    cfg, _, classes = cli_env
    assert main(["classes", "--config", str(cfg)]) == 0
    assert classes.is_file()
    assert main(["validate", "--config", str(cfg)]) == 0
    assert main(["list", "--config", str(cfg), "--family", "DIN912", "--size", "M4"]) == 0
    out = capsys.readouterr().out
    assert "DIN912_M4x16" in out and "DIN912_M3x6" not in out
    assert main(["stats", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "Per family" in out and "Per class" in out and "Per finish" in out


def test_cli_max_length_override(cli_env, capsys):
    cfg, _, _ = cli_env
    main(["list", "--config", str(cfg), "--max-length", "10", "--family", "DIN912"])
    out = capsys.readouterr().out
    assert "DIN912_M3x10" in out and "DIN912_M3x30" not in out


def test_cli_validate_fails_on_errors(cli_env):
    cfg, cat_file, _ = cli_env
    text = cat_file.read_text(encoding="utf-8").replace("dk: 5.5", "dk: 2.5")
    cat_file.write_text(text, encoding="utf-8")
    assert main(["validate", "--config", str(cfg)]) == 1
    assert main(["classes", "--config", str(cfg)]) == 1


def test_cli_edits(cli_env, capsys):
    cfg, cat_file, _ = cli_env
    assert main(["disable", "DIN125A", "--config", str(cfg)]) == 0
    assert main(["enable", "NOPE*", "--config", str(cfg)]) == 1  # matches nothing
    assert main(["add-length", "DIN912", "M4", "20", "--config", str(cfg)]) == 0
    cat = load_catalogue(cat_file)
    assert "DIN912_M4x20" in [p.id for p in cat.parts]
    assert not cat.part("DIN125A_M3").enabled
    assert main(["add-length", "DIN934", "M3", "20", "--config", str(cfg)]) == 2


def test_cli_edit_rejected_when_invalid(cli_env):
    cfg, cat_file, _ = cli_env
    before = cat_file.read_text(encoding="utf-8")
    # M3 has b = 18 and full thread up to 20; a 19 mm M3 is fine, but break b first
    cat_file.write_text(before.replace("b: 18", "b: 25"), encoding="utf-8")
    edited = cat_file.read_text(encoding="utf-8")
    assert main(["add-length", "DIN912", "M3", "24", "--config", str(cfg)]) == 2
    assert cat_file.read_text(encoding="utf-8") == edited  # not saved


def test_cli_add_family(cli_env, tmp_path):
    cfg, cat_file, _ = cli_env
    template = tmp_path / "fam.yaml"
    template.write_text(
        "families:\n"
        "  DIN985:\n"
        "    standard: DIN 985\n"
        "    kind: nut\n"
        "    shape: nyloc_nut\n"
        "    finishes: {zinc_clear: all}\n"
        "    sizes:\n"
        "      M3: {pitch: 0.5, dims: {s: 5.5, h: 4.0, m: 2.9, e: 6.01}}\n",
        encoding="utf-8",
    )
    assert main(["add-family", "--from-template", str(template), "--config", str(cfg)]) == 0
    assert "DIN985_M3" in [p.id for p in load_catalogue(cat_file).parts]
    assert "# keep sorted" in cat_file.read_text(encoding="utf-8")
    assert main(["add-family", "--from-template", str(template), "--config", str(cfg)]) == 2


def test_shipped_template_is_valid(cli_env):
    cfg, cat_file, _ = cli_env
    template = SHIPPED.parents[1] / "tools" / "catalogue" / "templates" / "family_template.yaml"
    assert main(["add-family", "--from-template", str(template), "--config", str(cfg)]) == 0


def test_cli_export_csv(cli_env, tmp_path):
    cfg, _, _ = cli_env
    out = tmp_path / "parts.csv"
    assert main(["export-csv", str(out), "--config", str(cfg)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 8
    row = next(r for r in rows if r["id"] == "DIN912_M3x30")
    assert row["dim_dk"] == "5.5" and row["common"] == "False" and row["length"] == "30"


def test_classes_file(tmp_path):
    shutil.copy(MINIMAL, tmp_path / "c.yaml")
    cat = load_catalogue(tmp_path / "c.yaml")
    doc = build_classes(cat)
    assert doc["names"] == {
        0: "DIN125A_M3",
        1: "DIN912_M3",
        2: "DIN912_M4",
        3: "DIN934_M3",
        4: "TNUT_DROP_S6_M4",
    }
    assert build_classes(load_catalogue(tmp_path / "c.yaml")) == doc  # deterministic
    from screwsynth.catalogue.classes import write_classes

    write_classes(cat, tmp_path / "classes.yaml")
    assert classes_up_to_date(cat, tmp_path / "classes.yaml")
    assert read_yaml(tmp_path / "classes.yaml")["nc"] == 5


def test_shipped_classes_yaml_is_current():
    from screwsynth.common.config import load_config

    cat = load_catalogue(SHIPPED, max_length_mm=load_config().catalogue.max_length_mm)
    assert classes_up_to_date(cat, SHIPPED.parent / "classes.yaml")
