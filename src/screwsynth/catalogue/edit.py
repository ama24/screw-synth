"""Comment- and format-preserving edits of catalogue.yaml (ruamel.yaml round trip)."""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from screwsynth.common.io import write_text
from screwsynth.common.naming import format_size

from .schema import LENGTH_KINDS, CatalogueError


def _yaml() -> Any:
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # never re-wrap long flow lines
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def load_rt(path: str | os.PathLike[str]) -> Any:
    """Load a YAML file for round-trip editing."""
    return _yaml().load(Path(path).read_text(encoding="utf-8"))


def dump_rt(doc: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()


def save_rt(doc: Any, path: str | os.PathLike[str]) -> None:
    write_text(path, dump_rt(doc))


def _size_key(family: Mapping[str, Any], size: str) -> str:
    sizes = family.get("sizes") or {}
    if size in sizes:
        return size
    if family.get("kind") == "tnut":
        raise CatalogueError(f"size {size!r} not found; T-nut sizes look like S6_M4")
    key = format_size(size)
    if key not in sizes:
        raise CatalogueError(f"size {key!r} not in the family's size table ({list(sizes)})")
    return key


def _as_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def add_length(doc: Any, family: str, size: str, length: float, uncommon: bool = False) -> bool:
    """Insert a length (sorted) into family/size. Returns False if it was already present."""
    from ruamel.yaml.comments import CommentedSeq

    fams = doc.get("families") or {}
    if family not in fams:
        raise CatalogueError(f"unknown family {family!r}")
    fam = fams[family]
    if fam.get("kind") not in LENGTH_KINDS:
        raise CatalogueError(f"{family} has no lengths (only {', '.join(LENGTH_KINDS)} do)")
    if length <= 0:
        raise CatalogueError("length must be > 0")
    spec = fam["sizes"][_size_key(fam, size)]
    value = _as_number(length)
    changed = False
    keys = ["lengths", "uncommon"] if uncommon else ["lengths"]
    for key in keys:
        if key not in spec:
            seq = CommentedSeq()
            seq.fa.set_flow_style()
            spec[key] = seq
        seq = spec[key]
        if any(float(v) == float(value) for v in seq):
            continue
        idx = next((i for i, v in enumerate(seq) if float(v) > float(value)), len(seq))
        seq.insert(idx, value)
        changed = True
    return changed


def set_enabled(doc: Any, pattern: str, enabled: bool) -> None:
    """Append an enable/disable rule (replacing earlier rules with the same pattern)."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    rules = doc.get("rules")
    if rules is None:
        rules = CommentedSeq()
        doc["rules"] = rules
    for i in reversed(range(len(rules))):
        if str(rules[i].get("match")) == pattern:
            del rules[i]
    rules.fa.set_block_style()
    rule = CommentedMap([("match", pattern), ("enabled", enabled)])
    rule.fa.set_flow_style()
    rules.append(rule)


def add_families(doc: Any, template: Any, replace: bool = False) -> list[str]:
    """Copy the families of a template document into the catalogue. Returns their names."""
    fams = (template or {}).get("families") if isinstance(template, Mapping) else None
    if not isinstance(fams, Mapping) or not fams:
        raise CatalogueError("template must contain a non-empty 'families' mapping")
    target = doc["families"]
    names = []
    for name, body in fams.items():
        if name in target and not replace:
            raise CatalogueError(f"family {name!r} already exists (use --replace)")
        target[name] = body
        names.append(str(name))
    return names
