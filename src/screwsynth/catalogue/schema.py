"""Catalogue data model: families with per-size tables, expanded into parts in code.

Pure Python (no pydantic, no numpy) so the expanded catalogue can also be used inside Blender.
`parse_catalogue` raises CatalogueError for structural problems (wrong types, unknown keys,
unparsable sizes); semantic checks (dimension sanity, finishes, ids) live in validate.py.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from screwsynth.common.naming import (
    UNKNOWN,
    format_size,
    plain_id,
    profile_id,
    screw_id,
    size_value,
    tnut_id,
    unknown_id,
)

KINDS = ("screw", "standoff", "nut", "washer", "tnut", "bracket", "unknown")
# kinds whose parts have a nominal length (a part per length) and measured keypoints
LENGTH_KINDS = ("screw", "standoff")
# kinds whose sizes are extrusion profiles (2020, 3030, 4040) instead of thread sizes
PROFILE_KINDS = ("bracket",)
NOMINAL_LENGTH_MODES = ("bearing_to_tip", "head_top_to_tip")
SOURCES = ("parametric", "step")
MATERIALS = ("metal", "plastic")

# Dimensions each parametric shape needs (standard symbols, mm). "d" is filled in from the size.
REQUIRED_DIMS: dict[str, tuple[str, ...]] = {
    "socket_cap": ("d", "dk", "k", "s", "t", "b", "ds"),
    "hex": ("d", "s", "k", "e"),
    "button": ("d", "dk", "k", "s", "t"),
    "countersunk": ("d", "dk", "k", "s", "t", "angle"),
    "set_screw": ("d", "s", "t"),
    "threaded_rod": ("d",),
    "standoff_mf": ("d", "s", "e", "ls", "lf"),
    "standoff_ff": ("d", "s", "e", "lf"),
    "standoff_mm": ("d", "s", "e", "ls"),
    "hex_nut": ("d", "s", "m", "e"),
    "nyloc_nut": ("d", "s", "h", "m", "e"),
    "washer": ("d1", "d2", "h"),
    "tnut_slide": ("d", "length", "width", "height", "neck_width", "neck_height"),
    "tnut_drop": ("d", "length", "width", "height", "neck_width", "neck_height"),
    "corner_bracket": ("a", "w", "t", "ts", "dh", "slot", "key_w", "key_h"),
    "step": (),
}
SHAPE_KIND = {
    "socket_cap": "screw",
    "hex": "screw",
    "button": "screw",
    "countersunk": "screw",
    "set_screw": "screw",
    "threaded_rod": "screw",
    "standoff_mf": "standoff",
    "standoff_ff": "standoff",
    "standoff_mm": "standoff",
    "hex_nut": "nut",
    "nyloc_nut": "nut",
    "washer": "washer",
    "tnut_slide": "tnut",
    "tnut_drop": "tnut",
    "corner_bracket": "bracket",
}

# Non-catalogue parts the machine must reject (class UNKNOWN): type -> default parameter ranges
# (mm). Each variant draws its parameters from these ranges, seeded by its part id.
UNKNOWN_TYPES: dict[str, dict[str, tuple[float, float]]] = {
    "wood_screw": {"d": (3.0, 5.0), "length": (12.0, 50.0)},
    "threaded_rod": {"d": (3.0, 8.0), "length": (10.0, 70.0)},
    "split_washer": {"d": (3.0, 10.0)},
    "tooth_washer": {"d": (3.0, 10.0)},
    "pin": {"d": (2.0, 8.0), "length": (6.0, 40.0)},
    "bracket": {"size": (12.0, 40.0)},
    "standoff": {"d": (3.0, 6.0), "length": (8.0, 30.0)},
    "plate": {"size": (15.0, 50.0)},
    "blob": {"size": (5.0, 25.0)},
}
_UNKNOWN_KEYS = {"enabled", "variants", "finishes", "types", "notes"}

_FAMILY_KEYS = {
    "title",
    "standard",
    "kind",
    "shape",
    "nominal_length",
    "enabled",
    "verified",
    "dims_source",
    "finishes",
    "common_dims",
    "sizes",
    "source",
    "tnut_kind",
    "length_step",
    "notes",
}
_SIZE_KEYS = {
    "pitch",
    "dims",
    "lengths",
    "min_length",
    "uncommon",
    "enabled",
    "common",
    "verified",
    "step",
    "styles",
}
DEFAULT_CUT_MAX_LENGTH_MM = 80.0  # cut-to-length families when no maximum is configured
_TNUT_KEY_RE = re.compile(r"^S(?P<slot>[1-9]\d*)_(?P<size>M\d+(?:\.\d+)?)$")
_FAMILY_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z][A-Z0-9]*)*$")
_PROFILE_KEY_RE = re.compile(r"^\d{4}$")
_TNUT_FAMILY_RE = re.compile(r"^TNUT_[A-Z0-9_]+$")


class CatalogueError(ValueError):
    """The catalogue file is structurally invalid."""


@dataclass(frozen=True)
class Finish:
    id: str
    label: str
    material: str


@dataclass
class SizeSpec:
    key: str  # table key: "M4", or "S6_M4" for T-nuts
    size: str  # thread size, e.g. "M4"
    pitch: float | None
    dims: dict[str, float]
    lengths: list[float] = field(default_factory=list)
    uncommon: list[float] = field(default_factory=list)
    enabled: bool | None = None
    common: bool = True
    verified: bool | None = None
    slot: int | None = None
    step: dict[str, Any] | None = None
    min_length: float | None = None  # cut-to-length families: shortest piece
    # profile parts: look-alike variants of the class, style -> dimension overrides (one part each)
    styles: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class Family:
    name: str
    kind: str
    standard: str
    shape: str
    enabled: bool
    verified: bool
    dims_source: str
    finishes: dict[str, tuple[str, ...] | None]  # finish -> size keys, None = all sizes
    sizes: dict[str, SizeSpec]
    title: str = ""
    nominal_length: str | None = None
    source: str = "parametric"
    tnut_kind: str | None = None
    # cut to length (threaded rod): lengths from each size's min_length up to the configured
    # maximum in these steps, instead of a list of standard lengths
    length_step: float | None = None

    def finishes_for(self, size_key: str) -> tuple[str, ...]:
        return tuple(f for f, keys in self.finishes.items() if keys is None or size_key in keys)


@dataclass(frozen=True)
class Part:
    id: str
    class_name: str
    family: str
    kind: str
    standard: str
    shape: str
    size: str
    length: float | None
    pitch: float | None
    enabled: bool
    common: bool
    finishes: tuple[str, ...]
    dims: Mapping[str, float]
    dims_source: str
    verified: bool
    source: str = "parametric"
    nominal_length: str | None = None
    slot: int | None = None
    tnut_kind: str | None = None
    step: Mapping[str, Any] | None = None
    size_key: str = ""
    cut_to_length: bool = False  # any length (threaded rod): measured, rounded to 1 mm
    style: str | None = None  # profile parts: variant within the class (BRACKET_2020_CAST)

    @property
    def fully_threaded(self) -> bool | None:
        """For screws: True when the thread runs up to the head (None for non-screws)."""
        if self.kind != "screw" or self.length is None:
            return None
        limit = self.dims.get("full_thread_max_length")
        if "b" not in self.dims or limit is None:
            return True
        return self.length <= limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "family": self.family,
            "kind": self.kind,
            "standard": self.standard,
            "shape": self.shape,
            "size": self.size,
            "length": self.length,
            "pitch": self.pitch,
            "enabled": self.enabled,
            "common": self.common,
            "finishes": list(self.finishes),
            "dims": dict(self.dims),
            "dims_source": self.dims_source,
            "verified": self.verified,
            "source": self.source,
            "nominal_length": self.nominal_length,
            "fully_threaded": self.fully_threaded,
            "slot": self.slot,
            "tnut_kind": self.tnut_kind,
            "step": dict(self.step) if self.step is not None else None,
            "cut_to_length": self.cut_to_length,
            "style": self.style,
        }

    @property
    def hash(self) -> str:
        """Hash of everything that defines the part's geometry (used for incremental builds)."""
        geom = {
            k: v
            for k, v in self.to_dict().items()
            if k
            not in ("enabled", "common", "finishes", "verified", "dims_source", "cut_to_length")
        }
        return _hash(geom)


@dataclass(frozen=True)
class Rule:
    match: str
    enabled: bool

    def matches(self, part: Part) -> bool:
        return any(
            fnmatch.fnmatchcase(c, self.match)
            for c in (part.id, part.class_name, part.family, part.size)
        )


@dataclass(frozen=True)
class UnknownSpec:
    """The `unknown:` section: junk types, their parameter ranges and finishes."""

    enabled: bool
    variants: int  # random variants per type
    finishes: tuple[str, ...]
    types: dict[str, dict[str, tuple[float, float]]]  # type -> parameter -> (lo, hi)
    type_finishes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    type_enabled: dict[str, bool] = field(default_factory=dict)


@dataclass
class Catalogue:
    version: int
    finishes: dict[str, Finish]
    families: dict[str, Family]
    rules: list[Rule] = field(default_factory=list)
    max_length_mm: float | None = None
    unknown: UnknownSpec | None = None
    cut_max_length_mm: float | None = (
        None  # longest cut-to-length piece (catalogue.rod_max_length_mm)
    )
    _parts: list[Part] | None = field(default=None, init=False, repr=False)

    @property
    def parts(self) -> list[Part]:
        """All parts (enabled or not), expanded family x size x length up to max_length_mm."""
        if self._parts is None:
            self._parts = expand(self)
        return self._parts

    @property
    def enabled_parts(self) -> list[Part]:
        return [p for p in self.parts if p.enabled]

    def part(self, part_id: str) -> Part:
        for p in self.parts:
            if p.id == part_id:
                return p
        raise KeyError(part_id)

    def classes(self, enabled_only: bool = True) -> list[str]:
        """Class names in deterministic order: family, then slot, then numeric size."""
        parts = self.enabled_parts if enabled_only else self.parts
        first: dict[str, Part] = {}
        for p in parts:
            first.setdefault(p.class_name, p)
        return sorted(first, key=lambda c: class_sort_key(first[c]))

    @property
    def hash(self) -> str:
        return _hash([[p.id, p.hash] for p in self.enabled_parts])


def class_sort_key(part: Part) -> tuple[str, int, float]:
    return (part.family, part.slot or 0, size_value(part.size) if part.size != "-" else 0.0)


def _hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------- parsing


def _expect(value: Any, typ: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(value, typ):
        names = typ.__name__ if isinstance(typ, type) else "/".join(t.__name__ for t in typ)
        raise CatalogueError(f"{where}: expected {names}, got {type(value).__name__}")
    return value


def _num(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogueError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _num_list(value: Any, where: str) -> list[float]:
    _expect(value, Sequence, where)
    if isinstance(value, str):
        raise CatalogueError(f"{where}: expected a list of numbers")
    return [_num(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _check_keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CatalogueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _parse_size_key(key: str, kind: str, where: str) -> tuple[str, int | None]:
    if kind in PROFILE_KINDS:
        if not _PROFILE_KEY_RE.match(str(key)):
            raise CatalogueError(f"{where}: profile size keys look like 2020, got {key!r}")
        return str(key), None
    if kind == "tnut":
        m = _TNUT_KEY_RE.match(str(key))
        if not m:
            raise CatalogueError(f"{where}: T-nut size keys look like S6_M4, got {key!r}")
        return format_size(m["size"]), int(m["slot"])
    try:
        size = format_size(key)
    except ValueError as exc:
        raise CatalogueError(f"{where}: {exc}") from None
    if size != str(key):
        raise CatalogueError(f"{where}: write the size as {size!r}, not {key!r}")
    return size, None


def _parse_size(
    key: str, raw: Any, fam_kind: str, shape: str, common_dims: Mapping[str, float], where: str
) -> SizeSpec:
    _expect(raw, Mapping, where)
    _check_keys(raw, _SIZE_KEYS, where)
    size, slot = _parse_size_key(key, fam_kind, where)
    dims_raw = _expect(raw.get("dims", {}), Mapping, f"{where}.dims")
    dims = {str(k): _num(v, f"{where}.dims.{k}") for k, v in common_dims.items()}
    dims.update({str(k): _num(v, f"{where}.dims.{k}") for k, v in dims_raw.items()})
    if shape != "washer" and fam_kind not in PROFILE_KINDS:
        dims.setdefault("d", size_value(size))
    if shape == "socket_cap":
        dims.setdefault("ds", dims["d"])
    pitch = raw.get("pitch")
    min_length = raw.get("min_length")
    styles: dict[str, dict[str, float]] = {}
    for name, over in _expect(raw.get("styles", {}) or {}, Mapping, f"{where}.styles").items():
        if fam_kind not in PROFILE_KINDS:
            raise CatalogueError(f"{where}.styles: only profile parts (brackets) have styles")
        over = _expect(over or {}, Mapping, f"{where}.styles.{name}")
        styles[str(name)] = {str(k): _num(v, f"{where}.styles.{name}.{k}") for k, v in over.items()}
    step = raw.get("step")
    if step is not None:
        step = dict(_expect(step, Mapping, f"{where}.step"))
    return SizeSpec(
        key=str(key),
        size=size,
        pitch=None if pitch is None else _num(pitch, f"{where}.pitch"),
        dims=dims,
        lengths=_num_list(raw.get("lengths", []), f"{where}.lengths"),
        uncommon=_num_list(raw.get("uncommon", []), f"{where}.uncommon"),
        enabled=None if raw.get("enabled") is None else bool(raw["enabled"]),
        common=bool(raw.get("common", True)),
        verified=None if raw.get("verified") is None else bool(raw["verified"]),
        slot=slot,
        step=step,
        min_length=None if min_length is None else _num(min_length, f"{where}.min_length"),
        styles=styles,
    )


def _parse_finish_sizes(value: Any, where: str) -> tuple[str, ...] | None:
    if value == "all" or value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CatalogueError(f"{where}: expected 'all' or a list of size keys")
    return tuple(str(v) for v in value)


def parse_family(name: str, raw: Any) -> Family:
    where = f"families.{name}"
    _expect(raw, Mapping, where)
    _check_keys(raw, _FAMILY_KEYS, where)
    for key in ("kind", "standard", "sizes"):
        if key not in raw:
            raise CatalogueError(f"{where}: missing required key {key!r}")
    kind = str(raw["kind"])
    if kind not in KINDS:
        raise CatalogueError(f"{where}.kind: {kind!r} is not one of {KINDS}")
    name_re = _TNUT_FAMILY_RE if kind == "tnut" else _FAMILY_RE
    reserved = name.split("_")[0] == UNKNOWN or (kind != "tnut" and name.split("_")[0] == "TNUT")
    if not name_re.match(name) or name == "TNUT" or reserved:
        raise CatalogueError(
            f"{where}: family names use upper-case letters and digits, e.g. DIN912"
        )
    tnut_kind = None if raw.get("tnut_kind") is None else str(raw["tnut_kind"]).upper()
    if (kind == "tnut") != (tnut_kind in ("SLIDE", "DROP")):
        raise CatalogueError(f"{where}: T-nut families (and only they) need tnut_kind SLIDE|DROP")
    source = str(raw.get("source", "parametric"))
    if source not in SOURCES:
        raise CatalogueError(f"{where}.source: {source!r} is not one of {SOURCES}")
    shape = str(raw.get("shape", "step" if source == "step" else ""))
    if shape not in REQUIRED_DIMS:
        raise CatalogueError(f"{where}.shape: {shape!r} is not one of {sorted(REQUIRED_DIMS)}")
    common_dims = _expect(raw.get("common_dims", {}), Mapping, f"{where}.common_dims")
    finishes_raw = _expect(raw.get("finishes", {}), Mapping, f"{where}.finishes")
    sizes_raw = _expect(raw["sizes"], Mapping, f"{where}.sizes")
    sizes = {
        str(k): _parse_size(str(k), v, kind, shape, common_dims, f"{where}.sizes.{k}")
        for k, v in sizes_raw.items()
    }
    return Family(
        name=name,
        kind=kind,
        standard=str(raw["standard"]),
        shape=shape,
        enabled=bool(raw.get("enabled", True)),
        verified=bool(raw.get("verified", False)),
        dims_source=str(raw.get("dims_source", "")),
        finishes={
            str(f): _parse_finish_sizes(v, f"{where}.finishes.{f}") for f, v in finishes_raw.items()
        },
        sizes=sizes,
        title=str(raw.get("title", "")),
        nominal_length=None if raw.get("nominal_length") is None else str(raw["nominal_length"]),
        source=source,
        tnut_kind=tnut_kind,
        length_step=None
        if raw.get("length_step") is None
        else _num(raw["length_step"], f"{where}.length_step"),
    )


def parse_catalogue(
    data: Any, max_length_mm: float | None = None, cut_max_length_mm: float | None = None
) -> Catalogue:
    """Build a Catalogue from the loaded YAML mapping (PyYAML or ruamel round-trip data)."""
    _expect(data, Mapping, "catalogue")
    _check_keys(data, {"version", "finishes", "families", "rules", "unknown"}, "catalogue")
    finishes: dict[str, Finish] = {}
    for fid, raw in _expect(data.get("finishes", {}), Mapping, "finishes").items():
        _expect(raw, Mapping, f"finishes.{fid}")
        finishes[str(fid)] = Finish(
            id=str(fid),
            label=str(raw.get("label", fid)),
            material=str(raw.get("material", "metal")),
        )
    families = {
        str(name): parse_family(str(name), raw)
        for name, raw in _expect(data.get("families", {}) or {}, Mapping, "families").items()
    }
    rules = parse_rules(data.get("rules", []), "rules")
    unknown = parse_unknown(data.get("unknown"), set(finishes))
    return Catalogue(
        version=int(data.get("version", 1)),
        finishes=finishes,
        families=families,
        rules=rules,
        max_length_mm=max_length_mm,
        unknown=unknown,
        cut_max_length_mm=cut_max_length_mm,
    )


def parse_unknown(raw: Any, finishes: set[str]) -> UnknownSpec | None:
    """The optional `unknown:` section (see UNKNOWN_TYPES for the types and their ranges)."""
    if raw is None:
        return None
    _expect(raw, Mapping, "unknown")
    _check_keys(raw, _UNKNOWN_KEYS, "unknown")
    variants = int(raw.get("variants", 6))
    if variants < 1:
        raise CatalogueError("unknown.variants must be >= 1")
    fin = tuple(
        str(f) for f in _expect(raw.get("finishes", []) or [], Sequence, "unknown.finishes")
    )
    types: dict[str, dict[str, tuple[float, float]]] = {}
    type_fin: dict[str, tuple[str, ...]] = {}
    type_on: dict[str, bool] = {}
    for name, spec in _expect(raw.get("types", {}) or {}, Mapping, "unknown.types").items():
        name = str(name)
        where = f"unknown.types.{name}"
        if name not in UNKNOWN_TYPES:
            raise CatalogueError(f"{where}: unknown type; choose from {sorted(UNKNOWN_TYPES)}")
        spec = dict(_expect(spec or {}, Mapping, where))
        ranges = dict(UNKNOWN_TYPES[name])
        for key, value in spec.items():
            if key == "finishes":
                type_fin[name] = tuple(
                    str(f) for f in _expect(value, Sequence, f"{where}.finishes")
                )
            elif key == "enabled":
                type_on[name] = bool(value)
            elif key in ranges:
                lo, hi = _num_list(value, f"{where}.{key}")
                if not 0 < lo <= hi:
                    raise CatalogueError(f"{where}.{key}: need 0 < low <= high, got {value}")
                ranges[key] = (lo, hi)
            else:
                raise CatalogueError(f"{where}: unknown key {key!r}; allowed {sorted(ranges)}")
        types[name] = ranges
    for f in (*fin, *(f for fs in type_fin.values() for f in fs)):
        if f not in finishes:
            raise CatalogueError(f"unknown: finish {f!r} is not defined under finishes")
    if not fin and any(n not in type_fin for n in types):
        raise CatalogueError("unknown.finishes: list the finishes the unknown parts can have")
    return UnknownSpec(
        enabled=bool(raw.get("enabled", True)),
        variants=variants,
        finishes=fin,
        types=types,
        type_finishes=type_fin,
        type_enabled=type_on,
    )


def parse_rules(data: Any, where: str) -> list[Rule]:
    """[{match: <glob>, enabled: bool}, ...] -> rules (applied in order; the last match wins)."""
    rules = []
    for i, raw in enumerate(_expect(data or [], Sequence, where)):
        _expect(raw, Mapping, f"{where}[{i}]")
        _check_keys(raw, {"match", "enabled"}, f"{where}[{i}]")
        if "match" not in raw or "enabled" not in raw:
            raise CatalogueError(f"{where}[{i}]: needs both 'match' and 'enabled'")
        rules.append(Rule(match=str(raw["match"]), enabled=bool(raw["enabled"])))
    return rules


def load_catalogue(
    path: str | os.PathLike[str],
    max_length_mm: float | None = None,
    cut_max_length_mm: float | None = None,
) -> Catalogue:
    from screwsynth.common.io import read_yaml

    return parse_catalogue(read_yaml(path), max_length_mm, cut_max_length_mm)


def catalogue_from_config(cfg: Any, path: str | os.PathLike[str] | None = None) -> Catalogue:
    """The catalogue as every tool should see it: paths.catalogue, catalogue.max_length_mm, and the
    config's catalogue.extra_rules applied after the file's own rules. Dataset recipes select their
    parts through extra_rules, so catalogue.yaml itself never has to change per dataset."""
    from screwsynth.common.paths import resolve

    cat = load_catalogue(
        resolve(path or cfg.get("paths.catalogue", "configs/catalogue.yaml")),
        max_length_mm=cfg.get("catalogue.max_length_mm"),
        cut_max_length_mm=cfg.get("catalogue.rod_max_length_mm"),
    )
    cat.rules = [
        *cat.rules,
        *parse_rules(cfg.get("catalogue.extra_rules"), "catalogue.extra_rules"),
    ]
    return cat


# ----------------------------------------------------------------------------------- expansion


def _part_ids(
    fam: Family, spec: SizeSpec, length: float | None, style: str | None = None
) -> tuple[str, str]:
    if fam.kind == "tnut":
        pid = tnut_id(fam.tnut_kind or "", spec.slot or 0, spec.size)
        return pid, pid
    if fam.kind in PROFILE_KINDS:
        return profile_id(fam.name, spec.key, style), profile_id(fam.name, spec.key)
    cname = plain_id(fam.name, spec.size)
    if length is None:
        return cname, cname
    return screw_id(fam.name, spec.size, length), cname


def expand(cat: Catalogue) -> list[Part]:
    """Expand families into parts. Lengths above max_length_mm are left out."""
    parts: list[Part] = []
    for fam in cat.families.values():
        for spec in fam.sizes.values():
            if fam.length_step:
                lengths: list[float | None] = list(cut_lengths(spec, fam.length_step, cat))
            elif fam.kind in LENGTH_KINDS:
                lengths = [
                    L for L in spec.lengths if cat.max_length_mm is None or cat.max_length_mm >= L
                ]
            else:
                lengths = [None]
            base_enabled = fam.enabled and (spec.enabled if spec.enabled is not None else True)
            styles: list[str | None] = list(spec.styles) or [None]
            for length, style in ((L, st) for L in lengths for st in styles):
                pid, cname = _part_ids(fam, spec, length, style)
                part = Part(
                    id=pid,
                    class_name=cname,
                    family=fam.name,
                    kind=fam.kind,
                    standard=fam.standard,
                    shape=fam.shape,
                    size=spec.size,
                    length=length,
                    pitch=spec.pitch,
                    enabled=base_enabled,
                    common=spec.common and (length is None or length not in spec.uncommon),
                    finishes=fam.finishes_for(spec.key),
                    dims={**spec.dims, **(spec.styles.get(style, {}) if style else {})},
                    dims_source=fam.dims_source,
                    verified=spec.verified if spec.verified is not None else fam.verified,
                    source=fam.source,
                    nominal_length=fam.nominal_length,
                    slot=spec.slot,
                    tnut_kind=fam.tnut_kind,
                    step=_step_for(spec, length),
                    size_key=spec.key,
                    cut_to_length=bool(fam.length_step),
                    style=style,
                )
                for rule in cat.rules:
                    if rule.matches(part):
                        part = _replace_enabled(part, rule.enabled)
                parts.append(part)
    for part in _unknown_parts(cat.unknown):
        for rule in cat.rules:
            if rule.matches(part):
                part = _replace_enabled(part, rule.enabled)
        parts.append(part)
    return parts


def cut_lengths(spec: SizeSpec, step: float, cat: Catalogue) -> list[float]:
    """Lengths of a cut-to-length size: min_length, then the multiples of step above it, up to
    catalogue.rod_max_length_mm (or max_length_mm, or 80 mm). M3, min 20, step 5, max 60:
    20, 25, ..., 60; M2, min 12: 12, 15, 20, ..."""
    hi = cat.cut_max_length_mm or cat.max_length_mm or DEFAULT_CUT_MAX_LENGTH_MM
    lo = spec.min_length
    if lo is None or step <= 0 or lo > hi:
        return []
    out = [lo]
    k = math.floor(lo / step + 1e-9) + 1
    while k * step <= hi + 1e-9:
        out.append(round(k * step, 6))
        k += 1
    return out


def _unknown_parts(spec: UnknownSpec | None) -> list[Part]:
    """Variants of every unknown type, their parameters drawn from the type's ranges with a
    generator seeded by the part id (the same id always gives the same part)."""
    import random

    if spec is None:
        return []
    out = []
    for name, ranges in spec.types.items():
        for n in range(1, spec.variants + 1):
            pid = unknown_id(name, n)
            rng = random.Random(int(hashlib.sha256(pid.encode()).hexdigest()[:12], 16))
            dims = {k: round(rng.uniform(lo, hi), 2) for k, (lo, hi) in ranges.items()}
            dims["seed"] = float(rng.randrange(1_000_000))  # shape details (head style, blob form)
            out.append(
                Part(
                    id=pid,
                    class_name=UNKNOWN,
                    family=UNKNOWN,
                    kind="unknown",
                    standard="-",
                    shape=f"unknown_{name}",
                    size="-",
                    length=None,
                    pitch=None,
                    enabled=spec.enabled and spec.type_enabled.get(name, True),
                    common=True,
                    finishes=spec.type_finishes.get(name, spec.finishes),
                    dims=dims,
                    dims_source="random",
                    verified=False,  # no standard to check against
                    size_key=name,
                )
            )
    return out


def _step_for(spec: SizeSpec, length: float | None) -> dict[str, Any] | None:
    """STEP entry of a part: per-length mapping for screws, a single mapping otherwise."""
    if spec.step is None:
        return None
    if length is None:
        return spec.step
    for key, value in spec.step.items():
        try:
            if float(key) == length:
                return dict(value) if isinstance(value, Mapping) else {"path": value}
        except (TypeError, ValueError):
            continue
    return None


def _replace_enabled(part: Part, enabled: bool) -> Part:
    return replace(part, enabled=enabled)
