"""Semantic validation of a parsed catalogue."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from screwsynth.common.naming import parse_part_id, size_value

from .schema import (
    LENGTH_KINDS,
    MATERIALS,
    NOMINAL_LENGTH_MODES,
    REQUIRED_DIMS,
    SHAPE_KIND,
    Catalogue,
    Family,
    SizeSpec,
)

COS30 = math.cos(math.radians(30))


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7s} [{self.code}] {self.where}: {self.message}"


class _Collector:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def error(self, code: str, where: str, message: str) -> None:
        self.issues.append(Issue("error", code, where, message))

    def warning(self, code: str, where: str, message: str) -> None:
        self.issues.append(Issue("warning", code, where, message))

    def check(self, ok: bool, code: str, where: str, message: str) -> None:
        if not ok:
            self.error(code, where, message)


def validate(cat: Catalogue) -> list[Issue]:
    """Return all problems found; the catalogue is valid when no issue has level "error"."""
    c = _Collector()
    _check_finish_defs(cat, c)
    for fam in cat.families.values():
        _check_family(cat, fam, c)
        for spec in fam.sizes.values():
            _check_size(fam, spec, c)
    _check_parts(cat, c)
    _check_rules(cat, c)
    return c.issues


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == "error"]


def _check_finish_defs(cat: Catalogue, c: _Collector) -> None:
    for f in cat.finishes.values():
        c.check(
            f.material in MATERIALS,
            "finish-material",
            f"finishes.{f.id}",
            f"material {f.material!r} not in {MATERIALS}",
        )


def _check_family(cat: Catalogue, fam: Family, c: _Collector) -> None:
    where = f"families.{fam.name}"
    if fam.shape in SHAPE_KIND:
        c.check(
            SHAPE_KIND[fam.shape] == fam.kind,
            "shape-kind",
            where,
            f"shape {fam.shape!r} belongs to kind {SHAPE_KIND[fam.shape]!r}, not {fam.kind!r}",
        )
    if fam.kind in LENGTH_KINDS:
        c.check(
            fam.nominal_length in NOMINAL_LENGTH_MODES,
            "nominal-length",
            where,
            f"{fam.kind}s need nominal_length in {NOMINAL_LENGTH_MODES}",
        )
    if not fam.sizes:
        c.error("no-sizes", where, "family has no sizes")
    if not fam.finishes:
        c.error("no-finishes", where, "family lists no finishes")
    for finish, keys in fam.finishes.items():
        if finish not in cat.finishes:
            c.error("unknown-finish", f"{where}.finishes", f"unknown finish {finish!r}")
        for key in keys or ():
            if key not in fam.sizes:
                c.error(
                    "finish-size",
                    f"{where}.finishes.{finish}",
                    f"size {key!r} is not in the family's size table",
                )


def _check_size(fam: Family, spec: SizeSpec, c: _Collector) -> None:
    where = f"families.{fam.name}.sizes.{spec.key}"
    d = spec.dims
    for name, value in d.items():
        c.check(value > 0, "dim-positive", where, f"{name} must be > 0, got {value}")
    if fam.source == "parametric":
        missing = [k for k in REQUIRED_DIMS[fam.shape] if k not in d]
        if missing:
            c.error("missing-dim", where, f"missing dimension(s) {missing} for shape {fam.shape}")
            return
    if "d" in d:
        c.check(
            abs(d["d"] - size_value(spec.size)) < 1e-9,
            "d-mismatch",
            where,
            f"d = {d['d']} does not match size {spec.size}",
        )

    if fam.kind in ("screw", "standoff", "nut", "tnut"):
        if spec.pitch is None:
            c.error("pitch", where, "threaded parts need a pitch")
        else:
            c.check(
                0 < spec.pitch < 0.3 * size_value(spec.size),
                "pitch",
                where,
                f"pitch {spec.pitch} is implausible for {spec.size}",
            )

    if fam.length_step is not None:
        c.check(fam.length_step > 0, "length-step", where, "length_step must be > 0")
        c.check(
            spec.min_length is not None and spec.min_length > 0,
            "min-length",
            where,
            "cut-to-length sizes need min_length > 0",
        )
        c.check(
            not spec.lengths,
            "length",
            where,
            "cut-to-length sizes have no lengths list (min_length + length_step instead)",
        )
    elif fam.kind in LENGTH_KINDS:
        if not spec.lengths:
            c.error("no-lengths", where, f"{fam.kind} size has no lengths")
        c.check(all(L > 0 for L in spec.lengths), "length", where, "lengths must be > 0")
        extra = sorted(set(spec.uncommon) - set(spec.lengths))
        c.check(not extra, "uncommon", where, f"uncommon lengths {extra} are not in lengths")
    elif spec.lengths or spec.uncommon:
        c.error("length", where, f"{fam.kind} parts have no lengths")
    if spec.min_length is not None and fam.length_step is None:
        c.error("min-length", where, "min_length needs the family's length_step (cut to length)")

    if fam.source == "step":
        _check_step(fam, spec, where, c)
    else:
        _check_shape(fam.shape, d, where, c)


def _check_shape(shape: str, d: dict[str, float], where: str, c: _Collector) -> None:
    def sane(ok: bool, msg: str) -> None:
        c.check(ok, "dim-sanity", where, msg)

    if shape in ("socket_cap", "button", "countersunk"):
        sane(d["dk"] > d["d"], "head diameter dk must be > d")
        sane(d["s"] < d["dk"], "socket s must be < head diameter dk")
        sane(d["t"] < d["k"], "socket depth t must be < head height k")
    if shape == "socket_cap":
        sane(0.9 * d["d"] <= d["ds"] <= d["d"] * 1.001, "shank ds must be close to d")
        sane(d["k"] <= d["d"] * 1.001, "head height k must be <= d")
    if shape == "countersunk":
        sane(60 <= d["angle"] <= 120, "countersink angle must be within 60..120 deg")
        cone = (d["dk"] - d["d"]) / 2 / math.tan(math.radians(d["angle"] / 2))
        sane(d["k"] >= 0.9 * cone, f"head height k must be >= cone height {cone:.2f}")
    if shape in ("hex", "hex_nut", "nyloc_nut"):
        sane(d["s"] > d["d"], "across flats s must be > d")
        sane(d["s"] < d["e"] <= d["s"] / COS30 + 0.01, "across corners e must be in (s, s/cos30]")
    if shape == "hex":
        sane(d["k"] < d["d"], "hex head height k must be < d")
        if "dw" in d:
            sane(d["d"] < d["dw"] <= d["s"], "washer face dw must be in (d, s]")
    if shape == "nyloc_nut":
        sane(d["h"] > d["m"], "total height h must be > metal height m")
    if shape == "washer":
        sane(d["d2"] > d["d1"], "outer diameter d2 must be > inner d1")
        sane(d["h"] < d["d2"] - d["d1"], "washer thickness h is implausibly large")
    if shape == "set_screw":
        sane(d["s"] < 0.75 * d["d"], "socket s must leave a wall to the thread (s < 0.75 d)")
        sane(d["t"] < 2 * d["d"], "socket depth t is implausibly deep")
    if shape.startswith("standoff_"):
        sane(d["s"] > d["d"], "across flats s must be > d")
        sane(d["s"] < d["e"] <= d["s"] / COS30 + 0.01, "across corners e must be in (s, s/cos30]")
        for key in ("ls", "lf"):
            if key in d:
                sane(0.5 * d["d"] <= d[key] <= 5 * d["d"], f"{key} is implausible for d = {d['d']}")
    if shape == "corner_bracket":
        sane(d["t"] < d["a"] / 3, "wall t must be < a / 3")
        sane(2 * d["ts"] + d["dh"] < d["w"], "the hole dh must fit between the side walls")
        sane(d["dh"] + d["slot"] < d["a"] - d["t"], "the hole/slot must fit on the leg")
        sane(d["key_w"] < d["w"] and d["key_h"] < d["t"], "the alignment key must be small")
    if shape in ("tnut_slide", "tnut_drop"):
        sane(d["neck_width"] > d["d"] * 0.9, "neck_width must leave room for the thread")
        sane(d["neck_width"] <= d["width"], "neck_width must be <= width")
        sane(d["neck_height"] < d["height"], "neck_height must be < height")
        sane(d["length"] > d["d"], "length must be > thread diameter")


def _check_step(fam: Family, spec: SizeSpec, where: str, c: _Collector) -> None:
    if spec.step is None:
        c.error("step-source", where, "source: step needs a 'step' entry")
        return
    entries = spec.step.items() if fam.kind in LENGTH_KINDS else [(spec.key, spec.step)]
    if fam.kind in LENGTH_KINDS:
        keys = set()
        for k in spec.step:
            try:
                keys.add(float(k))
            except (TypeError, ValueError):
                c.error("step-source", where, f"screw step entries are keyed by length, got {k!r}")
        missing = sorted(set(spec.lengths) - keys)
        c.check(not missing, "step-source", where, f"no step entry for length(s) {missing}")
    for key, entry in entries:
        if not isinstance(entry, dict) or not entry.get("path"):
            c.error("step-source", f"{where}.step.{key}", "needs a 'path' to the STEP file")
            continue
        if fam.kind in LENGTH_KINDS:
            kps = entry.get("keypoints") or {}
            ok = all(
                isinstance(kps.get(n), (list, tuple)) and len(kps[n]) == 3
                for n in ("head_top", "bearing", "tip")
            )
            c.check(
                ok,
                "step-source",
                f"{where}.step.{key}",
                "screws from STEP need keypoints head_top, bearing, tip as [x, y, z] (mm)",
            )


def _check_parts(cat: Catalogue, c: _Collector) -> None:
    counts = Counter(p.id for p in cat.parts)
    for pid, n in counts.items():
        if n > 1:
            c.error("duplicate-id", pid, f"part id appears {n} times")

    by_class: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for p in cat.parts:
        try:
            parsed = parse_part_id(p.id)
        except ValueError as exc:
            c.error("part-id", p.id, str(exc))
            continue
        c.check(parsed.class_name == p.class_name, "class-mismatch", p.id, "class name mismatch")
        by_class[p.class_name].add((p.family, p.kind, p.size))
        if not p.finishes:
            c.error("no-finishes", p.id, "part has no finishes")
        if p.kind == "screw" and p.length is not None:
            d = p.dims
            if p.fully_threaded is False:
                c.check(
                    d["b"] <= p.length,
                    "thread-length",
                    p.id,
                    f"thread length b = {d['b']} exceeds length {p.length}",
                )
            if p.shape == "countersunk" and "k" in d:
                c.check(p.length > d["k"], "length", p.id, "length must exceed head height k")
    for cname, keys in by_class.items():
        if len(keys) > 1:
            c.error(
                "class-mismatch", cname, f"parts of one class differ in family/kind/size: {keys}"
            )

    if not cat.classes():
        c.warning("no-classes", "catalogue", "no part is enabled")


def _check_rules(cat: Catalogue, c: _Collector) -> None:
    for i, rule in enumerate(cat.rules):
        if not any(rule.matches(p) for p in cat.parts):
            c.warning("rule-unmatched", f"rules[{i}]", f"pattern {rule.match!r} matches no part")
