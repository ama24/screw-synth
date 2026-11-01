"""screwsynth-catalogue: inspect, validate and edit configs/catalogue.yaml."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from screwsynth.common.config import load_config
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

from . import edit
from .classes import classes_up_to_date, write_classes
from .schema import Catalogue, CatalogueError, Part, load_catalogue, parse_catalogue, parse_rules
from .validate import errors, validate

log = logging.getLogger("screwsynth.catalogue")


# ------------------------------------------------------------------------------------- helpers


def _fmt_len(x: float | None) -> str:
    return "" if x is None else f"{x:g}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


class _Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        cfg = load_config(args.config)
        self.catalogue_path = resolve(args.catalogue or cfg.get("paths.catalogue"))
        self.classes_path = resolve(cfg.get("paths.classes", "configs/classes.yaml"))
        self.max_length = (
            args.max_length if args.max_length is not None else cfg.get("catalogue.max_length_mm")
        )
        self.cut_max_length = cfg.get("catalogue.rod_max_length_mm")
        # config-level selection (dataset recipes); edits never write these into the file
        self.extra_rules = parse_rules(cfg.get("catalogue.extra_rules"), "catalogue.extra_rules")

    def load(self) -> Catalogue:
        cat = load_catalogue(self.catalogue_path, self.max_length, self.cut_max_length)
        cat.rules = [*cat.rules, *self.extra_rules]
        return cat

    def save_checked(self, doc: Any) -> Catalogue:
        """Validate an edited round-trip document and write it only if it has no errors."""
        cat = parse_catalogue(doc, self.max_length, self.cut_max_length)
        errs = errors(validate(cat))
        if errs:
            for issue in errs:
                log.error("%s", issue)
            raise CatalogueError("edit rejected: the catalogue would not validate")
        edit.save_rt(doc, self.catalogue_path)
        log.info("saved %s", self.catalogue_path)
        return cat


def _filter(parts: list[Part], args: argparse.Namespace) -> list[Part]:
    if getattr(args, "family", None):
        parts = [p for p in parts if p.family == args.family]
    if getattr(args, "size", None):
        parts = [p for p in parts if p.size.lower() == args.size.lower()]
    if getattr(args, "enabled_only", False):
        parts = [p for p in parts if p.enabled]
    return parts


# ------------------------------------------------------------------------------------ commands


def cmd_list(ctx: _Ctx, args: argparse.Namespace) -> int:
    parts = _filter(ctx.load().parts, args)
    rows = [
        (
            p.id,
            p.class_name,
            _fmt_len(p.length),
            "yes" if p.enabled else "no",
            "yes" if p.common else "no",
            ",".join(p.finishes),
        )
        for p in parts
    ]
    _out(_table(["id", "class", "length", "enabled", "common", "finishes"], rows))
    _out(f"\n{len(parts)} part(s)")
    return 0


def cmd_stats(ctx: _Ctx, args: argparse.Namespace) -> int:
    cat = ctx.load()
    parts = cat.parts
    enabled = cat.enabled_parts
    classes = cat.classes()

    fam_rows = []
    kinds = {name: fam.kind for name, fam in cat.families.items()}
    if any(p.kind == "unknown" for p in parts):
        kinds["UNKNOWN"] = "unknown"
    for name, kind in kinds.items():
        fp = [p for p in parts if p.family == name]
        fe = [p for p in fp if p.enabled]
        fam_rows.append(
            (
                name,
                kind,
                len(fp),
                len(fe),
                sum(p.common for p in fe),
                len({p.class_name for p in fe}),
            )
        )
    _out(f"Max length: {_fmt_len(cat.max_length_mm) or 'unlimited'} mm\n")
    _out("Per family")
    _out(_table(["family", "kind", "parts", "enabled", "common", "classes"], fam_rows))

    per_class: dict[str, list[Part]] = defaultdict(list)
    for p in enabled:
        per_class[p.class_name].append(p)
    class_rows = []
    for i, cname in enumerate(classes):
        lens = [p.length for p in per_class[cname] if p.length is not None]
        span = f"{min(lens):g}-{max(lens):g}" if lens else ""
        class_rows.append((i, cname, len(per_class[cname]), span))
    _out("\nPer class (enabled)")
    _out(_table(["id", "class", "parts", "lengths"], class_rows))

    finish_counts = Counter(f for p in enabled for f in p.finishes)
    _out("\nPer finish (enabled parts offered in it)")
    _out(_table(["finish", "parts"], [(f, finish_counts.get(f, 0)) for f in cat.finishes]))

    _out(
        f"\nTotal: {len(parts)} parts, {len(enabled)} enabled, {len(classes)} classes, "
        f"{len(cat.finishes)} finishes"
    )
    return 0


def cmd_validate(ctx: _Ctx, args: argparse.Namespace) -> int:
    cat = ctx.load()
    issues = validate(cat)
    for issue in issues:
        (log.error if issue.level == "error" else log.warning)("%s", issue)
    if not classes_up_to_date(cat, ctx.classes_path):
        log.warning("%s is out of date; run `screwsynth-catalogue classes`", ctx.classes_path)
    n_err = len(errors(issues))
    _out(
        f"{len(cat.parts)} parts, {len(cat.classes())} enabled classes: "
        f"{n_err} error(s), {len(issues) - n_err} warning(s)"
    )
    return 1 if n_err else 0


def _set_enabled(ctx: _Ctx, args: argparse.Namespace, enabled: bool) -> int:
    doc = edit.load_rt(ctx.catalogue_path)
    edit.set_enabled(doc, args.pattern, enabled)
    cat = parse_catalogue(doc, max_length_mm=ctx.max_length)
    hit = [p for p in cat.parts if cat.rules[-1].matches(p)]
    if not hit:
        log.error("pattern %r matches no part; nothing changed", args.pattern)
        return 1
    ctx.save_checked(doc)
    _out(f"{'enabled' if enabled else 'disabled'} {len(hit)} part(s) matching {args.pattern!r}")
    _out("Run `screwsynth-catalogue classes` to regenerate classes.yaml.")
    return 0


def cmd_enable(ctx: _Ctx, args: argparse.Namespace) -> int:
    return _set_enabled(ctx, args, True)


def cmd_disable(ctx: _Ctx, args: argparse.Namespace) -> int:
    return _set_enabled(ctx, args, False)


def cmd_add_length(ctx: _Ctx, args: argparse.Namespace) -> int:
    doc = edit.load_rt(ctx.catalogue_path)
    if not edit.add_length(doc, args.family, args.size, args.length, uncommon=args.uncommon):
        _out(f"{args.family} {args.size} already has length {args.length:g}; nothing changed")
        return 0
    ctx.save_checked(doc)
    if ctx.max_length is not None and args.length > ctx.max_length:
        log.warning(
            "length %g exceeds catalogue.max_length_mm = %g; it is stored but not expanded",
            args.length,
            ctx.max_length,
        )
    _out(f"added length {args.length:g} to {args.family} {args.size}")
    return 0


def cmd_add_family(ctx: _Ctx, args: argparse.Namespace) -> int:
    doc = edit.load_rt(ctx.catalogue_path)
    names = edit.add_families(doc, edit.load_rt(args.from_template), replace=args.replace)
    ctx.save_checked(doc)
    _out(f"added famil{'y' if len(names) == 1 else 'ies'}: {', '.join(names)}")
    return 0


def cmd_export_csv(ctx: _Ctx, args: argparse.Namespace) -> int:
    parts = _filter(ctx.load().parts, args)
    dim_keys = sorted({k for p in parts for k in p.dims})
    fields = [
        "id",
        "class_name",
        "family",
        "standard",
        "kind",
        "size",
        "length",
        "pitch",
        "enabled",
        "common",
        "finishes",
        "verified",
        "dims_source",
        *(f"dim_{k}" for k in dim_keys),
    ]
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for p in parts:
            w.writerow(
                [
                    p.id,
                    p.class_name,
                    p.family,
                    p.standard,
                    p.kind,
                    p.size,
                    _fmt_len(p.length),
                    "" if p.pitch is None else f"{p.pitch:g}",
                    p.enabled,
                    p.common,
                    ";".join(p.finishes),
                    p.verified,
                    p.dims_source,
                    *("" if k not in p.dims else f"{p.dims[k]:g}" for k in dim_keys),
                ]
            )
    _out(f"wrote {len(parts)} part(s) to {path}")
    return 0


def cmd_classes(ctx: _Ctx, args: argparse.Namespace) -> int:
    cat = ctx.load()
    errs = errors(validate(cat))
    if errs:
        for issue in errs:
            log.error("%s", issue)
        log.error("fix the catalogue before generating classes")
        return 1
    out = resolve(args.out) if args.out else ctx.classes_path
    doc = write_classes(cat, out)
    _out(f"wrote {doc['nc']} classes to {out}")
    return 0


# -------------------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config file (default configs/config.yaml)")
    common.add_argument("--catalogue", help="catalogue file (default paths.catalogue)")
    common.add_argument(
        "--max-length",
        type=float,
        help="override catalogue.max_length_mm (mm) for this command",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    p = argparse.ArgumentParser(prog="screwsynth-catalogue", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, func: Any, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, parents=[common], help=help_, description=help_)
        sp.set_defaults(func=func)
        return sp

    for name, func, help_ in (
        ("list", cmd_list, "list parts"),
        ("export-csv", cmd_export_csv, "write one CSV row per part"),
    ):
        sp = add(name, func, help_)
        if name == "export-csv":
            sp.add_argument("path")
        sp.add_argument("--family")
        sp.add_argument("--size", help="e.g. M4")
        sp.add_argument("--enabled-only", action="store_true")

    add("stats", cmd_stats, "counts of parts, classes and finishes")
    add("validate", cmd_validate, "check the catalogue; exit code 1 on errors")
    for name, func in (("enable", cmd_enable), ("disable", cmd_disable)):
        sp = add(name, func, f"{name} parts matching a part id, class, family, size or glob")
        sp.add_argument("pattern", help="e.g. DIN912_M4x16, DIN912_M4, DIN985, M2, 'DIN933_M*x7?'")

    sp = add("add-length", cmd_add_length, "add a standard length to a screw size")
    sp.add_argument("family")
    sp.add_argument("size")
    sp.add_argument("length", type=float)
    sp.add_argument("--uncommon", action="store_true", help="also mark it common: false")

    sp = add("add-family", cmd_add_family, "add families from a template YAML file")
    sp.add_argument("--from-template", required=True, metavar="FILE")
    sp.add_argument("--replace", action="store_true", help="overwrite families with the same name")

    sp = add("classes", cmd_classes, "regenerate classes.yaml")
    sp.add_argument("--out", help="output path (default paths.classes)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.func(_Ctx(args), args))
    except (CatalogueError, KeyError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
