"""screwsynth-partgen: build render meshes, collision proxies and keypoints for catalogue parts."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

from screwsynth.catalogue.schema import Catalogue, Part, catalogue_from_config
from screwsynth.common.config import load_config
from screwsynth.common.log import setup_logging
from screwsynth.common.naming import size_value
from screwsynth.common.paths import resolve

from . import assets
from .preview import contact_sheet

log = logging.getLogger("screwsynth.partgen")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


class _Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_config(args.config)
        self.assets_dir = resolve(
            args.assets_dir or self.cfg.get("paths.assets_dir", "data/assets")
        )
        self.catalogue = catalogue_from_config(self.cfg)
        self.settings = assets.Settings.from_config(self.cfg, getattr(args, "lod", None))


def _select(cat: Catalogue, args: argparse.Namespace) -> list[Part]:
    parts = cat.enabled_parts
    if getattr(args, "id", None):
        try:
            return [cat.part(i) for i in args.id]
        except KeyError as exc:
            raise SystemExit(f"unknown part id {exc}") from None
    if getattr(args, "family", None):
        parts = [p for p in parts if p.family == args.family]
    if not (getattr(args, "all", False) or getattr(args, "family", None)):
        raise SystemExit("choose --all, --family F or --id ID")
    return parts


def _summary(results: list[assets.Result]) -> int:
    built = [r for r in results if r.status == "built"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    _out(
        f"\nparts: {len(results)}  built: {len(built)}  skipped: {len(skipped)}  "
        f"failed: {len(failed)}"
    )
    tris = [r.triangles for r in built]
    if tris:
        _out(
            f"triangles: min {min(tris)}  median {int(statistics.median(tris))}  max {max(tris)}"
            f"  ({max(built, key=lambda r: r.triangles).part_id})"
        )
        _out(f"build time: {sum(r.seconds for r in built):.1f} s total")
    for r in failed:
        _out(f"FAILED {r.part_id}: {r.error or '; '.join(r.problems)}")
    return 1 if failed else 0


def cmd_build(ctx: _Ctx, args: argparse.Namespace) -> int:
    parts = _select(ctx.catalogue, args)
    workers = args.workers if args.workers is not None else int(ctx.cfg.get("partgen.workers", 0))
    workers = workers or min(4, os.cpu_count() or 1)
    done = 0

    def progress(r: assets.Result) -> None:
        nonlocal done
        done += 1
        if r.status != "skipped":
            log.info(
                "[%d/%d] %-20s %s %d tris %.2fs",
                done,
                len(parts),
                r.part_id,
                r.status,
                r.triangles,
                r.seconds,
            )

    results = assets.build_many(parts, ctx.assets_dir, ctx.settings, args.force, workers, progress)
    return _summary(results)


def cmd_check(ctx: _Ctx, args: argparse.Namespace) -> int:
    results = [
        assets.recheck_asset(p, ctx.assets_dir, ctx.settings) for p in _select(ctx.catalogue, args)
    ]
    for r in results:
        if r.status == "failed":
            log.error("%s: %s", r.part_id, r.error or "; ".join(r.problems))
    return _summary(results)


def cmd_preview(ctx: _Ctx, args: argparse.Namespace) -> int:
    for pid in args.id:
        folder = assets.asset_dir(ctx.assets_dir, pid)
        mesh, _, meta = assets.load_asset(folder)
        assets.write_preview(folder, mesh, meta, ctx.settings.preview_px)
        _out(f"wrote {folder / assets.PREVIEW}")
    return 0


def contact_sheet_parts(cat: Catalogue) -> list[Part]:
    """One representative part per family: M4 (or the middle size), middle length."""
    chosen = []
    for fam in cat.families.values():
        parts = [p for p in cat.enabled_parts if p.family == fam.name]
        if not parts:
            continue
        by_key: dict[str, list[Part]] = {}
        for p in parts:
            by_key.setdefault(p.size_key, []).append(p)
        keys = sorted(by_key, key=lambda k: (by_key[k][0].slot or 0, size_value(by_key[k][0].size)))
        key = next((k for k in keys if k.endswith("M4")), keys[len(keys) // 2])
        of_size = by_key[key]
        chosen.append(of_size[len(of_size) // 2])
    shapes: set[str] = set()  # unknown parts: one variant of each type
    for p in cat.enabled_parts:
        if p.kind == "unknown" and p.shape not in shapes:
            shapes.add(p.shape)
            chosen.append(p)
    return chosen


def cmd_contact_sheet(ctx: _Ctx, args: argparse.Namespace) -> int:
    from PIL import Image

    parts = contact_sheet_parts(ctx.catalogue)
    results = assets.build_many(parts, ctx.assets_dir, ctx.settings, force=False, workers=1)
    if any(r.status == "failed" for r in results):
        return _summary(results)
    images = [Image.open(assets.asset_dir(ctx.assets_dir, p.id) / assets.PREVIEW) for p in parts]
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(images, out, columns=args.columns)
    _out(f"wrote {out} ({', '.join(p.id for p in parts)})")
    return 0


ITEMS_FORMAT = "screwsynth-items"
ITEMS_VERSION = 1


def items_table(cat: Catalogue, assets_dir: Path) -> dict:
    """What a sorter needs to know about each built part, without the meshes: its class,
    length, volume and main dimensions (items.json; the screwpick sorter's Files tab takes it
    for how full its bins are)."""
    from screwsynth.common.io import read_json

    rows = []
    for p in cat.enabled_parts:
        meta_path = assets.asset_dir(assets_dir, p.id) / assets.PART_JSON
        if not meta_path.is_file():
            continue
        meta = read_json(meta_path)
        mesh = meta.get("mesh", {})
        rows.append(
            {
                "id": p.id,
                "class": meta.get("class_name"),
                "family": meta.get("family"),
                "kind": meta.get("kind"),
                "length": meta.get("length"),
                "volume_mm3": round(float(mesh.get("volume_mm3", 0.0)), 2),
                "bbox_mm": [
                    round(hi - lo, 2)
                    for lo, hi in zip(
                        mesh.get("bbox_min", []), mesh.get("bbox_max", []), strict=False
                    )
                ],
                "dims": meta.get("dims", {}),
            }
        )
    return {
        "format": ITEMS_FORMAT,
        "version": ITEMS_VERSION,
        "catalogue_hash": cat.hash,
        "units": "mm",
        "items": rows,
    }


def cmd_items(ctx: _Ctx, args: argparse.Namespace) -> int:
    from screwsynth.common.io import write_json

    table = items_table(ctx.catalogue, ctx.assets_dir)
    if not table["items"]:
        log.error(
            "no built parts in %s: run `screwsynth-partgen build --all` first", ctx.assets_dir
        )
        return 1
    out = resolve(args.out)
    write_json(out, table)
    missing = len(ctx.catalogue.enabled_parts) - len(table["items"])
    _out(
        f"wrote {out} ({len(table['items'])} items"
        + (f", {missing} not built yet)" if missing else ")")
    )
    return 0


def cmd_clean(ctx: _Ctx, args: argparse.Namespace) -> int:
    keep = {p.id for p in ctx.catalogue.enabled_parts} if args.orphans else None
    removed = assets.clean(Path(ctx.assets_dir), keep)
    _out(f"removed {len(removed)} asset folder(s) from {ctx.assets_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config file (default configs/config.yaml)")
    common.add_argument("--assets-dir", help="output folder (default paths.assets_dir)")
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(prog="screwsynth-partgen", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def selection(sp: argparse.ArgumentParser) -> None:
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--all", action="store_true", help="every enabled part")
        g.add_argument("--family", help="every enabled part of a family")
        g.add_argument("--id", nargs="+", help="part id(s)")

    sp = sub.add_parser("build", parents=[common], help="build assets (incremental)")
    selection(sp)
    sp.add_argument("--lod", choices=["low", "med", "high"], help="default partgen.lod")
    sp.add_argument("--force", action="store_true", help="rebuild even if up to date")
    sp.add_argument("--workers", type=int, help="processes (default partgen.workers, 0 = CPUs)")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("check", parents=[common], help="re-run the checks on existing assets")
    selection(sp)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("preview", parents=[common], help="regenerate preview.png")
    sp.add_argument("--id", nargs="+", required=True)
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("contact-sheet", parents=[common], help="one part per family in one image")
    sp.add_argument("--out", default="tools/partgen/contact_sheet.png")
    sp.add_argument("--columns", type=int, default=5)
    sp.set_defaults(func=cmd_contact_sheet)

    sp = sub.add_parser(
        "items", parents=[common], help="items.json: class, length, volume per part"
    )
    sp.add_argument("--out", default="items.json")
    sp.set_defaults(func=cmd_items)

    sp = sub.add_parser("clean", parents=[common], help="delete generated asset folders")
    sp.add_argument("--orphans", action="store_true", help="only parts no longer enabled")
    sp.set_defaults(func=cmd_clean)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(_Ctx(args), args))


if __name__ == "__main__":
    sys.exit(main())
