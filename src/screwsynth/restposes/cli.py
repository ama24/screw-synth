"""screwsynth-restposes: precompute how every part rests on the plate."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from screwsynth.catalogue.schema import LENGTH_KINDS, catalogue_from_config
from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve
from screwsynth.partgen.preview import BACKGROUND, contact_sheet, render_view

from . import simulate
from .compute import GENERATOR_VERSION, Settings, compute_poses, rest_pose_json

log = logging.getLogger("screwsynth.restposes")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


class _Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        cfg = load_config(args.config)
        self.cfg = cfg
        self.assets_dir = resolve(cfg.get("paths.assets_dir", "data/assets"))
        self.out_dir = resolve(args.out_dir or cfg.get("paths.rest_poses_dir", "data/rest_poses"))
        self.settings = Settings.from_config(cfg.get("restposes", {}), int(cfg.get("seed", 0)))
        self.drops = int(cfg.get("restposes.validate_drops", 50))
        self.catalogue = catalogue_from_config(cfg)

    def ids(self, args: argparse.Namespace) -> list[str]:
        if getattr(args, "id", None):
            return list(args.id)
        if getattr(args, "family", None):
            return [p.id for p in self.catalogue.enabled_parts if p.family == args.family]
        if getattr(args, "all", False):
            return [p.id for p in self.catalogue.enabled_parts]
        raise SystemExit("choose --all, --family F or --id ID")

    def load_asset(self, pid: str) -> tuple[trimesh.Trimesh, dict[str, Any]]:
        folder = self.assets_dir / pid
        if not (folder / "part.json").is_file():
            raise FileNotFoundError(f"{pid}: no asset; run `screwsynth-partgen build --id {pid}`")
        return trimesh.load_mesh(folder / "render.ply", process=False), read_json(
            folder / "part.json"
        )

    def path(self, pid: str) -> Path:
        return self.out_dir / f"{pid}.json"


def _up_to_date(path: Path, meta: dict, settings: Settings) -> bool:
    if not path.is_file():
        return False
    old = read_json(path)
    return (
        old.get("catalogue_hash") == meta.get("catalogue_hash")
        and old.get("asset_generator_version") == meta.get("generator_version")
        and old.get("generator_version") == GENERATOR_VERSION
        and old.get("method", {}).get("overrides") == [dict(o) for o in settings.overrides]
        and old.get("method", {}).get("min_probability") == settings.min_probability
        and old.get("method", {}).get("com_samples") == settings.com_samples
        and old.get("method", {}).get("measurable_tilt_deg") == settings.measurable_tilt_deg
    )


def cmd_compute(ctx: _Ctx, args: argparse.Namespace) -> int:
    ids = ctx.ids(args)
    n_done = n_skip = n_fail = 0
    for i, pid in enumerate(ids, 1):
        try:
            mesh, meta = ctx.load_asset(pid)
            path = ctx.path(pid)
            if not args.force and not args.validate and _up_to_date(path, meta, ctx.settings):
                n_skip += 1
                continue
            poses = compute_poses(mesh, meta, ctx.settings)
            doc = rest_pose_json(meta, poses, ctx.settings)
            if args.validate:
                cmp = simulate.validate(mesh, poses, meta["shape"], ctx.drops, ctx.settings.seed)
                if cmp is not None:
                    doc["validation"] = cmp.to_json()
                    for d in cmp.disagreements:
                        log.warning("%s: %s", pid, d)
            write_json(path, doc)
            n_done += 1
            log.info(
                "[%d/%d] %s: %d pose(s), top %s %.2f",
                i,
                len(ids),
                pid,
                len(poses),
                poses[0].category,
                poses[0].probability,
            )
        except Exception as exc:  # keep going; report at the end
            n_fail += 1
            log.error("%s: %s", pid, exc)
    _out(f"\nparts: {len(ids)}  computed: {n_done}  skipped: {n_skip}  failed: {n_fail}")
    return 1 if n_fail else 0


def pose_sheet(mesh: trimesh.Trimesh, doc: dict[str, Any], height_px: int = 220) -> Image.Image:
    """One front view per pose, resting on a drawn plate line, with its probability."""
    tiles = []
    posed = []
    for pose in doc["poses"]:
        m = mesh.copy()
        m.apply_transform(np.asarray(pose["transform"]))
        posed.append(m)
    ext = max(float(np.ptp(m.vertices, axis=0)[[0, 2]].max()) for m in posed)
    px = height_px * 0.8 / max(ext, 1e-6)
    for pose, m in zip(doc["poses"], posed, strict=True):
        lo, hi = m.bounds
        centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, ext / 2])
        img = render_view(m, "front", px, (height_px, height_px), centre)
        draw = ImageDraw.Draw(img)
        y_plate = height_px / 2 + (ext / 2) * px
        draw.line([(0, y_plate), (height_px, y_plate)], fill=(60, 110, 200), width=2)
        draw.text((4, 4), f"#{pose['index']} p={pose['probability']:.2f}", fill=(0, 0, 0))
        draw.text((4, 18), f"{pose['category']} tilt {pose['axis_tilt_deg']:.1f}", fill=(0, 0, 0))
        tiles.append(img)
    sheet = Image.new("RGB", (len(tiles) * (height_px + 4) + 4, height_px + 28), BACKGROUND)
    ImageDraw.Draw(sheet).text((4, 4), doc["part_id"], fill=(0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, (4 + i * (height_px + 4), 24))
    return sheet


def cmd_show(ctx: _Ctx, args: argparse.Namespace) -> int:
    for pid in args.id:
        mesh, _ = ctx.load_asset(pid)
        doc = read_json(ctx.path(pid))
        out = ctx.out_dir / f"{pid}.png"
        pose_sheet(mesh, doc).save(out)
        _out(f"wrote {out}")
    return 0


def _family_rows(ctx: _Ctx) -> list[tuple]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for p in ctx.catalogue.enabled_parts:
        path = ctx.path(p.id)
        if path.is_file():
            by_family[p.family].append(read_json(path))
    rows = []
    for fam, docs in by_family.items():
        n_poses = [len(d["poses"]) for d in docs]
        cats: dict[str, float] = defaultdict(float)
        for d in docs:
            for pose in d["poses"]:
                cats[pose["category"]] += pose["probability"] / len(docs)
        top = max(cats, key=cats.get)
        tilts = [d["poses"][0]["axis_tilt_deg"] for d in docs]
        meas = np.mean([sum(p["probability"] for p in d["poses"] if p["measurable"]) for d in docs])
        rows.append(
            (
                fam,
                len(docs),
                f"{np.mean(n_poses):.1f}",
                f"{top} {cats[top]:.2f}",
                f"{min(tilts):.1f}..{max(tilts):.1f}",
                f"{meas:.2f}" if docs[0]["kind"] in LENGTH_KINDS else "-",
            )
        )
    return rows


def cmd_summary(ctx: _Ctx, args: argparse.Namespace) -> int:
    rows = _family_rows(ctx)
    headers = (
        "family",
        "parts",
        "poses/part",
        "main category (mean p)",
        "top-pose tilt",
        "P(measurable)",
    )
    cells = [headers, *[tuple(map(str, r)) for r in rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        _out("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))
        if j == 0:
            _out("  ".join("-" * w for w in widths))
    return 0


def cmd_contact_sheet(ctx: _Ctx, args: argparse.Namespace) -> int:
    from screwsynth.partgen.cli import contact_sheet_parts

    sheets = []
    for part in contact_sheet_parts(ctx.catalogue):
        mesh, _ = ctx.load_asset(part.id)
        sheets.append(pose_sheet(mesh, read_json(ctx.path(part.id))))
    out = resolve(args.out)
    contact_sheet(sheets, out, columns=2, cell=(1100, 260))
    _out(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("--out-dir", help="default paths.rest_poses_dir")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(prog="screwsynth-restposes", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("compute", parents=[common], help="compute rest poses (incremental)")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--family")
    g.add_argument("--id", nargs="+")
    sp.add_argument("--validate", action="store_true", help="also drop-test with pybullet")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_compute)

    sp = sub.add_parser("show", parents=[common], help="image of each pose with its probability")
    sp.add_argument("--id", nargs="+", required=True)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("summary", parents=[common], help="table per family")
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser("contact-sheet", parents=[common], help="poses of one part per family")
    sp.add_argument("--out", default="tools/restposes/contact_sheet.png")
    sp.set_defaults(func=cmd_contact_sheet)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(_Ctx(args), args))


if __name__ == "__main__":
    sys.exit(main())
