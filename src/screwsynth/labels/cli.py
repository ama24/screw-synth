"""screwsynth-labels: export processed scenes to YOLO/COCO datasets, preview, check, stats, merge."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

from . import export as ex
from . import inspect as insp

log = logging.getLogger("screwsynth.labels")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def cmd_export(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    st = ex.Settings.from_config(
        cfg.get("labels", {}), int(cfg.get("seed", 0) if args.seed is None else args.seed)
    )
    if args.imgsz:
        st.imgsz = args.imgsz
    if args.val_real:
        st.val_real = args.val_real
    out_root = resolve(cfg.get("paths.output_dir", "output"))
    scenes = []
    for run in args.run:
        proc = out_root / run / "processed"
        if not proc.is_dir():
            raise SystemExit(f"{proc} not found: run `screwsynth-post run --run {run}` first")
        scenes += sorted(d for d in proc.iterdir() if (d / "meta.json").is_file())
    if not scenes:
        raise SystemExit("no processed scenes")
    formats = (
        [f.strip() for f in args.formats.split(",")] if args.formats else list(ex.DEFAULT_FORMATS)
    )
    bad = set(formats) - set(ex.FORMATS)
    if bad:
        raise SystemExit(f"unknown formats {sorted(bad)}; choose from {ex.FORMATS}")
    classes_path = resolve(cfg.get("paths.classes", "configs/classes.yaml"))
    names = ex.load_names(classes_path)
    out = resolve(args.out)
    summary = ex.export(scenes, out, names, formats, st, package_files(cfg, classes_path))
    _out(f"exported {len(scenes)} scenes -> {out} ({', '.join(formats)})")
    _print_stats(summary)
    return 0


def package_files(cfg: Any, classes_path: Path) -> dict[str, str]:
    """classes.yaml and parts.json for the model package: a sorter (sortering-pnp) takes the class
    list and the measuring table from the model it runs."""
    from screwsynth.measure.geometry import parts_table

    files = {"classes.yaml": classes_path.read_text(encoding="utf-8")}
    try:
        files["parts.json"] = json.dumps(parts_table(cfg), indent=1) + "\n"
    except Exception as exc:  # noqa: BLE001 - the dataset is still fine without it
        log.warning("no parts.json in the dataset: %s", exc)
    return files


def _print_stats(s: dict) -> None:
    _out(f"images {s.get('images')}, instances {s.get('instances')}, dropped {s.get('dropped', 0)}")
    low = s.get("classes_below_min") or []
    if low:
        _out(
            f"{len(low)} class(es) below {s.get('min_instances_per_class')} instances: {', '.join(low[:12])}"
            f"{' ...' if len(low) > 12 else ''}"
        )


def cmd_stats(args: argparse.Namespace) -> int:
    s = read_json(resolve(args.dataset) / "stats.json")
    if args.json:
        _out(json.dumps(s, indent=2))
        return 0
    _print_stats(s)
    for key in ("per_class", "per_finish", "per_size"):
        _out(f"\n{key}:")
        for k, v in (s.get(key) or {}).items():
            _out(f"  {k:24s} {v}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    ds = resolve(args.dataset)
    out = insp.preview(ds, args.n, ds / f"preview_{args.split}.jpg", args.split)
    _out(f"wrote {out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    problems = insp.check(resolve(args.dataset))
    for p in problems[:200]:
        _out(p)
    _out(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_merge(args: argparse.Namespace) -> int:
    ex.merge([resolve(d) for d in args.datasets], resolve(args.out))
    _out(f"merged into {resolve(args.out)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(prog="screwsynth-labels", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("export", parents=[common], help="processed scenes -> datasets")
    sp.add_argument("--run", nargs="+", required=True, help="one or more output/<run> names")
    sp.add_argument("--out", required=True, help="e.g. datasets/run01")
    sp.add_argument(
        "--formats", help=f"comma list of {','.join(ex.FORMATS)} (default seg,pose,coco)"
    )
    sp.add_argument("--imgsz", type=int, help="stage-1 image size (default labels.stage1_imgsz)")
    sp.add_argument("--seed", type=int, help="split seed (default config seed)")
    sp.add_argument(
        "--val-real", help="path of the real validation set (images folder) for *_real.yaml"
    )
    sp.set_defaults(func=cmd_export)
    sp = sub.add_parser("preview", parents=[common], help="contact sheet with drawn labels")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--split", default="val")
    sp.set_defaults(func=cmd_preview)
    sp = sub.add_parser("check", parents=[common], help="validate every label file")
    sp.add_argument("--dataset", required=True)
    sp.set_defaults(func=cmd_check)
    sp = sub.add_parser("stats", parents=[common], help="instances per class/finish/size/length")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_stats)
    sp = sub.add_parser("merge", parents=[common], help="combine exported datasets")
    sp.add_argument("datasets", nargs="+")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_merge)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
