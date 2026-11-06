"""screwsynth-real: sessions, labelling and the real dataset (tool 09). The photos are taken on the sorter: sortering-pnp's Photos tab."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, read_yaml, write_json
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

from . import cvat, matching
from .session import (
    SessionError,
    load_session,
    new_session,
    read_inventory,
    write_inventory_template,
)

log = logging.getLogger("screwsynth.realdata")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


class _Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_config(args.config)
        self.root = resolve(self.cfg.get("paths.real_dir", "data/real"))
        self.calib_path = (
            resolve(self.cfg.get("paths.calibration_dir", "data/calibration")) / "calibration.json"
        )

    def session(self, name: str) -> Path:
        d = self.root / name
        load_session(d)
        return d

    def calibration(self) -> dict | None:
        if self.calib_path.is_file():
            return read_json(self.calib_path)
        log.warning(
            "%s not found: using the nominal camera (calibrate with tool 03)", self.calib_path
        )
        return None

    def nominal(self) -> Any:
        """The nominal camera (configured height) for when there is no calibration."""
        from screwsynth.calibration.geometry import nominal_from_config

        return nominal_from_config(self.cfg)

    def classes(self) -> dict[str, int]:
        names = read_yaml(resolve(self.cfg.get("paths.classes", "configs/classes.yaml")))["names"]
        return {v: int(k) for k, v in names.items()}

    def finishes(self) -> list[str]:
        from screwsynth.catalogue.schema import catalogue_from_config

        return list(catalogue_from_config(self.cfg).finishes)


def cmd_session_new(ctx: _Ctx, args: argparse.Namespace) -> int:
    settings = resolve(args.camera_settings) if args.camera_settings else None
    d = new_session(ctx.root, args.name, args.operator, args.notes, settings, ctx.calib_path)
    _out(
        f"created {d}\nnext: fill in {d / 'inventory.csv'} (what is on the plate) BEFORE capturing"
    )
    return 0


def cmd_inventory_template(ctx: _Ctx, args: argparse.Namespace) -> int:
    d = ctx.session(args.session)
    path = d / "inventory.csv"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists (use --overwrite)")
    write_inventory_template(path, args.parts or [])
    _out(f"wrote {path}")
    return 0


def cmd_preannotate(ctx: _Ctx, args: argparse.Namespace) -> int:
    from .build import prepare_images
    from .preannotate import preannotate

    d = ctx.session(args.session)
    calib = ctx.calibration()
    images = prepare_images(d, calib, ctx.nominal())
    stage1 = args.stage1 or ctx.cfg.get("measure.stage1_weights")
    if not stage1:
        raise SystemExit("no stage-1 weights: pass --stage1 or set measure.stage1_weights")
    stage2 = args.stage2 or ctx.cfg.get("measure.stage2_weights")
    anns = preannotate(images, resolve(stage1), resolve(stage2) if stage2 else None, calib, ctx.cfg)
    write_json(d / "annotations" / "preannotations.json", cvat.to_json(anns))
    _out(
        f"pre-annotated {len(anns)} image(s): next `screwsynth-real export-cvat --session {d.name}`"
    )
    return 0


def cmd_export_cvat(ctx: _Ctx, args: argparse.Namespace) -> int:
    from .build import prepare_images

    d = ctx.session(args.session)
    pre = d / "annotations" / "preannotations.json"
    if pre.is_file():
        anns = cvat.from_json(read_json(pre))
    else:  # no model yet: empty annotations for the cropped images
        import cv2

        anns = []
        for p in prepare_images(d, ctx.calibration(), ctx.nominal()):
            h, w = cv2.imread(str(p)).shape[:2]
            anns.append(cvat.ImageAnn(p.name, w, h, []))
    classes = list(ctx.classes())
    xml = d / "annotations" / "cvat.xml"
    cvat.export_cvat(anns, classes, xml, task=d.name)
    write_json(d / "annotations" / "cvat_labels.json", cvat.cvat_labels(classes))
    _out(
        f"wrote {xml}\n"
        f"CVAT: create a project with the labels from {d / 'annotations' / 'cvat_labels.json'} (Raw), a task "
        f"with the images in {d / 'annotations' / 'images'}, then Upload annotations -> 'CVAT 1.1' -> {xml.name}.\n"
        f"After correcting: Export task dataset -> 'CVAT for images 1.1' and run "
        f"`screwsynth-real import --session {d.name} --xml <exported annotations.xml>`"
    )
    return 0


def cmd_import(ctx: _Ctx, args: argparse.Namespace) -> int:
    d = ctx.session(args.session)
    anns = cvat.import_cvat(resolve(args.xml))
    write_json(d / "annotations" / "corrected.json", cvat.to_json(anns))
    n = sum(len(a.objects) for a in anns)
    kp = sum(1 for a in anns for o in a.objects if o.keypoints)
    _out(
        f"imported {len(anns)} image(s), {n} object(s), {kp} with keypoints: next `screwsynth-real match --session {d.name}`"
    )
    return 0


def _corrected(d: Path) -> list[cvat.ImageAnn]:
    path = d / "annotations" / "corrected.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: import the corrected CVAT annotations first")
    return cvat.from_json(read_json(path))


def cmd_match(ctx: _Ctx, args: argparse.Namespace) -> int:
    from screwsynth.measure.geometry import Geometry

    from .build import measured_lengths, roi_camera

    d = ctx.session(args.session)
    inventory = read_inventory(d / "inventory.csv", ctx.finishes())
    cam, _, _ = roi_camera(ctx.calibration(), ctx.nominal())
    geo = Geometry.from_config(ctx.cfg)
    found = []
    for ann in _corrected(d):
        lengths = measured_lengths(ann, cam, geo)
        screws = [(o.id, o.label, lengths.get(o.id)) for o in ann.objects if o.label in geo]
        found += matching.match_image(ann.name, screws, inventory, args.margin)
    path = d / "matches.csv"
    final = matching.write_matches(path, found, matching.read_matches(path))
    _out(f"{path}: {matching.summary(final)}")
    amb = [m for m in final if m.status in ("ambiguous", "over_count", "not_in_inventory")]
    for m in amb[:30]:
        meas = f"{m.measured_mm:.2f} mm" if m.measured_mm is not None else "no keypoints"
        _out(
            f"  {m.image} #{m.object_id} {m.class_name}: {m.status} ({meas}; candidates {m.candidates})"
        )
    if amb:
        _out(
            "Fill in length_mm and set status to 'manual' for these rows in matches.csv, then build the dataset."
        )
    return 0


def cmd_build_dataset(ctx: _Ctx, args: argparse.Namespace) -> int:
    from screwsynth.catalogue.schema import catalogue_from_config
    from screwsynth.labels import export as ex

    from .build import build_scenes

    calib = ctx.calibration()
    class_ids = ctx.classes()
    catalogue = catalogue_from_config(ctx.cfg)
    scenes = []
    sessions = args.sessions or sorted(
        p.name for p in ctx.root.iterdir() if (p / "annotations" / "corrected.json").is_file()
    )
    for name in sessions:
        d = ctx.session(name)
        inventory = read_inventory(d / "inventory.csv", ctx.finishes())
        matches = matching.read_matches(d / "matches.csv")
        scenes += build_scenes(
            d, _corrected(d), matches, inventory, calib, class_ids, catalogue, ctx.nominal()
        )
    if not scenes:
        raise SystemExit("no labelled images: import corrected annotations first")
    st = ex.Settings.from_config(dict(ctx.cfg.get("labels", {}) or {}), int(ctx.cfg.get("seed", 0)))
    st.split = {
        args.split: 1.0
    }  # a real validation set is ONLY ever val (never mixed into training)
    out = resolve(ctx.cfg.get("paths.datasets_dir", "datasets")) / args.name
    names = {i: n for n, i in class_ids.items()}
    formats = [f.strip() for f in args.formats.split(",")]
    summary = ex.export(scenes, out, names, formats, st)
    _out(
        f"{out}: {len(scenes)} image(s) from {len(sessions)} session(s), split {args.split}; {summary.get('instances')}"
    )
    if args.split == "val":
        _out(
            f"use it with tool 07: screwsynth-labels export ... --val-real {out / 'seg' / 'images' / 'val'}"
        )
    return 0


def cmd_report(ctx: _Ctx, args: argparse.Namespace) -> int:
    from screwsynth.measure.evaluate import detection_metrics, length_accuracy
    from screwsynth.measure.geometry import Geometry
    from screwsynth.measure.infer import Pipeline, Settings

    from . import report

    real = resolve(args.dataset)
    out_dir = resolve(args.out) if args.out else real / "sim_to_real"
    st = Settings.from_config(ctx.cfg.to_dict())
    pipe = Pipeline(
        resolve(args.stage1),
        resolve(args.stage2) if args.stage2 else None,
        Geometry.from_config(ctx.cfg),
        st,
        None,
    )

    def evaluate(ds: Path, split: str) -> dict:
        res = {
            "name": ds.name,
            "detection": detection_metrics(resolve(args.stage1), ds, split, st.stage1_imgsz),
        }
        if (ds / "crops.csv").is_file():
            res["length"] = length_accuracy(pipe, ds, split, out_dir / ds.name)
        return res

    real_eval = evaluate(real, "val")
    synthetic = evaluate(resolve(args.synthetic_dataset), "val") if args.synthetic_dataset else None
    html = None
    realism = {"suggestions": []}
    if args.run:
        html = out_dir / "realism.html"
        out_dir.mkdir(parents=True, exist_ok=True)
        realism = report.realism(
            resolve(ctx.cfg.get("paths.output_dir", "output")) / args.run, real, html
        )
    path = report.write(out_dir, real_eval, synthetic, realism, html)
    _out(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(prog="screwsynth-real", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("session", help="capture sessions")
    ss = sp.add_subparsers(dest="action", required=True)
    s = ss.add_parser(
        "new", parents=[common], help="create data/real/<name> with an inventory template"
    )
    s.add_argument("--name")
    s.add_argument("--operator", default="")
    s.add_argument("--notes", default="")
    s.add_argument("--camera-settings", help="the locked camera_settings.json (fingerprinted)")
    s.set_defaults(func=cmd_session_new)

    sp = sub.add_parser("inventory", help="inventory sheets")
    si = sp.add_subparsers(dest="action", required=True)
    s = si.add_parser("template", parents=[common], help="(re)write inventory.csv")
    s.add_argument("--session", required=True)
    s.add_argument("--parts", nargs="*", help="part ids to pre-fill, e.g. DIN912_M4x16 DIN934_M4")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_inventory_template)

    s = sub.add_parser(
        "preannotate", parents=[common], help="pre-annotate with the current models (tool 08)"
    )
    s.add_argument("--session", required=True)
    s.add_argument("--stage1")
    s.add_argument("--stage2")
    s.set_defaults(func=cmd_preannotate)

    s = sub.add_parser("export-cvat", parents=[common], help="CVAT 1.1 XML + label definitions")
    s.add_argument("--session", required=True)
    s.set_defaults(func=cmd_export_cvat)

    s = sub.add_parser("import", parents=[common], help="import corrected CVAT 1.1 XML")
    s.add_argument("--session", required=True)
    s.add_argument("--xml", required=True)
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("match", parents=[common], help="link labelled screws to inventory lengths")
    s.add_argument("--session", required=True)
    s.add_argument(
        "--margin", type=float, default=0.4, help="share of the gap between lengths accepted"
    )
    s.set_defaults(func=cmd_match)

    s = sub.add_parser(
        "build-dataset", parents=[common], help="labelled sessions -> tool-07 dataset"
    )
    s.add_argument("--name", default="val_real")
    s.add_argument("--sessions", nargs="*", help="default: every session with imported annotations")
    s.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="val: validation set (default); train: a fine-tuning set from OTHER sessions",
    )
    s.add_argument("--formats", default="seg,pose,coco,crops")
    s.set_defaults(func=cmd_build_dataset)

    s = sub.add_parser("report", parents=[common], help="sim-to-real report")
    s.add_argument("--dataset", required=True, help="the real dataset (build-dataset)")
    s.add_argument("--stage1", required=True)
    s.add_argument("--stage2")
    s.add_argument("--synthetic-dataset", help="synthetic dataset for the side-by-side comparison")
    s.add_argument(
        "--run", help="a rendered run (output/<run>) for the image-statistics comparison"
    )
    s.add_argument("--out")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.func(_Ctx(args), args))
    except SessionError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    sys.exit(main())
