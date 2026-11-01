"""screwsynth-infer / screwsynth-eval / screwsynth-compare-measure: photo -> parts, sizes and
lengths, and their evaluation (tool 08, measurement side)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from screwsynth.calibration.geometry import nominal_from_config
from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json, write_text
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

log = logging.getLogger("screwsynth.measure")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def show_image(path: Path) -> bool:
    """Open an image in the system's default viewer; False when there is no screen (e.g. over SSH)."""
    import os
    import shutil
    import subprocess

    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
        return True
    if sys.platform == "darwin":
        opener = "open"
    else:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
        opener = "xdg-open"
    if shutil.which(opener) is None:
        return False
    subprocess.Popen(  # the viewer outlives this command
        [opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return True


def _calibration(cfg, args: argparse.Namespace) -> dict | None:
    if getattr(args, "nominal_camera", False):
        return None
    path = resolve(
        getattr(args, "calibration", None)
        or Path(cfg.get("paths.calibration_dir", "data/calibration")) / "calibration.json"
    )
    if not path.is_file():
        log.warning("%s not found: using the nominal camera (calibrate with tool 03)", path)
        return None
    return read_json(path)


# --------------------------------------------------------------------------- screwsynth-infer


def infer_main(argv: Sequence[str] | None = None) -> int:
    import cv2

    from .geometry import Geometry
    from .infer import Pipeline, Settings, draw

    p = argparse.ArgumentParser(
        prog="screwsynth-infer", description="photo -> parts, sizes, lengths (JSON)"
    )
    p.add_argument("--image", required=True, nargs="+", help="full frame or ROI image(s)")
    p.add_argument("--stage1", help="stage-1 weights (default measure.stage1_weights)")
    p.add_argument(
        "--stage2",
        help="stage-2 keypoint weights (default measure.stage2_weights; 'none' = classical only)",
    )
    p.add_argument("--calibration", help="calibration.json (default paths.calibration_dir)")
    p.add_argument("--nominal-camera", action="store_true")
    p.add_argument("--method", choices=["auto", "keypoints", "classical"])
    p.add_argument("--out", help="output folder (default next to the image)")
    p.add_argument("--save-vis", action="store_true", help="also write <image>_vis.jpg")
    p.add_argument(
        "--show", action="store_true", help="write <image>_vis.jpg and open it in the image viewer"
    )
    p.add_argument("--config")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    cfg = load_config(args.config)
    st = Settings.from_config(cfg.to_dict())
    if args.method:
        st.method = args.method
    stage1 = args.stage1 or cfg.get("measure.stage1_weights")
    if not stage1:
        raise SystemExit("no stage-1 weights: pass --stage1 or set measure.stage1_weights")
    stage2 = args.stage2 or cfg.get("measure.stage2_weights")
    stage2 = None if stage2 in (None, "none") else resolve(stage2)
    pipe = Pipeline(
        resolve(stage1),
        stage2,
        Geometry.from_config(cfg),
        st,
        _calibration(cfg, args),
        resolve(cfg.get("paths.classes", "configs/classes.yaml")),
        nominal=nominal_from_config(cfg),
    )
    for img in args.image:
        path = resolve(img)
        result, scene, _ = pipe.run(path)
        out_dir = resolve(args.out) if args.out else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / f"{path.stem}_result.json", result)
        vis = out_dir / f"{path.stem}_vis.jpg"
        if args.save_vis or args.show:
            cv2.imwrite(str(vis), draw(scene, result))
        if args.show and not show_image(vis):
            log.warning("no screen to show %s on (over SSH?): open it on the Pi's desktop", vis)
        n = len(result["objects"])
        measured = [o for o in result["objects"] if o.get("length")]
        _out(
            f"{path.name}: {n} object(s), {len(measured)} measured, {result['timing_s']['total']:.2f} s -> {out_dir}"
        )
    return 0


# ---------------------------------------------------------------------------- screwsynth-eval


def eval_main(argv: Sequence[str] | None = None) -> int:
    from .evaluate import detection_metrics, length_accuracy, write_report
    from .geometry import Geometry
    from .infer import Pipeline, Settings

    p = argparse.ArgumentParser(
        prog="screwsynth-eval", description="mAP, size confusion, length accuracy"
    )
    p.add_argument(
        "--dataset", required=True, help="tool-07 dataset (synthetic) or tool-09 real set"
    )
    p.add_argument("--stage1", required=True)
    p.add_argument("--stage2", help="keypoint weights; omit for classical measurement only")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--compare-with", help="a second dataset (e.g. synthetic val) for the sim-to-real gap"
    )
    p.add_argument("--out", help="report folder (default <dataset>/eval_<weights name>)")
    p.add_argument("--config")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    cfg = load_config(args.config)
    st = Settings.from_config(cfg.to_dict())
    geo = Geometry.from_config(cfg)
    pipe = Pipeline(
        resolve(args.stage1), resolve(args.stage2) if args.stage2 else None, geo, st, None
    )

    def evaluate(ds: Path) -> dict:
        out = {
            "name": ds.name,
            "detection": detection_metrics(resolve(args.stage1), ds, args.split, st.stage1_imgsz),
        }
        if (ds / "crops.csv").is_file():
            out["length"] = length_accuracy(pipe, ds, args.split, out_dir / ds.name)
        return out

    ds = resolve(args.dataset)
    out_dir = resolve(args.out) if args.out else ds / f"eval_{Path(args.stage1).stem}"
    res = evaluate(ds)
    other = evaluate(resolve(args.compare_with)) if args.compare_with else None
    write_report(out_dir, ds.name, res, other)
    _out((out_dir / "report.md").read_text(encoding="utf-8"))
    return 0


# ------------------------------------------------------------------ screwsynth-compare-measure


def cmd_compare(args: argparse.Namespace) -> int:
    from .evaluate import compare_measurement, comparison_markdown
    from .geometry import Geometry
    from .infer import Settings

    cfg = load_config(args.config)
    ds = resolve(args.dataset)
    res = compare_measurement(
        ds,
        Geometry.from_config(cfg),
        args.split,
        resolve(args.stage2) if args.stage2 else None,
        Settings.from_config(cfg.to_dict()),
    )
    out = resolve(args.out) if args.out else ds / "measurement_comparison"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison.json", res)
    md = comparison_markdown(res)
    write_text(out / "comparison.md", md)
    _out(md)
    return 0


def compare_main(argv: Sequence[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(
        prog="screwsynth-compare-measure",
        description="method A (keypoints) vs B (classical) on the true masks",
        parents=[common],
    )
    p.add_argument(
        "--dataset", required=True, help="tool-07 dataset exported with the crops format"
    )
    p.add_argument(
        "--stage2", help="keypoint model weights (otherwise only A with the true keypoints)"
    )
    p.add_argument("--split", default="val", help="train | val | all")
    p.add_argument("--out")
    args = p.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return cmd_compare(args)
