"""Capture a real session (tool 09) on the Pi: home/park the bed, capture with the locked settings,
save image + sidecar into data/real/<session>/images."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json
from screwsynth.common.paths import resolve
from screwsynth.realdata.session import load_session

from .printer import MoonrakerPrinter, Printer, PrinterSettings, open_printer

log = logging.getLogger("screwrig.session")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--session", required=True, help="name of a session made with screwsynth-real")
    ap.add_argument("--settings", required=True, help="camera_settings.json from screwrig lock")
    ap.add_argument(
        "--recheck", action="store_true", help="check the camera/plate pose on the board first"
    )
    ap.add_argument("--no-printer", action="store_true", help="do not home/park the bed")
    ap.add_argument("--config")


def printer_from_config(cfg, disabled: bool = False) -> Printer | MoonrakerPrinter | None:
    """The printer to home/park the bed with: through Klipper (Moonraker) when it answers,
    else the serial port (rig.printer.driver); None when disabled or there is none."""
    if disabled:
        return None
    return open_printer(PrinterSettings.from_config(dict(cfg.get("rig.printer", {}) or {})))


def _recheck(cam, printer: Printer | MoonrakerPrinter | None, d: Path, meta: dict) -> None:
    from screwsynth.calibration.cli import main as calib_main

    from .camera import save_capture

    input("Recheck: lay the calibration board flat on the plate, then press Enter > ")
    if printer:
        printer.home_and_park()
    img, md = cam.grab()
    path = save_capture(d, "recheck", img, md, {"session": meta["id"]})
    code = calib_main(["recheck", "--image", str(path)])
    meta["recheck"] = {"time": datetime.now().isoformat(timespec="seconds"), "ok": code == 0}
    write_json(d / "session.json", meta)
    if code != 0:
        raise SystemExit(
            "the camera/plate pose has drifted: recalibrate the extrinsic before capturing"
        )
    input("Recheck OK. Remove the board, put the parts on the plate, then press Enter > ")


def run(args: argparse.Namespace) -> int:
    from .camera import Camera, save_capture

    cfg = load_config(args.config)
    d = resolve(cfg.get("paths.real_dir", "data/real")) / args.session
    load_session(d)
    meta = read_json(d / "session.json")
    settings_path = resolve(args.settings)
    settings = read_json(settings_path)
    printer = printer_from_config(cfg, args.no_printer)
    cam = Camera(settings)  # refuses to run when the locked settings cannot be applied
    try:
        if args.recheck:
            _recheck(cam, printer, d, meta)
        n = len(list((d / "images").glob("img_*.png")))
        print(
            "Enter = home/park the bed and take a photo; type a note before Enter to store it; q = quit"
        )
        while True:
            note = input(f"img_{n + 1:04d} > ").strip()
            if note.lower() == "q":
                break
            if printer:
                printer.home_and_park()
            img, md = cam.grab()
            n += 1
            save_capture(
                d / "images",
                f"img_{n:04d}",
                img,
                md,
                {
                    "session": meta["id"],
                    "settings": settings,
                    "settings_file": str(settings_path),
                    "notes": note,
                    "operator": meta.get("operator", ""),
                },
            )
            print(f"saved img_{n:04d}.png")
    finally:
        cam.close()
        if printer:
            printer.close()
    return 0
