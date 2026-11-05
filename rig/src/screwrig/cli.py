"""screwrig: run the rig from the Raspberry Pi.

screwrig lock           lock exposure, gain, white balance and focus -> camera_settings.json
screwrig calib-capture  calibration captures (intrinsics, extrinsic, sensor, single)
screwrig capture        real-data session: home/park the bed, capture, save (tool 09)
screwrig printer        home/park the bed, report the position
screwrig detect         Enter = photo -> segmentation -> result page in the browser
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from screwsynth.common.config import load_config
from screwsynth.common.log import setup_logging

from . import capture, detect, lock, session


def cmd_printer(args: argparse.Namespace) -> int:
    pr = session.printer_from_config(load_config(args.config))
    if pr is None:
        raise SystemExit(
            "no printer: Klipper's Moonraker does not answer at rig.printer.url and no serial "
            "rig.printer.port is set (configs/config.yaml)"
        )
    try:
        if args.action == "park":
            pr.home_and_park()
        print("\n".join(pr.send("M114")))
    finally:
        pr.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="screwrig", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, mod, text in (
        ("lock", lock, "lock the camera settings (run once per rig change)"),
        ("calib-capture", capture, "calibration captures with the locked settings"),
        ("capture", session, "capture a real-data session"),
        ("detect", detect, "photo -> segmentation -> result page, one Enter per photo"),
    ):
        sp = sub.add_parser(name, help=text, description=mod.__doc__)
        mod.add_arguments(sp)
        sp.set_defaults(func=mod.run)
    sp = sub.add_parser("printer", help="home/park the bed (only Y moves are ever sent)")
    sp.add_argument(
        "action",
        choices=["park", "status"],
        help="park: home Y, then park Y if rig.printer.park_y_mm is set",
    )
    sp.add_argument("--config")
    sp.set_defaults(func=cmd_printer)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
