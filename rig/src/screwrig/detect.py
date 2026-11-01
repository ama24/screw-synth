"""Live detection on the rig: Enter = photo -> segmentation -> result in the browser.

The model is loaded once. Each Enter counts down, takes a photo, runs screwsynth-infer's pipeline and
writes det_NNNN.jpg, det_NNNN_vis.jpg and det_NNNN_result.json; numbering continues across runs.
A small web page (http://<pi>:8000) always shows the latest result and updates by itself, so the
results can be watched from the PC while working over SSH.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json
from screwsynth.common.paths import resolve

log = logging.getLogger("screwrig.detect")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>screwrig detect</title>
<style>
body{margin:0;background:#111;color:#eee;font:16px system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
#status{font-weight:600}#name{color:#999}
img{display:block;max-width:100%;max-height:calc(100vh - 140px);margin:0 auto}
ul{columns:2;margin:8px 16px 16px;padding-left:20px;color:#ccc}
</style></head><body>
<header><span id="status">waiting for the first photo (press Enter in the terminal)</span>
<span id="name"></span></header>
<img id="img" alt=""><ul id="list"></ul>
<script>
let shown = "";
async function poll() {
  try {
    const r = await fetch("latest.json?" + Date.now());
    if (r.ok) {
      const s = await r.json();
      document.getElementById("status").textContent = s.status;
      if (s.image && s.image !== shown) {
        shown = s.image;
        document.getElementById("img").src = s.image + "?" + Date.now();
        document.getElementById("name").textContent = s.image;
        const ul = document.getElementById("list");
        ul.innerHTML = "";
        for (const o of s.objects || []) {
          const li = document.createElement("li"); li.textContent = o; ul.appendChild(li);
        }
      }
    }
  } catch (e) {}
  setTimeout(poll, 1000);
}
poll();
</script></body></html>
"""


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--stage1",
        help="stage-1 weights (default measure.stage1_weights, else the newest model in models/)",
    )
    ap.add_argument("--stage2", help="keypoint weights (default measure.stage2_weights)")
    ap.add_argument("--out", default="data/real/detect", help="folder for photos and results")
    ap.add_argument(
        "--settings", help="camera_settings.json from screwrig lock (default: automatic camera)"
    )
    ap.add_argument("--calibration", help="calibration.json (default paths.calibration_dir)")
    ap.add_argument("--nominal-camera", action="store_true")
    ap.add_argument("--countdown", type=int, default=3, help="seconds before each photo")
    ap.add_argument("--port", type=int, default=8000, help="port of the result page")
    ap.add_argument("--config")


def find_model(models_dir: Path) -> Path | None:
    """Newest model under models/: an NCNN export folder if there is one (fastest on a Pi), else best.pt."""
    if not models_dir.is_dir():
        return None
    for pattern in ("*_ncnn_model", "best.pt"):
        found = [p for p in models_dir.rglob(pattern) if p.exists()]
        if found:
            return max(found, key=lambda p: p.stat().st_mtime)
    return None


def next_number(out: Path) -> int:
    """1 + the highest det_NNNN already in the folder (so a rerun never overwrites)."""
    nums = [int(m.group(1)) for p in out.glob("det_*") if (m := re.match(r"det_(\d+)", p.name))]
    return max(nums, default=0) + 1


def summary_lines(result: dict[str, Any]) -> list[str]:
    """One line per object: class, length (when measured) and score."""
    lines = []
    for o in result["objects"]:
        text = o["class"]
        length = o.get("length") or {}
        if length.get("length_mm") is not None:
            text += f" {length['length_mm']:g} mm"
        elif length.get("length_raw_mm") is not None:
            text += f" ~{length['length_raw_mm']:.1f} mm"
        lines.append(f"{text} ({o['score']:.2f})")
    return lines


def local_address() -> str:
    """The Pi's address on the local network (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname() + ".local"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base class signature
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(out: Path, port: int) -> http.server.ThreadingHTTPServer:
    """Serve the result folder in the background; tries the next ports if one is taken."""
    handler = functools.partial(_QuietHandler, directory=str(out))
    for p in range(port, port + 10):
        try:
            server = http.server.ThreadingHTTPServer(("", p), handler)
        except OSError:
            continue
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    raise SystemExit(f"ports {port}-{port + 9} are all taken")


def _status(
    out: Path, status: str, image: str | None = None, objects: list[str] | None = None
) -> None:
    latest = read_json(out / "latest.json") if (out / "latest.json").is_file() else {}
    latest["status"] = status
    if image is not None:
        latest["image"], latest["objects"] = image, objects or []
    write_json(out / "latest.json", latest)


class _Snapper:
    """Locked camera (screwrig lock settings) when given, otherwise rpicam-still with auto settings."""

    def __init__(self, settings: str | None) -> None:
        self.cam = None
        if settings:
            from .camera import Camera

            self.cam = Camera(read_json(resolve(settings)))

    def snap(self, path: Path) -> None:
        if self.cam is not None:
            import cv2

            img, _ = self.cam.grab()
            if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"could not write {path}")
            return
        cmd = ["rpicam-still", "-n", "-t", "500", "--width", "4608", "--height", "2592"]
        subprocess.run([*cmd, "-q", "95", "-o", str(path)], check=True, capture_output=True)

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close()


def torch_problem(version: str, machine: str) -> str | None:
    """Why this PyTorch build will crash here, or None. The PyPI build for 64-bit ARM is the NVIDIA
    (CUDA) one; on a Raspberry Pi it stops with "Illegal instruction" at the first tensor operation."""
    if machine.lower() in ("aarch64", "arm64") and "+cu" in version:
        return (
            f"PyTorch {version} is the NVIDIA (CUDA) build, which stops a Raspberry Pi with "
            '"Illegal instruction". Install the CPU build:\n'
            "  pip uninstall -y torch torchvision\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    return None


def run(args: argparse.Namespace) -> int:
    import platform

    import cv2
    import torch

    problem = torch_problem(torch.__version__, platform.machine())
    if problem:
        raise SystemExit(problem)

    from screwsynth.calibration.geometry import nominal_from_config
    from screwsynth.measure.cli import _calibration
    from screwsynth.measure.geometry import Geometry
    from screwsynth.measure.infer import Pipeline, Settings, draw

    cfg = load_config(args.config)
    stage1 = args.stage1 or cfg.get("measure.stage1_weights") or find_model(resolve("models"))
    if not stage1:
        raise SystemExit("no model: copy one into models/ or pass --stage1")
    stage2 = args.stage2 or cfg.get("measure.stage2_weights")
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(PAGE, encoding="utf-8")
    _status(out, "loading the model...")
    server = serve(out, args.port)
    url = f"http://{local_address()}:{server.server_address[1]}/"
    print(f"Results page (open it in the PC's browser): {url}", flush=True)
    print(f"Loading {stage1} ...", flush=True)
    pipe = Pipeline(
        resolve(stage1),
        None if stage2 in (None, "none") else resolve(stage2),
        Geometry.from_config(cfg),
        Settings.from_config(cfg.to_dict()),
        _calibration(cfg, args),
        resolve(cfg.get("paths.classes", "configs/classes.yaml")),
        nominal=nominal_from_config(cfg),
    )
    snapper = _Snapper(args.settings)
    _status(out, "ready: press Enter in the terminal")
    try:
        while True:
            n = next_number(out)
            if input(f"\nEnter = photo det_{n:04d}, q = quit > ").strip().lower() == "q":
                break
            for i in range(args.countdown, 0, -1):
                print(f"photo in {i}...", flush=True)
                time.sleep(1)
            photo = out / f"det_{n:04d}.jpg"
            _status(out, f"taking det_{n:04d}...")
            try:
                snapper.snap(photo)
                print("photo taken, you can let go. Analysing...", flush=True)
                _status(out, f"analysing det_{n:04d}...")
                result, scene, _ = pipe.run(photo)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                log.error("det_%04d failed: %s", n, exc)
                _status(out, f"det_{n:04d} failed: {exc}")
                continue  # one bad photo does not end the session
            write_json(out / f"det_{n:04d}_result.json", result)
            vis = f"det_{n:04d}_vis.jpg"
            cv2.imwrite(str(out / vis), draw(scene, result))
            lines = summary_lines(result)
            took = result["timing_s"]["total"]
            _status(out, f"det_{n:04d}: {len(lines)} object(s), {took:.1f} s", vis, lines)
            print(f"det_{n:04d}: {len(lines)} object(s) in {took:.1f} s -> {url}")
            for line in lines:
                print("  " + line)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        snapper.close()
        server.shutdown()
    print(f"photos and results are in {out}")
    return 0
