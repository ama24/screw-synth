"""rig/requirements-pi.txt pins every package for the Pi (exact version + hash)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pi_requirements_are_pinned_with_hashes():
    text = (ROOT / "rig" / "requirements-pi.txt").read_text(encoding="utf-8")
    pattern = r"^([a-z0-9_.-]+)==(\S+) \\\n((?:\s+--hash=sha256:[0-9a-f]{64})+)"
    reqs = re.findall(pattern, text, re.M)
    body = [ln for ln in text.splitlines() if ln and not ln.startswith(("#", "--", " "))]
    assert len(body) == len(reqs), "every requirement needs ==version and --hash"
    versions = {n: v for n, v, _ in reqs}
    assert {"torch", "torchvision", "ultralytics", "ncnn", "numpy", "pyserial"} <= set(versions)
    assert "opencv-contrib-python-headless" in versions and "opencv-python" not in versions
    assert versions["torch"].endswith("+cpu")  # the build that runs on a Pi
