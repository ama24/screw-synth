import argparse
import json

import cv2
import numpy as np
import pytest

from screwrig import camera, cli
from screwrig import printer as printer_mod
from screwrig.printer import (
    MoonrakerPrinter,
    Printer,
    PrinterError,
    PrinterSettings,
    allowed,
    open_printer,
    park_commands,
)
from screwsynth.common.io import read_json


def test_pi_focus():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (200, 300)).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2)
    assert camera.focus_metric(sharp) > camera.focus_metric(blurred)
    pos = [2.8, 2.9, 3.0, 3.1, 3.2, 3.3]
    scores = [-((p - 3.07) ** 2) for p in pos]
    assert camera.best_focus(pos, scores) == pytest.approx(3.07, abs=1e-6)


def test_pi_settings_check():
    locked = {
        "ExposureTime": 10000,
        "AnalogueGain": 1.5,
        "ColourGains": [2.0, 1.6],
        "LensPosition": 3.3,
        "size": [4608, 2592],
    }
    good = {
        "ExposureTime": 10020,
        "AnalogueGain": 1.51,
        "ColourGains": (2.0, 1.605),
        "LensPosition": 3.3,
        "size": [4608, 2592],
    }
    assert camera.settings_mismatch(locked, good) == []
    bad = dict(good, ExposureTime=12000, ColourGains=(2.3, 1.6))
    problems = camera.settings_mismatch(locked, bad)
    assert len(problems) == 2 and any("ExposureTime" in p for p in problems)
    c = camera.controls_from_settings(locked)
    assert c["AeEnable"] is False and c["AwbEnable"] is False and c["AfMode"] == 0


def test_pi_coverage_and_save(tmp_path):
    cov = camera.Coverage((4608, 2592), (16, 9))
    cov.add(np.array([[10.0, 10.0], [4600, 2590], [2300, 1300]]))
    assert cov.fraction == pytest.approx(3 / 144)
    assert cov.ascii().count("\n") == 8
    p = camera.save_capture(
        tmp_path, "x", np.zeros((4, 4, 3), np.uint8), {"ExposureTime": 1, "ColourGains": (1.0, 2.0)}
    )
    assert p.is_file() and read_json(p.with_suffix(".json"))["metadata"]["ColourGains"] == [
        1.0,
        2.0,
    ]


class FakeSerial:
    def __init__(self, port, baud, timeout=1):
        self.sent = []
        self.replies = []

    def write(self, data):
        self.sent.append(data.decode())
        self.replies.append(b"ok\n")

    def readline(self):
        return self.replies.pop(0) if self.replies else b""

    def close(self):
        pass


def test_printer_only_homes_and_parks():
    st = PrinterSettings(port="COM9", park_y_mm=200, feed_mm_min=3000, timeout_s=2)
    assert allowed("G28 Y", st) and allowed("g0 y200 f3000", st) and allowed("M400", st)
    for bad in ("G28", "G28 X", "G0 Y150 F3000", "G1 X10", "M104 S200", "G0 Z5"):
        assert not allowed(bad, st), bad
    assert not allowed("G0 Y200 F3000", PrinterSettings(port="x"))  # park only when configured
    pr = Printer(st, opener=FakeSerial)
    pr.home_and_park()
    assert (
        pr.ser.sent
        == [c + "\n" for c in park_commands(st)]
        == ["G28 Y\n", "M400\n", "G0 Y200 F3000\n", "M400\n"]
    )
    with pytest.raises(PrinterError):
        pr.send("G1 X50")
    assert "G1 X50\n" not in pr.ser.sent  # refused before anything is written
    with pytest.raises(PrinterError):
        Printer(PrinterSettings(port=None), opener=FakeSerial)


def test_cli_parses_every_command():
    p = cli.build_parser()
    assert p.parse_args(["lock", "--out", "s.json"]).func.__module__ == "screwrig.lock"
    a = p.parse_args(["calib-capture", "--out", "x", "--mode", "single"])
    assert a.mode == "single"
    a = p.parse_args(["capture", "--session", "s01", "--settings", "s.json", "--no-printer"])
    assert a.no_printer and a.func.__module__ == "screwrig.session"
    assert p.parse_args(["printer", "status"]).action == "status"
    with pytest.raises(SystemExit):
        p.parse_args(["printer", "G1 X10"])


def test_printer_needs_a_port(monkeypatch):
    from screwsynth.common.config import Config

    monkeypatch.setattr(printer_mod, "moonraker_answers", lambda st, http=None: False)
    with pytest.raises(SystemExit):
        cli.cmd_printer(argparse.Namespace(action="status", config=None))
    assert cli.session.printer_from_config(Config({"rig": {"printer": {"port": None}}})) is None


def test_detect_numbering_model_and_page(tmp_path):
    import time
    import urllib.request

    from screwrig import detect

    assert detect.next_number(tmp_path) == 1
    for name in ("det_0001.jpg", "det_0001_vis.jpg", "det_0007_result.json", "other.jpg"):
        (tmp_path / name).write_text("x")
    assert detect.next_number(tmp_path) == 8  # a rerun continues after the highest number

    models = tmp_path / "models"
    assert detect.find_model(models) is None
    (models / "a").mkdir(parents=True)
    (models / "a" / "best.pt").write_text("x")
    assert detect.find_model(models) == models / "a" / "best.pt"
    (models / "a" / "exports" / "best_ncnn_model").mkdir(parents=True)
    assert detect.find_model(models).name == "best_ncnn_model"  # NCNN preferred on the Pi

    result = {
        "objects": [
            {"class": "DIN912_M4", "score": 0.91, "length": {"length_mm": 16.0}},
            {"class": "DIN934_M4", "score": 0.8},
        ]
    }
    assert detect.summary_lines(result) == ["DIN912_M4 16 mm (0.91)", "DIN934_M4 (0.80)"]

    out = tmp_path / "out"
    out.mkdir()
    detect._status(out, "analysing", "det_0001_vis.jpg", ["DIN912_M4 16 mm (0.91)"])
    detect._status(out, "ready")  # keeps the last image
    server = detect.serve(out, 18765)
    try:
        port = server.server_address[1]
        for _ in range(20):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/latest.json").read()
                break
            except OSError:
                time.sleep(0.1)
        latest = json.loads(body)
        assert latest["status"] == "ready" and latest["image"] == "det_0001_vis.jpg"
    finally:
        server.shutdown()


def test_detect_refuses_the_cuda_torch_on_a_pi():
    from screwrig.detect import torch_problem

    assert "Illegal instruction" in torch_problem("2.15.0+cu132", "aarch64")
    assert "download.pytorch.org/whl/cpu" in torch_problem("2.15.0+cu132", "aarch64")
    assert torch_problem("2.15.0+cpu", "aarch64") is None
    assert torch_problem("2.5.1", "aarch64") is None
    assert torch_problem("2.6.0+cu124", "AMD64") is None  # the PC's GPU build is fine


class FakeMoonraker:
    """Moonraker's HTTP API as Klipper answers it."""

    def __init__(self, state="ready", idle="Idle", up=True):
        self.calls = []
        self.state, self.idle, self.up = state, idle, up

    def __call__(self, method, path, body):
        if not self.up:
            raise PrinterError("Moonraker does not answer")
        self.calls.append((method, path, body))
        if path.startswith("/printer/objects/query"):
            return {
                "result": {
                    "status": {
                        "webhooks": {"state": self.state},
                        "idle_timeout": {"state": self.idle},
                        "toolhead": {"position": [10.0, 200.0, 30.0, 0.0], "homed_axes": "y"},
                    }
                }
            }
        return {"result": "ok"}


def test_klipper_printer_goes_through_moonraker():
    st = PrinterSettings(park_y_mm=200, feed_mm_min=3000)
    http = FakeMoonraker()
    pr = open_printer(st, http)  # auto: Moonraker answers, so never the USB port
    assert isinstance(pr, MoonrakerPrinter)
    pr.home_and_park()
    scripts = [b["script"] for m, path, b in http.calls if path == "/printer/gcode/script"]
    assert scripts == ["G28 Y", "M400", "G0 Y200 F3000", "M400"]
    with pytest.raises(PrinterError, match="refusing"):
        pr.send("G1 X50")
    assert all(b is None or b.get("script") != "G1 X50" for _, _, b in http.calls)
    assert pr.send("M114") == ["X:10.00 Y:200.00 Z:30.00 homed:y"]


def test_klipper_busy_or_not_ready_is_not_moved():
    st = PrinterSettings(park_y_mm=200)
    for http, why in (
        (FakeMoonraker(idle="Printing"), "busy"),
        (FakeMoonraker(state="shutdown"), "not ready"),
    ):
        with pytest.raises(PrinterError, match=why):
            MoonrakerPrinter(st, http).home_and_park()
        assert not [c for c in http.calls if c[1] == "/printer/gcode/script"]


def test_auto_falls_back_to_serial_only_without_moonraker():
    down = FakeMoonraker(up=False)
    assert open_printer(PrinterSettings(), down) is None  # nothing to move
    pr = open_printer(PrinterSettings(port="COM9"), down, opener=FakeSerial)
    assert isinstance(pr, Printer)
    assert isinstance(open_printer(PrinterSettings(driver="moonraker"), down), MoonrakerPrinter)
    with pytest.raises(PrinterError, match="driver"):
        PrinterSettings.from_config({"driver": "octoprint"})
    st = PrinterSettings.from_config({"url": "http://pi:7125"})
    assert st.driver == "auto" and st.url == "http://pi:7125"
