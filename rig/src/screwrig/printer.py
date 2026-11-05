"""Homing / parking the Ender 3 bed before a capture.

Two ways to reach the printer (rig.printer.driver):
- moonraker: the printer runs Klipper (as with the screwpick sorter). Klipper owns the printer's
  USB port, and the board no longer speaks G-code over it, so the commands go through Moonraker's
  HTTP API (http://localhost:7125 on the Pi). A printer that is printing is never moved.
- serial: a printer with its stock firmware (Marlin), G-code over the USB serial port.
- auto (default): Moonraker if it answers, else the serial port if one is set.

Only homing Y and parking Y at the configured position are ever sent (plus waiting for moves to
finish). Any other G-code is refused: other motion commands need the operator's explicit approval
and are not part of this tool.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("screwrig.printer")


class PrinterError(RuntimeError):
    pass


DRIVERS = ("auto", "moonraker", "serial")


@dataclass
class PrinterSettings:
    driver: str = "auto"  # auto | moonraker (Klipper) | serial (stock firmware over USB)
    url: str = "http://localhost:7125"  # Moonraker
    port: str | None = None  # serial: e.g. /dev/ttyUSB0 on the Pi, COM5 on Windows
    baud: int = 115200
    park_y_mm: float | None = None  # None: home only
    feed_mm_min: float = 3000.0
    timeout_s: float = 60.0

    @classmethod
    def from_config(cls, cfg: dict) -> PrinterSettings:
        p = dict(cfg or {})
        d = cls()
        driver = str(p.get("driver", d.driver)).lower()
        if driver not in DRIVERS:
            raise PrinterError(f"rig.printer.driver must be one of {', '.join(DRIVERS)}")
        return cls(
            driver=driver,
            url=str(p.get("url", d.url)),
            port=p.get("port"),
            baud=int(p.get("baud", d.baud)),
            park_y_mm=None if p.get("park_y_mm") is None else float(p["park_y_mm"]),
            feed_mm_min=float(p.get("feed_mm_min", d.feed_mm_min)),
            timeout_s=float(p.get("timeout_s", d.timeout_s)),
        )


def allowed(line: str, st: PrinterSettings) -> bool:
    """The whitelist: home Y, park Y at the configured position, wait for moves, report position."""
    g = line.strip().upper()
    if g in ("G28 Y", "M400", "M114"):
        return True
    m = re.fullmatch(r"G0 Y(-?\d+(?:\.\d+)?) F(\d+(?:\.\d+)?)", g)
    return bool(m and st.park_y_mm is not None and abs(float(m[1]) - st.park_y_mm) < 1e-6)


def park_commands(st: PrinterSettings) -> list[str]:
    cmds = ["G28 Y", "M400"]
    if st.park_y_mm is not None:
        cmds += [f"G0 Y{st.park_y_mm:g} F{st.feed_mm_min:g}", "M400"]
    return cmds


class Printer:
    """Line-based G-code over serial, restricted to `allowed` commands."""

    def __init__(self, st: PrinterSettings, opener: Callable | None = None) -> None:
        self.st = st
        if not st.port:
            raise PrinterError("no printer port configured (rig.printer.port)")
        if opener is None:
            try:
                import serial  # pyserial
            except ImportError as exc:  # pragma: no cover - depends on the Pi
                raise PrinterError("pip install pyserial to drive the printer") from exc
            opener = serial.Serial
        self.ser = opener(st.port, st.baud, timeout=1)
        time.sleep(0.1)

    def send(self, line: str) -> list[str]:
        if not allowed(line, self.st):
            raise PrinterError(f"refusing to send {line!r}: only homing/parking Y is allowed")
        self.ser.write((line.strip() + "\n").encode("ascii"))
        replies, t0 = [], time.monotonic()
        while time.monotonic() - t0 < self.st.timeout_s:
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("ascii", errors="replace").strip()
            replies.append(text)
            if text.lower().startswith("ok"):
                return replies
            if text.lower().startswith(("error", "!!")):
                raise PrinterError(f"{line}: {text}")
        raise PrinterError(f"{line}: no 'ok' within {self.st.timeout_s:.0f} s")

    def home_and_park(self) -> None:
        for cmd in park_commands(self.st):
            log.info("printer: %s", cmd)
            self.send(cmd)

    def close(self) -> None:
        self.ser.close()


Http = Callable[[str, str, dict | None], dict]  # (method, path, json body) -> json reply


def _urllib_http(url: str, timeout: float) -> Http:
    def call(method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - local Moonraker
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise PrinterError(f"Moonraker {path}: {exc.code} {detail[:200]}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise PrinterError(f"Moonraker at {url} does not answer: {exc}") from exc

    return call


def moonraker_answers(st: PrinterSettings, http: Http | None = None) -> bool:
    try:
        (http or _urllib_http(st.url, 2.0))("GET", "/server/info", None)
        return True
    except PrinterError:
        return False


class MoonrakerPrinter:
    """The same few commands through Klipper (Moonraker's HTTP API), restricted to `allowed`."""

    def __init__(self, st: PrinterSettings, http: Http | None = None) -> None:
        self.st = st
        self.http = http or _urllib_http(st.url, st.timeout_s)

    def _status(self) -> dict[str, Any]:
        reply = self.http(
            "GET", "/printer/objects/query?webhooks&idle_timeout&toolhead=position,homed_axes", None
        )
        return reply.get("result", {}).get("status", {})

    def _check_free(self) -> None:
        s = self._status()
        state = s.get("webhooks", {}).get("state")
        if state != "ready":
            raise PrinterError(f"Klipper is not ready ({state}): see the printer's web page")
        if s.get("idle_timeout", {}).get("state") == "Printing":
            raise PrinterError("the printer is busy (printing or the sorter is moving): not moved")

    def send(self, line: str) -> list[str]:
        if not allowed(line, self.st):
            raise PrinterError(f"refusing to send {line!r}: only homing/parking Y is allowed")
        cmd = line.strip().upper()
        if cmd == "M114":  # Klipper prints M114 to its console: read the position instead
            th = self._status().get("toolhead", {})
            x, y, z = (list(th.get("position") or [0, 0, 0]) + [0, 0, 0])[:3]
            return [f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f} homed:{th.get('homed_axes', '')}"]
        self.http("POST", "/printer/gcode/script", {"script": cmd})
        return ["ok"]

    def home_and_park(self) -> None:
        self._check_free()
        for cmd in park_commands(self.st):
            log.info("printer (Klipper): %s", cmd)
            self.send(cmd)

    def close(self) -> None:
        pass


def open_printer(st: PrinterSettings, http: Http | None = None, opener: Callable | None = None):  # noqa: ANN201
    """The printer as configured (driver), or None when there is none to move."""
    driver = st.driver
    if driver == "auto":
        if moonraker_answers(st, http):
            driver = "moonraker"
        elif st.port:
            driver = "serial"
        else:
            return None
    if driver == "moonraker":
        return MoonrakerPrinter(st, http)
    return Printer(st, opener) if st.port else None
