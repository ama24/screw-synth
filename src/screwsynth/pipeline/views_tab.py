"""GUI: camera views per layout (Camera tab) and HDRI environments (Environment tab).

Each view setting is one value (fixed) or a range; the renderer covers ranges evenly. The HDRI files
(paths.hdri_dir) are shared by all recipes; nothing is downloaded, the tool works offline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from nicegui import ui

from screwsynth.common.paths import resolve
from screwsynth.scenegen import environments as envs
from screwsynth.scenegen.variation import CameraSettings, EnvironmentSettings, parse_value

from .recipe import get_path

log = logging.getLogger(__name__)

VIEW_FIELDS = (
    (
        "height_mm",
        "Camera height (mm)",
        [250, 400],
        "calibrated",
        "calibrated = the rig camera's own height (about 300 mm)",
    ),
    ("tilt_deg", "Tilt from vertical (deg)", [0, 30], None, "0 = looking straight down"),
    (
        "tilt_direction_deg",
        "Tilt direction (deg)",
        [0, 360],
        None,
        "the side the camera looks from: 0 = from +x, 90 = from +y; [0, 360] = all sides",
    ),
    ("roll_deg", "Image rotation (deg)", [0, 360], None, "rotation around the view axis"),
)


def fixed_or_range(
    ed: Any, path: str, label: str, default_range: list[float], keyword: str | None, note: str
) -> None:
    """One setting as 'fixed value', 'range' or a keyword (e.g. calibrated)."""
    modes = {"fixed": "fixed", "range": "range"}
    if keyword:
        modes = {keyword: keyword, **modes}

    def current() -> Any:
        return ed.get(path)

    def mode_of(v: Any) -> str:
        if keyword and v == keyword:
            return keyword
        return "range" if isinstance(v, list | tuple) else "fixed"

    with ui.row().classes("w-full items-center no-wrap gap-2"):
        ui.label(label).classes("w-56 shrink-0").tooltip(note)
        toggle = ui.toggle(modes, value=mode_of(current())).props("dense no-caps")
        toggle.mark(f"mode-{path}")
        box = ui.row().classes("items-center no-wrap gap-1")
        ui.label(note).classes("text-xs text-grey-7")

    def store(value: Any) -> None:
        try:
            parse_value(value, path, (keyword,) if keyword else ())
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        ed.set(path, value)

    def draw() -> None:
        box.clear()
        v = current()
        with box:
            if toggle.value == "range":
                pair = list(v) if isinstance(v, list | tuple) else list(default_range)

                def put(i: int, x: Any) -> None:
                    if x is None:
                        return
                    pair[i] = float(x)
                    store(list(pair))

                ui.number(value=pair[0], on_change=lambda e: put(0, e.value)).props(
                    "dense outlined"
                ).classes("w-28")
                ui.label("to")
                ui.number(value=pair[1], on_change=lambda e: put(1, e.value)).props(
                    "dense outlined"
                ).classes("w-28")
            elif toggle.value == "fixed":
                one = float(v) if isinstance(v, int | float) else float(default_range[0])
                ui.number(
                    value=one, on_change=lambda e: e.value is not None and store(float(e.value))
                ).props("dense outlined").classes("w-28")

    def switch_mode(e: Any) -> None:
        v = current()
        if e.value == keyword:
            store(keyword)
        elif e.value == "range" and not isinstance(v, list | tuple):
            store(list(default_range))
        elif e.value == "fixed" and not isinstance(v, int | float):
            store(float(v[0]) if isinstance(v, list | tuple) else float(default_range[0]))
        draw()

    toggle.on_value_change(switch_mode)
    draw()


def camera_views_card(page: Any) -> None:
    ed = page.ed
    with ui.card().classes("w-full"):
        ui.label("Camera views per layout").classes("font-bold")
        ui.label(
            "Each setting is one value (fixed) or a range. Ranges are covered evenly, also when "
            "you stop early or add layouts later. The defaults (calibrated height, tilt 0, "
            "rotation 0) are the rig's own view and give exactly the usual dataset. Images from "
            "any other view are marked not measurable: length measurement needs the rig view."
        ).classes("text-xs text-grey-7")
        for key, label, rng, keyword, note in VIEW_FIELDS:
            fixed_or_range(ed, f"scenegen.camera.{key}", label, rng, keyword, note)
        status = ui.label().classes("text-sm")

        def show() -> None:
            try:
                s = CameraSettings.from_config(get_path(ed.merged(), "scenegen.camera", {}))
            except ValueError as exc:
                status.set_text(f"Problem: {exc}")
                status.classes(replace="text-sm text-negative")
                return
            status.classes(replace="text-sm")
            status.set_text(
                "Rig view only (the usual dataset)."
                if s.is_rig_view
                else "Varied views: every layout gets its own camera; the plate and shroud grow "
                "to cover them all."
            )

        _watch(ed, status, show)
        ui.button(
            "Back to the rig view", icon="restart_alt", on_click=lambda: _reset_views(page)
        ).props("flat")


def _watch(ed: Any, element: Any, fn: Any) -> None:
    """Run fn now and after every edit, until `element` is gone (the page was rebuilt)."""

    def run() -> None:
        if getattr(element, "is_deleted", False):
            ed.on_change.remove(run)
            return
        fn()

    ed.on_change.append(run)
    fn()


def _reset_views(page: Any) -> None:
    for key, *_ in VIEW_FIELDS:
        page.ed.reset(f"scenegen.camera.{key}")
    page.body.refresh()


# --------------------------------------------------------------------------------- environment

ENV_GUIDE = """
**What an HDRI is.** A 360-degree HDR photo of a real place (workshop, office, outdoors). Wrapped
around the scene, it lights the parts and appears in their reflections, which makes shiny parts
(stainless, zinc, brass) look like they do in photos taken in a room.

**Offline.** The tool never downloads anything: download the files yourself, once, then add them
below (they are copied to the HDRI folder). Pick `.hdr` or `.exr`, 2K or 4K is enough (about
5-50 MB each). Useful kinds: indoor (workshop, office, kitchen, garage), studio, overcast outdoor.
20-40 files give plenty of variety.

**When it is on**, the shroud is left out (an open room), every layout gets one HDRI (each file
used about equally often), and strength and rotation are covered evenly over their ranges.
"""


def environment_tab(page: Any) -> None:
    ed = page.ed
    hdri_dir = resolve(get_path(ed.merged(), "paths.hdri_dir", "data/hdri"))
    with ui.column().classes("w-full max-w-5xl gap-3"):
        with ui.card().classes("w-full"):
            ui.label("HDRI environment (off by default)").classes("font-bold")
            with ui.expansion("What it does, and where to get HDRIs", icon="help").classes(
                "w-full"
            ):
                ui.markdown(ENV_GUIDE)
                for name, url, licence in envs.SOURCES:
                    with ui.row().classes("items-center gap-2"):
                        ui.link(name, url, new_tab=True).classes("font-medium")
                        ui.label(f"{url} - {licence}").classes("text-xs text-grey-7")
            page.w.field("scenegen.environment.enabled", "Use HDRI environments")
            fixed_or_range(
                ed,
                "scenegen.environment.strength",
                "Strength",
                [0.3, 1.5],
                None,
                "brightness of the environment",
            )
            fixed_or_range(
                ed,
                "scenegen.environment.rotation_deg",
                "Rotation (deg)",
                [0, 360],
                None,
                "turns the environment around the scene",
            )
            page.w.field("scenegen.environment.panel_light", "Keep the LED panel on")
            status = ui.label().classes("text-sm")

        with ui.card().classes("w-full"):
            ui.label(f"HDRI files (shared by all recipes: {hdri_dir})").classes("font-bold")
            table = ui.column().classes("w-full gap-2")

            def draw_table() -> None:
                table.clear()
                names = envs.files(hdri_dir)
                off = envs.disabled(hdri_dir)
                with table:
                    if not names:
                        ui.label("No HDRI files yet: add some below.").classes(
                            "text-sm text-grey-7"
                        )
                    for name in names:
                        path = hdri_dir / name
                        with ui.row().classes("w-full items-center no-wrap gap-3"):
                            preview = hdri_thumbnail(path)
                            if preview is not None:
                                ui.image(str(preview)).classes("w-48")
                            else:
                                ui.label("no preview").classes("w-48 text-xs text-grey-7")
                            with ui.column().classes("gap-0 grow"):
                                ui.label(name).classes("font-medium")
                                ui.label(f"{path.stat().st_size / 1e6:.1f} MB").classes(
                                    "text-xs text-grey-7"
                                )
                            ui.switch(
                                "use",
                                value=name not in off,
                                on_change=lambda e, n=name: (
                                    envs.set_enabled(hdri_dir, n, bool(e.value)),
                                    show(),
                                ),
                            )

                            def delete(n: str = name) -> None:
                                (hdri_dir / n).unlink(missing_ok=True)
                                envs.set_enabled(hdri_dir, n, True)  # forget its on/off entry
                                (hdri_dir / ".thumbs" / f"{n}.jpg").unlink(missing_ok=True)
                                draw_table()
                                show()

                            ui.button(icon="delete", on_click=delete).props(
                                "flat dense round"
                            ).tooltip("delete the file")

            async def uploaded(e: Any) -> None:
                name = Path(e.file.name).name
                if Path(name).suffix.lower() not in envs.SUFFIXES:
                    ui.notify(f"{name}: not an HDRI ({', '.join(envs.SUFFIXES)})", type="warning")
                    return
                hdri_dir.mkdir(parents=True, exist_ok=True)
                await e.file.save(hdri_dir / name)
                ui.notify(f"{name} added", type="positive")
                draw_table()
                show()

            draw_table()
            ui.upload(
                label="Add HDRI files (.hdr / .exr) from a folder",
                multiple=True,
                auto_upload=True,
                on_upload=uploaded,
            ).props("accept=.hdr,.exr").classes("w-full")

        def show() -> None:
            try:
                s = EnvironmentSettings.from_config(
                    get_path(ed.merged(), "scenegen.environment", {})
                )
            except ValueError as exc:
                status.set_text(f"Problem: {exc}")
                return
            n = len(envs.usable(hdri_dir))
            if not s.enabled:
                status.set_text("Off: the scene is lit as usual (shroud, LED panel).")
            elif n == 0:
                status.set_text("On, but no HDRI file is switched on: renders stay as usual.")
            else:
                status.set_text(f"On: {n} HDRI file(s), each used about equally often.")

        _watch(ed, status, show)


def hdri_thumbnail(path: Path, width: int = 320) -> Path | None:
    """A small tone-mapped JPEG preview of an HDRI (cached next to it), or None."""
    out = path.parent / ".thumbs" / f"{path.name}.jpg"
    if out.is_file() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # OpenCV reads .exr only with this
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32)
        x *= 0.5 / max(float(x.mean()), 1e-6)  # expose for mid-grey, then Reinhard + gamma
        x = (x / (1.0 + x)) ** (1 / 2.2)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.clip(x * 255, 0, 255).astype(np.uint8))
        return out
    except Exception as exc:  # noqa: BLE001 - a preview must never break the tab
        log.debug("no preview for %s: %s", path, exc)
        return None


DIRT_GUIDE = """Dirt layers a dark oil film and dust specks over the parts and the plate, so the model also
works when the parts come out of an oily box or the plate has not been cleaned. Parts and plate are
switched on and off separately.

- **Share**: how many parts (or layouts, for the plate) get dirt at all.
- **Oil**: how much of a dirty surface the dark oil covers (0 = none, 1 = all of it).
- **Dust**: how many dust specks (0 = none, 1 = as many as fit).
- **Dust speck size** in mm.

A part keeps its dirt in every photo of its layout, and the plate stays the same for all photos of
a layout (picker removals and lighting variants). Dirt never changes the labels."""


def dirt_tab(page: Any) -> None:
    """Dirt on the parts and on the plate, each with its own switch (scenegen.dirt)."""
    ed = page.ed
    with ui.column().classes("w-full max-w-5xl gap-3"):
        with ui.expansion("What dirt does", icon="help").classes("w-full"):
            ui.markdown(DIRT_GUIDE)
        for target, title in (("parts", "Dirt on the parts"), ("plate", "Dirt on the plate (bed)")):
            base = f"scenegen.dirt.{target}"
            with ui.card().classes("w-full").mark(f"dirt-{target}"):
                ui.label(title).classes("font-bold")
                page.w.field(f"{base}.enabled", f"{title}: on")
                page.w.field(
                    f"{base}.share",
                    "Share of dirty parts" if target == "parts" else "Share of dirty layouts",
                )
                fixed_or_range(
                    ed, f"{base}.oil", "Dark oil", [0.05, 0.5], None, "covered share, 0..1"
                )
                fixed_or_range(ed, f"{base}.dust", "Dust", [0.05, 0.4], None, "amount, 0..1")
                fixed_or_range(
                    ed, f"{base}.dust_size_mm", "Dust speck size (mm)", [0.1, 0.4], None, "mm"
                )
