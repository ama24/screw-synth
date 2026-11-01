"""Dataset builder GUI: a local web page (NiceGUI) to set up a recipe and run the whole pipeline.

    screwsynth-gui            (or: screwsynth-dataset gui [--port 8080] [--no-browser])

Everything happens on this PC; the page is served on http://localhost:8080 only. The GUI edits
recipes/<name>.yaml and runs exactly what `screwsynth-dataset run <name>` runs.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nicegui import app, ui

from screwsynth.common.config import config_path
from screwsynth.common.paths import resolve

from . import parts as parts_mod
from .background_tab import background_tab
from .forms import Field, build_fields, coerce, config_comments, to_yaml_text
from .recipe import (
    LABEL_FORMATS,
    STEPS,
    Recipe,
    RecipeError,
    get_path,
    list_recipes,
    load_base_config,
    load_recipe,
    recipes_dir,
    remove_path,
    set_path,
)
from .runner import Runner, tail
from .steps import (
    TITLES,
    Check,
    auto_render_workers,
    build_plan,
    dataset_outputs,
    latest_render,
    preflight,
    read_status,
)
from .views_tab import camera_views_card, dirt_tab, environment_tab

log = logging.getLogger("screwsynth.pipeline.gui")

# config sections shown under "All settings" (paths, calibration solving, training: elsewhere)
SETTINGS_SECTIONS = {
    "general": ("General", ["seed", "world_scale"]),
    "catalogue": ("Catalogue", None),
    "partgen": ("Part meshes", None),
    "restposes": ("Rest poses", None),
    "scenegen": ("Rendering", None),
    "postprocess": ("Post-process", None),
    "labels": ("Labels", None),
}
SKIP = frozenset({"catalogue.extra_rules"})  # edited in the Parts tab
STATE_ICONS = {
    "pending": ("radio_button_unchecked", "grey"),
    "running": ("autorenew", "primary"),
    "done": ("check_circle", "positive"),
    "failed": ("error", "negative"),
    "cancelled": ("cancel", "warning"),
    "skipped": ("remove_circle_outline", "grey"),
}


# ------------------------------------------------------------------------------ shared state


@dataclass
class RunState:
    """One pipeline run at a time, shared by every open browser tab."""

    runner: Runner | None = None
    thread: threading.Thread | None = None
    lines: list[str] = field(default_factory=list)
    offset: int = 0  # number of lines dropped from the front of `lines`
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > 5000:
                drop = len(self.lines) - 4000
                del self.lines[:drop]
                self.offset += drop

    def since(self, pos: int) -> tuple[list[str], int]:
        with self.lock:
            start = max(0, pos - self.offset)
            return self.lines[start:], self.offset + len(self.lines)

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


RUN = RunState()


# ------------------------------------------------------------------------------ recipe editor


class Editor:
    """The recipe being edited: values = config.yaml + recipe overrides."""

    def __init__(self, recipe: Recipe) -> None:
        self.base = load_base_config()
        self.comments = config_comments(config_path())
        self.recipe = recipe
        self.saved = json.dumps(recipe.to_dict(), sort_keys=True)
        self.on_change: list[Callable[[], None]] = []

    @property
    def dirty(self) -> bool:
        return json.dumps(self.recipe.to_dict(), sort_keys=True) != self.saved

    def merged(self) -> dict[str, Any]:
        return self.recipe.merged_config(self.base)

    def get(self, path: str) -> Any:
        over = get_path(self.recipe.config, path, _MISSING)
        return copy.deepcopy(over if over is not _MISSING else get_path(self.base, path))

    def overridden(self, path: str) -> bool:
        return get_path(self.recipe.config, path, _MISSING) is not _MISSING

    def set(self, path: str, value: Any) -> None:
        if value == get_path(self.base, path, _MISSING):
            remove_path(self.recipe.config, path)
        else:
            set_path(self.recipe.config, path, copy.deepcopy(value))
        self.changed()

    def reset(self, path: str) -> None:
        remove_path(self.recipe.config, path)
        self.changed()

    def changed(self) -> None:
        for cb in list(self.on_change):
            cb()

    def save(self) -> Path:
        path = self.recipe.save()
        self.saved = json.dumps(self.recipe.to_dict(), sort_keys=True)
        self.changed()
        return path

    def field(self, path: str) -> Field:
        value = get_path(self.base, path)
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        fs = build_fields(
            {path.rsplit(".", 1)[-1]: value},
            self.comments,
            f"{parent}." if parent else "",
            self.comments.get(parent, ""),
        )
        return fs[0]


_MISSING = object()


def default_recipe() -> Recipe:
    """The recipe opened at start: the last one used, else the first, else a new preview."""
    recipes = list_recipes()
    last = _query("recipe") or (app.storage.general.get("last_recipe") if _storage_ok() else None)
    for path in recipes:
        if path.stem == last:
            return load_recipe(path)
    if recipes:
        return load_recipe(recipes[0])
    r = Recipe("preview", "Quick look: 10 low-quality scenes with the nominal camera")
    r.pipeline.preview, r.pipeline.scenes = True, 10
    r.save()
    return r


def _storage_ok() -> bool:
    try:
        _ = app.storage.general
        return True
    except Exception:  # storage not available outside a running app (tests)
        return False


# ------------------------------------------------------------------------------ widgets


class Widgets:
    """Builds bound input widgets for config fields."""

    def __init__(self, ed: Editor) -> None:
        self.ed = ed

    def field(self, path: str, label: str | None = None, f: Field | None = None) -> None:
        f = f or self.ed.field(path)
        if f.kind == "section":
            with ui.expansion(label or f.key, icon="tune").classes("w-full").props("dense"):
                if f.help:
                    ui.label(f.help).classes("text-xs text-grey-7 whitespace-pre-line")
                for child in f.children:
                    self.field(child.path, f=child)
            return
        ed = self.ed
        value = ed.get(f.path)
        if not _fits(f.kind, value):
            # a recipe may hold another shape than config.yaml's default (a range where the
            # default is one number, e.g. scenegen.camera.tilt_deg): edit and save it as YAML
            f = dataclasses.replace(f, kind="yaml")
        with ui.row().classes("w-full items-center no-wrap gap-2"):
            name = ui.label(label or f.key).classes("w-56 shrink-0")
            if f.help:
                name.tooltip(f.help)
            box = ui.row().classes("grow items-center no-wrap gap-1")
            reset = ui.button(icon="restart_alt").props("flat dense round size=sm")
            reset.tooltip("back to the value in config.yaml: " + to_yaml_text(f.value))

        def mark() -> None:
            on = ed.overridden(f.path)
            reset.set_visibility(on)
            name.classes(
                add="text-primary font-bold" if on else "",
                remove="" if on else "text-primary font-bold",
            )

        def store(raw: Any) -> None:
            try:
                ed.set(f.path, coerce(f, raw))
            except (ValueError, TypeError) as exc:
                ui.notify(f"{f.path}: {exc}", type="warning")
                return
            mark()

        with box:
            self._input(f, value, store)

        def do_reset() -> None:
            ed.reset(f.path)
            box.clear()
            with box:
                self._input(f, ed.get(f.path), store)
            mark()

        reset.on_click(do_reset)
        mark()

    def _input(self, f: Field, value: Any, store: Callable[[Any], None]) -> None:
        kind = f.kind
        if kind == "bool":
            ui.switch(value=bool(value), on_change=lambda e: store(e.value)).mark(
                f"switch-{f.path}"
            )
        elif kind in ("int", "float"):
            step = 1 if kind == "int" else _step(value)
            ui.number(
                value=value, step=step, on_change=lambda e: e.value is not None and store(e.value)
            ).props("dense outlined").classes("w-40")
        elif kind == "pair":
            pair = list(value)
            step = (
                1
                if all(isinstance(v, int) for v in f.value)
                else _step(min(abs(v) for v in f.value) or 0.1)
            )

            def put(i: int, v: Any) -> None:
                if v is None:
                    return
                pair[i] = v
                store(list(pair))

            ui.number(value=pair[0], step=step, on_change=lambda e: put(0, e.value)).props(
                "dense outlined"
            ).classes("w-32")
            ui.label("to")
            ui.number(value=pair[1], step=step, on_change=lambda e: put(1, e.value)).props(
                "dense outlined"
            ).classes("w-32")
        elif kind == "choice":
            ui.select(f.choices, value=value, on_change=lambda e: store(e.value)).props(
                "dense outlined"
            ).classes("w-48")
        elif kind == "str":
            ui.input(value=value or "", on_change=lambda e: store(e.value)).props(
                "dense outlined"
            ).classes("w-72")
        else:  # yaml
            area = (
                ui.textarea(value=to_yaml_text(value))
                .props("dense outlined autogrow")
                .classes("w-full font-mono text-xs")
            )
            area.on("blur", lambda: store(area.value))


def _fits(kind: str, value: Any) -> bool:
    """Whether a value can be edited with the widget of a field kind."""
    if kind in ("int", "float"):
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind == "pair":
        return isinstance(value, list | tuple) and len(value) == 2
    if kind in ("str", "choice"):
        return isinstance(value, str)
    return True


def _step(value: Any) -> float:
    """Spinner step: a tenth of the value's order of magnitude (0.4 -> 0.01, 64 -> 1)."""
    v = abs(float(value or 0))
    if v == 0:
        return 0.01
    return max(10 ** (math.floor(math.log10(v)) - 1), 0.001) if v < 10 else 1


_THUMBS: dict[tuple[str, float], Any] = {}


def thumb(path: Path | None, width: int = 900) -> Any:
    """Downscaled PIL image of `path` (cached by modification time), or None."""
    if path is None or not path.is_file():
        return None
    key = (str(path), path.stat().st_mtime)
    if key not in _THUMBS:
        from PIL import Image

        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if im.width > width:
                    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
                _THUMBS[key] = im.copy()
        except OSError:
            return None
        if len(_THUMBS) > 40:
            _THUMBS.pop(next(iter(_THUMBS)))
    return _THUMBS[key]


def _newest(pattern: Path) -> Path | None:
    hits = (
        sorted(pattern.parent.glob(pattern.name), key=lambda p: p.stat().st_mtime)
        if pattern.parent.is_dir()
        else []
    )
    return hits[-1] if hits else None


# ------------------------------------------------------------------------------ the page


class BuilderPage:
    def __init__(self) -> None:
        self.ed = Editor(default_recipe())
        self.w = Widgets(self.ed)
        self.log_pos = 0
        self.status_widgets: dict[str, tuple[Any, Any, Any]] = {}
        self.preview_images: dict[str, tuple[Any, Any]] = {}

    # ---------------------------------------------------------------- layout

    def build(self) -> None:
        ui.page_title("screw-synth dataset builder")
        with ui.header().classes("items-center gap-3 bg-blue-grey-9"):
            ui.icon("hardware").classes("text-2xl")
            ui.label("screw-synth dataset builder").classes("text-lg font-bold")
            ui.space()
            self.recipe_select = (
                ui.select(
                    [p.stem for p in list_recipes()],
                    value=self.ed.recipe.name,
                    label="Recipe",
                    on_change=lambda e: self.switch(e.value),
                )
                .props("dense dark outlined options-dense")
                .classes("w-56")
            )
            ui.button("New", icon="add", on_click=self.new_dialog).props("flat color=white")
            ui.button("Save", icon="save", on_click=self.save).props("flat color=white").mark(
                "save"
            )
            self.dirty_badge = ui.badge("unsaved", color="orange")
        self.ed.on_change.append(lambda: self.dirty_badge.set_visibility(self.ed.dirty))
        self.dirty_badge.set_visibility(False)
        self.body()
        ui.timer(0.5, self.tick)
        lines, self.log_pos = RUN.since(0)
        if not lines:
            for line in tail(self.ed.recipe.run_dir(self.ed.base) / "pipeline.log", 300):
                self.log.push(line)
        else:
            for line in lines:
                self.log.push(line)

    @ui.refreshable_method
    def body(self) -> None:
        with ui.tabs().classes("w-full") as tabs:
            t_data = ui.tab("Dataset", icon="dataset")
            t_cam = ui.tab("Camera", icon="photo_camera")
            t_bg = ui.tab("Background", icon="texture")
            t_env = ui.tab("Environment", icon="panorama_photosphere")
            t_dirt = ui.tab("Dirt", icon="cleaning_services")
            t_parts = ui.tab("Parts", icon="category")
            t_set = ui.tab("All settings", icon="tune")
            t_run = ui.tab("Run", icon="play_arrow")
        wanted = _query("tab").lower()  # /?tab=run opens the Run tab (bookmarks)
        first = {
            "dataset": t_data,
            "camera": t_cam,
            "background": t_bg,
            "environment": t_env,
            "dirt": t_dirt,
            "parts": t_parts,
            "settings": t_set,
            "run": t_run,
        }
        with ui.tab_panels(tabs, value=first.get(wanted, t_data)).classes("w-full"):
            with ui.tab_panel(t_data):
                self.dataset_tab()
            with ui.tab_panel(t_cam):
                self.camera_tab()
            with ui.tab_panel(t_bg):
                background_tab(self)
            with ui.tab_panel(t_env):
                environment_tab(self)
            with ui.tab_panel(t_dirt):
                dirt_tab(self)
            with ui.tab_panel(t_parts):
                self.parts_tab()
            with ui.tab_panel(t_set):
                self.settings_tab()
            with ui.tab_panel(t_run):
                self.run_tab()

    # ---------------------------------------------------------------- dataset tab

    def dataset_tab(self) -> None:
        r, p, ed = self.ed.recipe, self.ed.recipe.pipeline, self.ed
        with ui.column().classes("w-full max-w-4xl gap-3"):
            with ui.card().classes("w-full"):
                ui.label(f"Dataset: {r.name}").classes("text-lg font-bold")
                ui.label(
                    f"Renders go to {r.run_dir(ed.base)}, the exported dataset to {r.dataset_dir(ed.base)}. "
                    "Use New to start another dataset (a copy of this one)."
                ).classes("text-sm text-grey-7")
                ui.textarea(
                    "Description",
                    value=r.description,
                    on_change=lambda e: self._set_attr(r, "description", e.value),
                ).props("dense outlined autogrow").classes("w-full")
            with ui.card().classes("w-full"):
                ui.label("Size").classes("font-bold")
                with ui.row().classes("items-center gap-4"):
                    ui.number(
                        "Scenes (layouts)",
                        value=p.scenes,
                        min=1,
                        step=100,
                        on_change=lambda e: e.value and self._set_attr(p, "scenes", int(e.value)),
                    ).props("dense outlined").classes("w-48")
                    ui.number(
                        "Scenes per Blender restart",
                        value=p.chunk,
                        min=1,
                        step=50,
                        on_change=lambda e: e.value and self._set_attr(p, "chunk", int(e.value)),
                    ).props("dense outlined").classes("w-56")
                    auto_n, _, why = auto_render_workers()
                    ui.number(
                        f"Renders in parallel (0 = auto: {auto_n})",
                        value=p.render_workers,
                        min=0,
                        max=16,
                        step=1,
                        on_change=lambda e: self._set_attr(p, "render_workers", int(e.value or 0)),
                    ).props("dense outlined").classes("w-64").mark("render-workers").tooltip(
                        "One Blender render keeps the GPU only partly busy (physics and scene set-up "
                        "run on the CPU), so several run side by side. Auto: as many as each NVIDIA "
                        f"card's memory holds, capped by RAM and CPU cores. Here: {why}."
                    )
                    ui.switch(
                        "Preview quality",
                        value=p.preview,
                        on_change=lambda e: self._set_attr(p, "preview", e.value),
                    ).tooltip(
                        "Low resolution and few samples (scenegen.preview): a quick check of the whole "
                        "pipeline, not for training."
                    )
                self.w.field("scenegen.variants_per_layout", "Lighting variants per layout")
                self.w.field("scenegen.objects", "Objects per normal scene")
                self.w.field("seed", "Random seed")
                ui.label(
                    "Rough guide: about 3,000 instances per class for stage 1. The Labels step reports "
                    "classes below labels.min_instances_per_class."
                ).classes("text-xs text-grey-7")
            with ui.card().classes("w-full"):
                ui.label("Export").classes("font-bold")
                ui.select(
                    list(LABEL_FORMATS),
                    value=list(p.formats),
                    multiple=True,
                    label="Formats",
                    on_change=lambda e: self._set_attr(
                        p, "formats", [f for f in LABEL_FORMATS if f in (e.value or [])]
                    ),
                ).props("dense outlined use-chips").classes("w-full")
                ui.label(
                    "seg = YOLO segmentation (stage 1), pose = YOLO keypoints, crops = full-resolution "
                    "measurement crops (stage 2), coco = COCO json."
                ).classes("text-xs text-grey-7")
                self.w.field("labels.stage1_imgsz", "Stage-1 image size (px)")
                self._split_field()

    def _split_field(self) -> None:
        split = self.ed.get("labels.split") or {"train": 0.9, "val": 0.1}
        with ui.row().classes("w-full items-center no-wrap gap-2"):
            ui.label("Validation share").classes("w-56").tooltip(
                "fraction of layouts in the val split (by layout, deterministic from the seed)"
            )

            def put(e: Any) -> None:
                if e.value is None:
                    return
                v = min(0.5, max(0.0, float(e.value)))
                self.ed.set("labels.split", {"train": round(1 - v, 4), "val": round(v, 4)})

            ui.number(value=split.get("val", 0.1), min=0, max=0.5, step=0.05, on_change=put).props(
                "dense outlined"
            ).classes("w-32")

    def _set_attr(self, obj: Any, attr: str, value: Any) -> None:
        setattr(obj, attr, value)
        self.ed.changed()

    # ---------------------------------------------------------------- camera tab

    def camera_tab(self) -> None:
        p, ed = self.ed.recipe.pipeline, self.ed
        calib_dir = resolve(get_path(ed.merged(), "paths.calibration_dir", "data/calibration"))
        calib = calib_dir / "calibration.json"
        with ui.column().classes("w-full max-w-4xl gap-3"):
            with ui.card().classes("w-full"):
                ui.label("Camera model used for rendering").classes("font-bold")
                ui.radio(
                    {
                        "nominal": "Nominal Pi Camera 3 (no calibration needed)",
                        "calibration": "My calibrated camera (tool 03)",
                    },
                    value=p.camera,
                    on_change=lambda e: self._set_attr(p, "camera", e.value),
                )
            with ui.card().classes("w-full"):
                ui.label("Calibration").classes("font-bold")
                if calib.is_file():
                    try:
                        data = json.loads(calib.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        data = {}
                    ui.label(f"Found {calib}").classes("text-positive")
                    rows = _calibration_summary(data)
                    for k, v in rows:
                        with ui.row().classes("gap-2"):
                            ui.label(k).classes("w-56 text-grey-7")
                            ui.label(v)
                    report = calib_dir / "report.html"
                    if report.is_file():
                        ui.label(f"Report: {report}").classes("text-xs")
                else:
                    ui.label(f"No calibration yet ({calib} is missing).").classes("text-warning")
                    ui.markdown(
                        "Calibration runs on the rig: print the board (`screwsynth-calib board`), capture "
                        "with the Pi scripts, then solve intrinsics, extrinsic and sensor profile. See "
                        "`tools/calibration/README.md`. Until then use the nominal camera."
                    ).classes("text-sm")
            with ui.card().classes("w-full"):
                ui.label("Nominal camera").classes("font-bold")
                ui.label("Used when 'Nominal' is selected above.").classes("text-xs text-grey-7")
                for f in ed.field("scenegen.nominal_camera").children:
                    self.w.field(f.path, f=f)
                self.w.field("scenegen.plate.thickness_mm", "TPU plate thickness (mm)")
            camera_views_card(self)

    # ---------------------------------------------------------------- parts tab

    def parts_tab(self) -> None:
        ed = self.ed
        try:
            ov = parts_mod.overview(ed.merged())
            selected = parts_mod.selected_classes(ed.merged())
        except Exception as exc:  # broken catalogue: show the error instead of a blank tab
            ui.label(f"Cannot read the catalogue: {exc}").classes("text-negative")
            return
        state: dict[str, Any] = {"selected": set(selected), "syncing": False}
        boxes: dict[str, Any] = {}
        fam_boxes: dict[str, Any] = {}

        def apply() -> None:
            rules = parts_mod.rules_for_selection(
                ov, state["selected"], ed.get("catalogue.extra_rules") or []
            )
            ed.set("catalogue.extra_rules", rules)
            s = parts_mod.summary(ed.merged())
            summary.set_text(
                f"{s['classes']} classes, {s['parts']} parts in {s['families']} families selected"
            )

        def sync() -> None:
            state["syncing"] = True  # setting values fires on_change; do not re-enter toggle
            try:
                for c, box in boxes.items():
                    box.value = c in state["selected"]
                for fam in ov.families:
                    on = [c in state["selected"] for c in fam.classes]
                    fam_boxes[fam.name].value = (
                        True if all(on) else (False if not any(on) else None)
                    )
            finally:
                state["syncing"] = False

        def toggle(classes: list[str], on: bool | None) -> None:
            if state["syncing"] or on is None:
                return
            if on:
                state["selected"] |= set(classes)
            else:
                state["selected"] -= set(classes)
            sync()
            apply()

        with ui.column().classes("w-full max-w-5xl gap-3"):
            with ui.card().classes("w-full"):
                summary = ui.label().classes("text-md font-bold").mark("parts-summary")
                self.w.field("catalogue.max_length_mm", "Longest screw length (mm)")
                self.w.field("catalogue.rod_max_length_mm", "Longest threaded rod piece (mm)")
                ui.label(
                    "Changing the longest length changes which parts exist; reopen this tab to refresh the list."
                ).classes("text-xs text-grey-7")
                sizes = sorted({c.rsplit("_", 1)[-1] for c in ov.classes}, key=_size_key)
                with ui.row().classes("items-center gap-1"):
                    ui.label("Sizes, all families:").classes("mr-2")
                    for size in sizes:
                        of_size = [c for c in ov.classes if c.rsplit("_", 1)[-1] == size]
                        with ui.button_group().props("flat"):
                            ui.button(size, on_click=lambda cs=of_size: toggle(cs, True)).props(
                                "dense no-caps color=primary"
                            ).tooltip(f"select {size} in every family")
                            ui.button(
                                icon="remove", on_click=lambda cs=of_size: toggle(cs, False)
                            ).props("dense color=grey-7").tooltip(
                                f"deselect {size} in every family"
                            )
            for fam in ov.families:
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        fb = ui.checkbox(
                            on_change=lambda e, f=fam: toggle(f.classes, e.value)
                        ).mark(f"family-{fam.name}")
                        fam_boxes[fam.name] = fb
                        ui.label(fam.name).classes("font-bold")
                        ui.label(f"{fam.title} ({fam.kind})").classes("text-grey-7")
                    with ui.row().classes("gap-x-4 gap-y-0 flex-wrap"):
                        for c in fam.classes:
                            lengths = fam.lengths.get(c) or []
                            tip = f"{fam.parts[c]} part(s)" + (
                                f", lengths {', '.join(f'{v:g}' for v in lengths)} mm"
                                if lengths
                                else ""
                            )
                            boxes[c] = ui.checkbox(
                                c.split("_", 1)[1] if "_" in c else c,
                                value=c in selected,
                                on_change=lambda e, cc=c: toggle([cc], e.value),
                            ).tooltip(tip)
            with ui.card().classes("w-full"):
                ui.label("Finishes").classes("font-bold")
                ui.label(
                    "How often each finish is drawn (relative weight; 0 = never). Finishes are only a "
                    "visual variation, never a class."
                ).classes("text-xs text-grey-7")
                weights = dict(ed.get("scenegen.finish_weights") or {})
                for fid, label in ov.finishes.items():
                    with ui.row().classes("items-center no-wrap gap-2 w-full"):
                        ui.label(label).classes("w-56")
                        val = ui.label(f"{weights.get(fid, 1.0):g}").classes("w-8")

                        def put(e: Any, fid: str = fid, val: Any = val) -> None:
                            w = dict(ed.get("scenegen.finish_weights") or {})
                            if float(e.value) == 1.0:
                                w.pop(fid, None)
                            else:
                                w[fid] = float(e.value)
                            ed.set("scenegen.finish_weights", w)
                            val.set_text(f"{float(e.value):g}")

                        ui.slider(
                            min=0,
                            max=3,
                            step=0.25,
                            value=float(weights.get(fid, 1.0)),
                            on_change=put,
                        ).classes("w-64")
                self.w.field("scenegen.insert_colours", "Nyloc insert colours")
        sync()
        s = parts_mod.summary(ed.merged())
        summary.set_text(
            f"{s['classes']} classes, {s['parts']} parts in {s['families']} families selected"
        )

    # ---------------------------------------------------------------- settings tab

    def settings_tab(self) -> None:
        ed = self.ed
        ui.label(
            "Every setting of every tool, from configs/config.yaml. Hover a name for its explanation. "
            "Values you change are stored in the recipe only (bold); the reset button brings back "
            "the config.yaml value."
        ).classes("text-sm text-grey-7 max-w-5xl")
        with ui.column().classes("w-full max-w-5xl gap-2"):
            for key, (title, only) in SETTINGS_SECTIONS.items():
                if only is not None:
                    data = {k: ed.base[k] for k in only if k in ed.base}
                    prefix = ""
                else:
                    data = ed.base.get(key) or {}
                    prefix = f"{key}."
                if not data:
                    continue
                with ui.expansion(title, icon="settings", value=False).classes(
                    "w-full border rounded"
                ):
                    if ed.comments.get(key):
                        ui.label(ed.comments[key]).classes("text-xs text-grey-7")
                    for f in build_fields(data, ed.comments, prefix, skip=SKIP):
                        self.w.field(f.path, f=f)

    # ---------------------------------------------------------------- run tab

    def run_tab(self) -> None:
        p = self.ed.recipe.pipeline
        with ui.row().classes("w-full gap-4 no-wrap items-start"):
            with ui.column().classes("w-96 shrink-0 gap-2"):
                with ui.card().classes("w-full"):
                    ui.label("Steps").classes("font-bold")
                    for key in STEPS:
                        with ui.row().classes("w-full items-center no-wrap gap-2"):
                            ui.checkbox(
                                value=key in p.steps,
                                on_change=lambda e, k=key: self._toggle_step(k, e.value),
                            )
                            icon = ui.icon("radio_button_unchecked", color="grey").classes(
                                "text-lg"
                            )
                            with ui.column().classes("grow gap-0"):
                                ui.label(TITLES[key]).classes("text-sm")
                                bar = ui.linear_progress(value=0, show_value=False).props(
                                    "rounded size=6px"
                                )
                                txt = ui.label("").classes("text-xs text-grey-7")
                            self.status_widgets[key] = (icon, bar, txt)
                    self.fresh = ui.checkbox(
                        "Start fresh (delete this dataset's renders and labels first)"
                    )
                    with ui.row().classes("gap-2"):
                        self.run_btn = (
                            ui.button("Run", icon="play_arrow", on_click=self.start)
                            .props("color=positive")
                            .mark("run")
                        )
                        self.stop_btn = ui.button("Stop", icon="stop", on_click=self.stop).props(
                            "color=negative"
                        )
                    ui.label(
                        "Parts, rest poses and renders already made are reused; only what is missing is built."
                    ).classes("text-xs text-grey-7")
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center w-full"):
                        ui.label("Checks").classes("font-bold")
                        ui.space()
                        ui.button(icon="refresh", on_click=self.refresh_checks).props(
                            "flat dense round"
                        )
                    self.checks = ui.column().classes("gap-1")
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center w-full"):
                        ui.label("Outputs").classes("font-bold")
                        ui.space()
                        ui.button(icon="refresh", on_click=self.refresh_outputs).props(
                            "flat dense round"
                        )
                    self.outputs = ui.column().classes("gap-1 w-full").mark("outputs")
            with ui.column().classes("grow gap-2 min-w-0"):
                self.log = ui.log(max_lines=3000).classes("w-full h-96 text-xs")
                with ui.row().classes("items-center w-full"):
                    ui.label("Previews").classes("font-bold")
                    ui.space()
                    ui.button(icon="refresh", on_click=self.refresh_previews).props(
                        "flat dense round"
                    )
                with ui.grid(columns=2).classes("w-full gap-2"):
                    for key, caption in (
                        ("render", "Latest render"),
                        ("post", "Post-processed (with labels)"),
                        ("parts", "Parts"),
                        ("restposes", "Rest poses"),
                        ("labels_train", "Labels (train)"),
                        ("labels_val", "Labels (val)"),
                    ):
                        with ui.card().classes("w-full p-1"):
                            ui.label(caption).classes("text-xs font-bold")
                            img = ui.image().classes("w-full")
                            empty = ui.label("not made yet").classes("text-xs text-grey-6")
                            self.preview_images[key] = (img, empty)
        self.refresh_checks()
        self.refresh_previews()
        self.refresh_outputs()
        self.show_status()

    def refresh_outputs(self) -> None:
        """Where the renders and the exported datasets are, with open/copy buttons."""
        ed = self.ed
        run_dir = ed.recipe.run_dir(ed.base)
        ds = ed.recipe.dataset_dir(ed.base)
        self.outputs.clear()
        with self.outputs:
            _output_row("Renders", run_dir, "Rendered and post-processed scenes.")
            _output_row("Dataset folder", ds, "Everything the label export wrote.")
            files = dataset_outputs(ds)
            if not files:
                ui.label("No dataset exported yet: run the Labels step.").classes(
                    "text-xs text-grey-7"
                )
                return
            ui.label(
                "To train with model-trainer: Datasets > Import, and paste the path of a .yaml file "
                "(Copy path). Each .yaml is a separate dataset for a different model."
            ).classes("text-xs text-grey-8")
            for name, path, text in files:
                _output_row(name, path, text)

    def _toggle_step(self, key: str, on: bool) -> None:
        p = self.ed.recipe.pipeline
        steps = set(p.steps) | {key} if on else set(p.steps) - {key}
        p.steps = [s for s in STEPS if s in steps]
        self.ed.changed()

    # ---------------------------------------------------------------- actions

    def switch(self, name: str | None) -> None:
        if not name or name == self.ed.recipe.name:
            return
        if self.ed.dirty:
            self._confirm_discard(
                lambda: self._load(name),
                revert=lambda: self.recipe_select.set_value(self.ed.recipe.name),
            )
        else:
            self._load(name)

    def _load(self, name: str) -> None:
        try:
            recipe = load_recipe(name)
        except RecipeError as exc:
            ui.notify(str(exc), type="negative")
            return
        self.ed.recipe = recipe
        self.ed.saved = json.dumps(recipe.to_dict(), sort_keys=True)
        app.storage.general["last_recipe"] = name
        self.status_widgets.clear()
        self.preview_images.clear()
        self.body.refresh()
        self.ed.changed()

    def _confirm_discard(self, then: Callable[[], None], revert: Callable[[], None]) -> None:
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Recipe '{self.ed.recipe.name}' has unsaved changes.")
            with ui.row():
                ui.button("Save", on_click=lambda: (self.save(), dlg.close(), then()))
                ui.button("Discard", on_click=lambda: (dlg.close(), then())).props("flat")
                ui.button("Cancel", on_click=lambda: (dlg.close(), revert())).props("flat")
        dlg.open()

    def save(self) -> None:
        try:
            path = self.ed.save()
        except RecipeError as exc:
            ui.notify(str(exc), type="negative")
            return
        ui.notify(f"saved {path}", type="positive")

    def new_dialog(self) -> None:
        with ui.dialog() as dlg, ui.card().classes("w-96"):
            ui.label("New dataset recipe").classes("font-bold")
            name = ui.input(
                "Name (letters, digits, _ - .)", value=f"{self.ed.recipe.name}_copy"
            ).classes("w-full")
            copy_cur = ui.checkbox("Start from the current recipe's settings", value=True)

            def create() -> None:
                n = (name.value or "").strip()
                if (recipes_dir() / f"{n}.yaml").exists():
                    ui.notify(f"recipe {n} already exists", type="warning")
                    return
                r = (
                    Recipe.from_dict({**self.ed.recipe.to_dict(), "name": n})
                    if copy_cur.value
                    else Recipe(n)
                )
                try:
                    r.save(recipes_dir() / f"{n}.yaml")
                except RecipeError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                dlg.close()
                self.recipe_select.set_options([p.stem for p in list_recipes()])
                self.ed.saved = json.dumps(
                    self.ed.recipe.to_dict(), sort_keys=True
                )  # do not ask again
                self.recipe_select.set_value(n)

            with ui.row():
                ui.button("Create", on_click=create)
                ui.button("Cancel", on_click=dlg.close).props("flat")
        dlg.open()

    def start(self) -> None:
        if RUN.busy:
            ui.notify("a run is already in progress", type="warning")
            return
        try:
            self.ed.save()
            plan = build_plan(self.ed.recipe, self.ed.base)
        except RecipeError as exc:
            ui.notify(str(exc), type="negative")
            return
        if not plan.steps:
            ui.notify("no steps selected", type="warning")
            return
        checks = self.refresh_checks()
        if any(c.level == "error" for c in checks):
            ui.notify("fix the errors under Checks first", type="negative")
            return
        self.log.clear()
        runner = Runner(plan, on_line=RUN.push)
        RUN.runner = runner
        RUN.thread = threading.Thread(
            target=runner.run, kwargs={"fresh": bool(self.fresh.value)}, daemon=True
        )
        RUN.thread.start()
        ui.notify(f"started {plan.recipe.name}", type="info")

    def stop(self) -> None:
        if RUN.runner is not None and RUN.busy:
            RUN.runner.cancel()
            ui.notify("stopping…", type="warning")

    def refresh_checks(self) -> list:
        try:
            checks = preflight(build_plan(self.ed.recipe, self.ed.base))
        except Exception as exc:  # show, do not crash the page
            checks = [Check("error", str(exc))]
        if hasattr(self, "checks"):
            self.checks.clear()
            colours = {"ok": "positive", "warning": "warning", "error": "negative"}
            icons = {"ok": "check", "warning": "warning", "error": "error"}
            with self.checks:
                for c in checks:
                    with ui.row().classes("no-wrap items-start gap-1"):
                        ui.icon(icons[c.level], color=colours[c.level]).classes("mt-1")
                        ui.label(c.message).classes("text-xs")
        return checks

    def refresh_previews(self) -> None:
        ed = self.ed
        run_dir = ed.recipe.run_dir(ed.base)
        ds = ed.recipe.dataset_dir(ed.base)
        paths = {
            "render": latest_render(run_dir),
            "post": _newest(run_dir / "show_*.png"),
            "parts": run_dir / "previews" / "parts.png",
            "restposes": run_dir / "previews" / "restposes.png",
            "labels_train": ds / "preview_train.jpg",
            "labels_val": ds / "preview_val.jpg",
        }
        for key, (img, empty) in self.preview_images.items():
            im = thumb(paths.get(key))
            if im is not None:
                img.set_source(im)
            img.set_visibility(im is not None)
            empty.set_visibility(im is None)

    # ---------------------------------------------------------------- polling

    def show_status(self) -> None:
        runner = (
            RUN.runner
            if RUN.runner and RUN.runner.plan.recipe.name == self.ed.recipe.name
            else None
        )
        if runner is not None:
            steps, progress = runner.status["steps"], runner.progress()
        else:
            st = read_status(self.ed.recipe.run_dir(self.ed.base))
            steps, progress = st.get("steps", {}), {}
        for key, (icon, bar, txt) in self.status_widgets.items():
            s = steps.get(key, {})
            state = s.get("state", "pending")
            name, colour = STATE_ICONS.get(state, STATE_ICONS["pending"])
            icon.name = name
            icon.props(f"color={colour}")
            done, total = progress.get(key, (s.get("commands_done", 0), s.get("commands_total", 0)))
            frac = 1.0 if state == "done" else (done / total if total else 0.0)
            bar.set_value(min(1.0, frac))
            secs = s.get("seconds")
            label = f"{done}/{total}" if total and state == "running" else state
            if secs is not None and state != "running":
                label += f" in {_duration(secs)}"
            txt.set_text(label)
        busy = RUN.busy
        if hasattr(self, "run_btn"):
            self.run_btn.set_enabled(not busy)
            self.stop_btn.set_enabled(busy)

    def tick(self) -> None:
        lines, self.log_pos = RUN.since(self.log_pos)
        for line in lines:
            self.log.push(line)
        self.show_status()
        self._ticks = getattr(self, "_ticks", 0) + 1
        if RUN.busy and self._ticks % 10 == 0:  # every 5 s while running
            self.refresh_previews()
        if not RUN.busy and getattr(self, "_was_busy", False):
            self.refresh_previews()
            self.refresh_checks()
        self._was_busy = RUN.busy


def _query(name: str) -> str:
    """URL parameter of the page (?tab=run&recipe=preview), or ''."""
    try:
        return ui.context.client.request.query_params.get(name) or ""
    except Exception:  # no request (tests)
        return ""


def _calibration_summary(data: dict[str, Any]) -> list[tuple[str, str]]:
    rows = []
    intr = data.get("intrinsics") or {}
    if "rms_px" in intr:
        rows.append(("Reprojection RMS", f"{intr['rms_px']:.3f} px"))
    if "model" in intr:
        rows.append(("Distortion model", str(intr["model"])))
    roi = data.get("roi") or {}
    if roi:
        rows.append(
            ("ROI", " ".join(f"{k}={v}" for k, v in roi.items() if not isinstance(v, (list, dict))))
        )
    for key in ("px_per_mm", "camera_height_mm", "created"):
        if key in data:
            rows.append((key.replace("_", " "), str(data[key])))
    return rows or [("contents", ", ".join(sorted(data))[:200])]


def _size_key(size: str) -> float:
    try:
        return float(size.lstrip("MS"))
    except ValueError:
        return 999.0


def _duration(seconds: float) -> str:
    s = int(seconds)
    return (
        f"{s // 3600}h {s % 3600 // 60:02d}m"
        if s >= 3600
        else (f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s")
    )


def page() -> None:
    BuilderPage().build()


def main(port: int = 8080, show: bool = True) -> None:
    ui.run(
        root=page,
        title="screw-synth dataset builder",
        port=port,
        show=show,
        reload=False,
        favicon="🔩",
        storage_secret="screw-synth-local",
    )


def open_in_explorer(path: Path) -> None:
    """Open a folder (or the folder holding a file) in the file manager of this computer."""
    import os
    import subprocess
    import sys

    target = path if path.is_dir() else path.parent
    if not target.exists():
        ui.notify(f"{target} does not exist yet", type="warning")
        return
    if sys.platform == "win32":
        if path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def _output_row(label: str, path: Path, text: str) -> None:
    exists = path.exists()
    with ui.column().classes("w-full gap-0"):
        with ui.row().classes("items-center w-full no-wrap gap-1"):
            ui.label(label).classes("text-sm font-medium" + ("" if exists else " text-grey-6"))
            ui.space()
            ui.button(icon="folder_open", on_click=lambda p=path: open_in_explorer(p)).props(
                "flat dense round size=sm"
            ).tooltip("Open in the file explorer")
            ui.button(
                icon="content_copy",
                on_click=lambda p=path: (ui.clipboard.write(str(p)), ui.notify(f"Copied {p}")),
            ).props("flat dense round size=sm").tooltip("Copy the path")
        ui.label(text).classes("text-xs text-grey-7")
        ui.label(str(path)).classes("text-xs text-grey-6 break-all")
