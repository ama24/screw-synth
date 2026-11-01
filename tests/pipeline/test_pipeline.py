import sys
import threading
import time

import pytest

from screwsynth.catalogue.schema import catalogue_from_config
from screwsynth.common.config import Config, config_path
from screwsynth.common.io import read_json, read_yaml
from screwsynth.pipeline import cli, forms, parts, recipe, runner, steps
from screwsynth.pipeline.recipe import Recipe, RecipeError, deep_merge, diff


@pytest.fixture
def base(tmp_path):
    cfg = recipe.load_base_config()
    cfg["paths"] = {
        **cfg["paths"],
        "output_dir": str(tmp_path / "out"),
        "datasets_dir": str(tmp_path / "ds"),
    }
    return cfg


@pytest.fixture
def no_env(monkeypatch):
    """Skip the pip-freeze environment snapshot (slow) in runner tests."""
    monkeypatch.setattr(runner, "write_environment", lambda path: None)


# ------------------------------------------------------------------------------ recipes


def test_recipe_round_trip(tmp_path):
    r = Recipe("run01", "test", config={"scenegen": {"objects": [5, 9]}})
    r.pipeline.scenes, r.pipeline.preview = 42, True
    path = r.save(tmp_path / "run01.yaml")
    back = recipe.load_recipe(path)
    assert back.to_dict() == r.to_dict()
    assert back.pipeline.scenes == 42 and back.config["scenegen"]["objects"] == [5, 9]


@pytest.mark.parametrize(
    "data",
    [
        {"name": "bad name/"},
        {"name": "ok", "pipeline": {"camera": "webcam"}},
        {"name": "ok", "pipeline": {"steps": ["render", "train"]}},
        {"name": "ok", "pipeline": {"formats": ["yolo"]}},
        {"name": "ok", "pipeline": {"scenes": 0}},
        {"name": "ok", "pipeline": {"sceens": 10}},
        {"name": "ok", "extra": 1},
    ],
)
def test_recipe_validation(data):
    with pytest.raises(RecipeError):
        Recipe.from_dict(data)


def test_merge_and_diff_are_inverse():
    base = {"a": 1, "s": {"x": [1, 2], "y": {"k": "v", "j": 0}}, "t": True}
    full = {"a": 1, "s": {"x": [1, 3], "y": {"k": "v", "j": 5}}, "t": True, "new": {"z": 1}}
    d = diff(base, full)
    assert d == {"s": {"x": [1, 3], "y": {"j": 5}}, "new": {"z": 1}}
    assert deep_merge(base, d) == full
    assert base["s"]["x"] == [1, 2]  # inputs untouched


def test_set_and_remove_path_prunes():
    d: dict = {}
    recipe.set_path(d, "scenegen.render.samples", 32)
    recipe.set_path(d, "scenegen.objects", [1, 2])
    recipe.remove_path(d, "scenegen.render.samples")
    assert d == {"scenegen": {"objects": [1, 2]}}
    recipe.remove_path(d, "scenegen.objects")
    assert d == {}


def test_merged_config_has_per_run_classes(base):
    r = Recipe("abc", config={"scenegen": {"render": {"samples": 8}}})
    cfg = r.merged_config(base)
    assert cfg["scenegen"]["render"]["samples"] == 8
    assert cfg["scenegen"]["render"]["denoiser"] == base["scenegen"]["render"]["denoiser"]
    assert cfg["paths"]["classes"].endswith("abc/classes.yaml")


# ------------------------------------------------------------------------------ parts


def test_extra_rules_select_parts(base):
    every = catalogue_from_config(Config(base))
    fams = {p.family for p in every.enabled_parts}
    assert "DIN912" in fams
    cfg = deep_merge(base, {"catalogue": {"extra_rules": [{"match": "DIN912", "enabled": False}]}})
    cat = catalogue_from_config(Config(cfg))
    assert "DIN912" not in {p.family for p in cat.enabled_parts}
    assert len(cat.enabled_parts) < len(every.enabled_parts)


def test_rules_for_selection_round_trip(base):
    ov = parts.overview(base)
    every = set(ov.classes)
    din912 = set(next(f for f in ov.families if f.name == "DIN912").classes)
    want = (every - din912) - {"DIN934_M2", "DIN934_M10"}
    keep = {"match": "DIN933_M8x7?", "enabled": False}  # a finer hand-written rule survives
    rules = parts.rules_for_selection(ov, want, [keep, {"match": "DIN912", "enabled": True}])
    assert {"match": "DIN912", "enabled": False} in rules
    assert {"match": "DIN934_M2", "enabled": False} in rules
    assert rules[-1] == keep and len(rules) == 4
    cfg = deep_merge(base, {"catalogue": {"extra_rules": rules}})
    assert parts.selected_classes(cfg) == want
    assert parts.rules_for_selection(ov, every) == []  # catalogue.yaml as is: no rules


def test_summary_counts(base):
    s = parts.summary(base)
    assert s["classes"] == len(read_yaml(config_path().parent / "classes.yaml")["names"])
    assert s["parts"] >= s["classes"] >= s["families"] > 0


# ------------------------------------------------------------------------------ forms


def test_comments_and_fields(base):
    comments = forms.config_comments(config_path())
    assert "random seed" in comments["seed"].lower()
    assert comments["scenegen.render.denoiser"].startswith("OPTIX | INTEL")
    assert "trimesh" in comments["restposes.com_samples"]  # block comment above a first child
    assert comments["restposes"].startswith("04 restposes")  # section header after a nested key
    fields = {f.key: f for f in forms.build_fields(base["scenegen"], comments, "scenegen.")}
    assert fields["objects"].kind == "pair"
    assert fields["materials"].kind == "yaml"
    render = {f.key: f for f in fields["render"].children}
    assert render["samples"].kind == "int" and render["depth"].kind == "bool"
    assert render["denoiser"].kind == "choice" and render["denoiser"].choices == [
        "OPTIX",
        "INTEL",
        "none",
    ]
    dist = next(
        f
        for f in forms.build_fields(base["postprocess"], comments, "postprocess.")
        if f.key == "distortion"
    )
    interp = {f.key: f for f in dist.children}["rgb_interpolation"]
    assert interp.kind == "choice"  # options taken from the parent's comment


def test_coerce():
    f = forms.Field("a.b", "b", "int", 3)
    assert forms.coerce(f, 4.0) == 4
    with pytest.raises(ValueError):
        forms.coerce(f, 4.5)
    pair = forms.Field("a.p", "p", "pair", [1, 2])
    assert forms.coerce(pair, (3.0, 4.0)) == [3, 4]
    y = forms.Field("a.y", "y", "yaml", {})
    assert forms.coerce(y, "{zinc_clear: 2}") == {"zinc_clear": 2}
    with pytest.raises(ValueError):
        forms.coerce(y, "{unclosed")


# ------------------------------------------------------------------------------ plan


def test_plan_commands(base):
    r = Recipe("p1")
    r.pipeline.scenes, r.pipeline.chunk, r.pipeline.preview = 450, 200, True
    r.pipeline.render_workers = 1  # explicit: "auto" depends on this computer's GPU
    r.pipeline.formats = ["seg", "crops"]
    plan = steps.build_plan(r, base)
    assert [s.key for s in plan.steps] == list(recipe.STEPS)
    render = plan.step("render").commands()
    assert [c[c.index("--scenes") + 1] for c in render] == ["200", "200", "50"]
    assert [c[c.index("--start") + 1] for c in render] == ["0", "200", "400"]
    assert all("--nominal-camera" in c and "--preview" in c and "--resume" in c for c in render)
    assert all(str(plan.config_path) in c for c in render)
    labels = plan.step("labels").commands()
    assert labels[0][labels[0].index("--formats") + 1] == "seg,crops"
    assert plan.dataset_dir in plan.step("labels").clean
    r.pipeline.camera, r.pipeline.steps = "calibration", ["render"]
    plan = steps.build_plan(r, base)
    assert [s.key for s in plan.steps] == ["render"]
    assert all("--nominal-camera" not in c for c in plan.step("render").commands())


def test_preflight_errors(base, tmp_path):
    base["paths"]["calibration_dir"] = str(tmp_path / "no_calibration")
    r = Recipe("p2")
    r.pipeline.camera, r.pipeline.steps = "calibration", ["render", "labels"]
    levels = {c.message.split(":")[0]: c.level for c in steps.preflight(steps.build_plan(r, base))}
    msgs = " ".join(
        c.message for c in steps.preflight(steps.build_plan(r, base)) if c.level == "error"
    )
    assert "calibration" in msgs and "classes.yaml" in msgs
    assert any(v == "error" for v in levels.values())


# ------------------------------------------------------------------------------ runner


def _plan(base, cmds_per_step: dict[str, list[list[str]]]):
    r = Recipe("fake")
    plan = steps.build_plan(r, base)
    plan.steps = [steps.Step(k, k, (lambda c=c: c)) for k, c in cmds_per_step.items()]
    return plan


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_runner_success_and_log(base, no_env):
    plan = _plan(
        base, {"a": [_py("print('hello')"), _py("print('world')")], "b": [_py("print('b')")]}
    )
    lines: list[str] = []
    assert runner.Runner(plan, on_line=lines.append).run()
    assert "hello" in lines and "world" in lines and "b" in lines
    st = read_json(steps.status_path(plan.run_dir))
    assert st["state"] == "done"
    assert st["steps"]["a"] == {
        **st["steps"]["a"],
        "state": "done",
        "commands_done": 2,
        "commands_total": 2,
    }
    log_lines = (plan.run_dir / "pipeline.log").read_text(encoding="utf-8").splitlines()
    assert log_lines.count("hello") == 1 and any(ln.startswith("$ ") for ln in log_lines)
    assert read_yaml(plan.config_path)["paths"]["classes"].endswith("fake/classes.yaml")


def test_runner_stops_at_failure(base, no_env):
    plan = _plan(
        base,
        {
            "a": [_py("import sys; print('boom'); sys.exit(3)"), _py("print('never')")],
            "b": [_py("print('never either')")],
        },
    )
    lines: list[str] = []
    assert not runner.Runner(plan, on_line=lines.append).run()
    assert "never" not in lines and "never either" not in lines
    st = read_json(steps.status_path(plan.run_dir))
    assert st["state"] == "failed"
    assert st["steps"]["a"]["state"] == "failed" and st["steps"]["a"]["returncode"] == 3
    assert st["steps"]["b"]["state"] == "skipped"


def test_runner_cancel_kills_command(base, no_env):
    plan = _plan(base, {"slow": [_py("import time; print('start', flush=True); time.sleep(60)")]})
    lines: list[str] = []
    run = runner.Runner(plan, on_line=lines.append)
    t = threading.Thread(target=run.run)
    t0 = time.perf_counter()
    t.start()
    while "start" not in lines and time.perf_counter() - t0 < 20:
        time.sleep(0.05)
    run.cancel()
    t.join(20)
    assert not t.is_alive() and time.perf_counter() - t0 < 30
    assert run.status["state"] == "cancelled"


def test_runner_fresh_removes_outputs(base, no_env):
    plan = _plan(base, {"a": [_py("pass")]})
    (plan.run_dir / "scenes" / "000000").mkdir(parents=True)
    plan.dataset_dir.mkdir(parents=True)
    assert runner.Runner(plan).run(fresh=True)
    assert not (plan.run_dir / "scenes").exists() and not plan.dataset_dir.exists()


def test_environment_snapshot(tmp_path):
    runner.write_environment(tmp_path / "env.txt")
    text = (tmp_path / "env.txt").read_text(encoding="utf-8")
    assert "git commit:" in text and "pyyaml" in text.lower()


# ------------------------------------------------------------------------------ cli


def test_cli_new_list_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "recipes_dir", lambda: tmp_path)
    assert cli.main(["new", "quick", "--preview"]) == 0
    r = recipe.load_recipe(tmp_path / "quick.yaml")
    assert r.pipeline.preview and r.pipeline.scenes == 10
    with pytest.raises(SystemExit):
        cli.main(["new", "quick"])
    assert cli.main(["new", "copy", "--from", str(tmp_path / "quick.yaml")]) == 0
    assert recipe.load_recipe(tmp_path / "copy.yaml").pipeline.preview
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "quick" in out and "copy" in out
    cli.main(["plan", str(tmp_path / "quick.yaml"), "--steps", "catalogue"])
    out = capsys.readouterr().out
    assert "screwsynth.catalogue.cli validate" in out and "blenderproc" not in out
    with pytest.raises(SystemExit):
        cli.main(["plan", str(tmp_path / "quick.yaml"), "--steps", "train"])


def test_shipped_recipes_are_valid():
    for path in recipe.list_recipes():
        recipe.load_recipe(path)


def test_gui_imports():
    pytest.importorskip("nicegui")
    from screwsynth.pipeline import gui

    ed = gui.Editor(Recipe("x"))
    assert not ed.dirty
    ed.set("scenegen.render.samples", 8)
    assert (
        ed.dirty
        and ed.overridden("scenegen.render.samples")
        and ed.get("scenegen.render.samples") == 8
    )
    ed.set("scenegen.render.samples", ed.base["scenegen"]["render"]["samples"])  # back to default
    assert ed.recipe.config == {}
    assert gui._step(0.4) == 0.01 and gui._step(64) == 1


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_gui_parts_selection_saves_rules(tmp_path, monkeypatch):
    """Simulated user: untick a family in the Parts tab, save, and the recipe gets the rule."""
    pytest.importorskip("nicegui")
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    Recipe("guitest").save(tmp_path / "guitest.yaml")
    async with user_simulation(root=gui.page) as user:
        await user.open("/?recipe=guitest&tab=parts")
        await user.should_see("screw-synth dataset builder")
        await user.should_see("106 classes")  # 105 + UNKNOWN
        user.find(kind=ui.checkbox, marker="family-DIN912").click()
        await user.should_see("98 classes")  # DIN912 has 8 sizes
        await user.should_see("unsaved")
        user.find(marker="save").click()
    saved = recipe.load_recipe(tmp_path / "guitest.yaml")
    assert saved.config["catalogue"]["extra_rules"] == [{"match": "DIN912", "enabled": False}]


@pytest.mark.anyio
async def test_gui_run_button_runs_plan(tmp_path, monkeypatch, base, no_env):
    """Simulated user: Run starts the plan in the background and the step status updates."""
    pytest.importorskip("nicegui")
    import asyncio

    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    Recipe("runtest").save(tmp_path / "runtest.yaml")

    def fake_plan(r, base_cfg=None):
        plan = steps.build_plan(r, base)
        plan.steps = [
            steps.Step("catalogue", "c", lambda: [_py("print('fake catalogue ran')")]),
            steps.Step("parts", "p", lambda: [_py("print('fake parts ran')")]),
        ]
        return plan

    monkeypatch.setattr(gui, "build_plan", fake_plan)
    monkeypatch.setattr(gui, "preflight", lambda plan: [])
    async with user_simulation(root=gui.page) as user:
        await user.open("/?recipe=runtest&tab=run")
        await user.should_see("Outputs")
        await user.should_see("No dataset exported yet")
        user.find(marker="run").click()
        for _ in range(100):
            if not gui.RUN.busy and gui.RUN.runner is not None:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.0)  # one more GUI tick
        assert gui.RUN.runner.status["state"] == "done"
        await user.should_see("fake parts ran")  # streamed into the log
    st = read_json(steps.status_path(tmp_path / "out" / "runtest"))
    assert st["steps"]["parts"]["state"] == "done"


@pytest.mark.anyio
async def test_gui_background_colours(tmp_path, monkeypatch):
    """Background tab: add a colour, then remove the default; the recipe keeps exactly that."""
    pytest.importorskip("nicegui")
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    Recipe("bgtest").save(tmp_path / "bgtest.yaml")
    async with user_simulation(root=gui.page) as user:
        await user.open("/?recipe=bgtest&tab=background")
        await user.should_see("Solid colours")
        await user.should_see("silicone_blue")
        user.find("Add colour").click()
        await user.should_see("colour_2")
        user.find(kind=ui.button, marker="remove-colour-0").click()
        user.find(marker="save").click()
    saved = recipe.load_recipe(tmp_path / "bgtest.yaml")
    assert saved.config["scenegen"]["plate"]["colours"] == [{"name": "colour_2", "hex": "#808080"}]


@pytest.mark.anyio
async def test_gui_background_switches(tmp_path, monkeypatch):
    """Background tab: solid colours on and photos off by default; the switches save to the recipe."""
    pytest.importorskip("nicegui")
    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    Recipe("swtest").save(tmp_path / "swtest.yaml")
    async with user_simulation(root=gui.page) as user:
        await user.open("/?recipe=swtest&tab=background")
        await user.should_see("use solid colours")
        await user.should_see("use surface photos")
        user.find(marker="enable-images").click()
        user.find(marker="enable-colours").click()
        user.find(marker="save").click()
    saved = recipe.load_recipe(tmp_path / "swtest.yaml")
    assert saved.config["scenegen"]["plate"]["background_enabled"] == {
        "calibrated_photo": True,
        "images": True,
        "colours": False,
    }


@pytest.mark.anyio
async def test_gui_camera_views_and_environment(tmp_path, monkeypatch):
    """Camera tab: a view setting switched from fixed to range is saved."""
    pytest.importorskip("nicegui")
    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    Recipe("viewtest").save(tmp_path / "viewtest.yaml")
    async with user_simulation(root=gui.page) as user:
        await user.open("/?recipe=viewtest&tab=camera")
        await user.should_see("Camera views per layout")
        await user.should_see("Rig view only (the usual dataset).")
        tilt = user.find(marker="mode-scenegen.camera.tilt_deg").elements.pop()
        tilt.value = "range"
        await user.should_see("Varied views")
        user.find(marker="save").click()
    saved = recipe.load_recipe(tmp_path / "viewtest.yaml")
    assert saved.config["scenegen"]["camera"]["tilt_deg"] == [0, 30]


@pytest.mark.anyio
async def test_gui_environment_tab_offline(tmp_path, monkeypatch):
    pytest.importorskip("nicegui")
    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    r = Recipe("envtest")
    r.config = {"scenegen": {"camera": {"tilt_deg": [0, 30], "height_mm": [250, 400]}}}
    r.save(tmp_path / "envtest.yaml")
    async with user_simulation(root=gui.page) as user:
        # every tab is built: a range where config.yaml has one number (or a word) must not crash
        await user.open("/?recipe=envtest&tab=environment")
        await user.should_see("Poly Haven")
        await user.should_see("Off: the scene is lit as usual (shroud, LED panel).")
        await user.should_see("HDRI environment (off by default)")
        await user.should_see("Off: the scene is lit as usual (shroud, LED panel).")


def test_dataset_outputs_explain_each_file(tmp_path):
    assert steps.dataset_outputs(tmp_path) == []
    for f in ("seg.yaml", "crops.yaml"):
        (tmp_path / f).write_text("names: {}\n")
    (tmp_path / "coco").mkdir()
    out = steps.dataset_outputs(tmp_path)
    assert [o[0] for o in out] == ["seg.yaml", "crops.yaml", "coco/"]
    assert "Stage 1" in out[0][2] and "length" in out[1][2]


# ------------------------------------------------------------------------- parallel renders


def test_auto_render_workers_follow_each_gpu():
    g = steps.GPU
    n, slots, why = steps.auto_render_workers([g(0, "RTX 2070 SUPER", 8.0)], free_gb=40, cores=16)
    assert n == 2 and slots == [0, 0] and "2 on RTX 2070 SUPER" in why
    n, slots, _ = steps.auto_render_workers([g(0, "A", 8.0), g(1, "B", 12.0)], free_gb=64, cores=32)
    assert n == 5 and slots[:4] == [0, 1, 0, 1]  # spread over both cards
    assert steps.auto_render_workers([g(0, "A", 24.0)], free_gb=11, cores=32)[0] == 2  # RAM limit
    assert steps.auto_render_workers([g(0, "A", 24.0)], free_gb=99, cores=4)[0] == 2  # CPU limit
    assert steps.auto_render_workers([], free_gb=99, cores=32)[0] == 1
    assert steps.GPU(0, "small", 4.0).workers == 1


def test_parallel_render_plan(base, monkeypatch):
    monkeypatch.setattr(
        steps, "nvidia_gpus", lambda: [steps.GPU(0, "A", 8.0), steps.GPU(1, "B", 8.0)]
    )
    monkeypatch.setattr(steps, "free_commit_gb", lambda: 64.0)
    monkeypatch.setattr(steps.os, "cpu_count", lambda: 32)
    r = Recipe("par")
    r.pipeline.scenes, r.pipeline.chunk = 100, 200
    plan = steps.build_plan(r, base)
    step = plan.step("render")
    assert step.parallel == 4
    cmds = step.commands()
    assert len(cmds) == 4 and {c[c.index("--scenes") + 1] for c in cmds} == {"25"}
    assert [c[c.index("--worker") + 1] for c in cmds] == ["0", "1", "2", "3"]
    assert step.lane_env(0) == {"CUDA_VISIBLE_DEVICES": "0"} and step.lane_env(1) == {
        "CUDA_VISIBLE_DEVICES": "1"
    }
    r.pipeline.render_workers = 1  # explicit: one Blender, no --worker (same files as before)
    single = steps.build_plan(r, base).step("render")
    assert single.parallel == 1 and all("--worker" not in c for c in single.commands())


def test_runner_runs_lanes_side_by_side(base, no_env):
    sleep = "import os, time; time.sleep(1.0); print('gpu', os.environ.get('MT_LANE'), flush=True)"
    plan = _plan(base, {"render": [_py(sleep) for _ in range(4)]})
    plan.steps[0].parallel = 4
    plan.steps[0].lane_env = lambda k: {"MT_LANE": str(k)}
    lines: list[str] = []
    t0 = time.perf_counter()
    assert runner.Runner(plan, on_line=lines.append).run()
    assert time.perf_counter() - t0 < 3.5  # 4 x 1 s side by side, not one after another
    assert sorted(ln.split()[-1] for ln in lines if " gpu " in f" {ln}") == ["0", "1", "2", "3"]
    st = read_json(steps.status_path(plan.run_dir))
    assert st["steps"]["render"]["commands_done"] == 4 and st["steps"]["render"]["state"] == "done"


def test_parallel_lane_failure_stops_new_work(base, no_env):
    fail = _py("import sys; sys.exit(2)")
    ok = _py("print('ran')")
    plan = _plan(base, {"render": [fail, _py("import time; time.sleep(0.5)"), ok, ok]})
    plan.steps[0].parallel = 2
    assert not runner.Runner(plan, on_line=lambda line: None).run()
    st = read_json(steps.status_path(plan.run_dir))
    assert st["steps"]["render"]["state"] == "failed" and st["steps"]["render"]["returncode"] == 2


def test_failed_command_restarts_when_step_allows(base, no_env, tmp_path):
    flag = (tmp_path / "crashed").as_posix()
    crash_once = _py(
        f"import os, sys, pathlib; f = pathlib.Path({flag!r}); "
        "print('resumed') if f.exists() else (f.touch(), sys.exit(3))"
    )
    plan = _plan(base, {"render": [crash_once, _py("print('ok')")]})
    plan.steps[0].parallel, plan.steps[0].retries = 2, 2
    lines: list[str] = []
    assert runner.Runner(plan, on_line=lines.append).run()
    assert any("restarting (1/2)" in line for line in lines)
    assert any("resumed" in line for line in lines)


def test_labels_progress_counts_exported_images(base):
    plan = steps.build_plan(Recipe("lp"), base)
    step = plan.step("labels")
    assert step.progress() is None  # export not started: count finished commands
    for k in range(5):
        (plan.run_dir / "processed" / f"{k:06d}").mkdir(parents=True)
        (plan.run_dir / "processed" / f"{k:06d}" / "meta.json").write_text("{}")
    staging = plan.dataset_dir / "_images"
    (staging / "train").mkdir(parents=True)
    (staging / "val").mkdir()
    for f in ("train/a.jpg", "train/b.jpg", "val/c.jpg"):
        (staging / f).write_bytes(b"")
    assert step.progress() == (3, 5)


@pytest.mark.anyio
async def test_gui_dirt_tab_switches_parts_and_plate_separately(tmp_path, monkeypatch):
    """Dirt tab: parts and plate each have their own switch; a range changed to one value saves."""
    pytest.importorskip("nicegui")
    from nicegui.testing.user_simulation import user_simulation

    from screwsynth.pipeline import gui

    monkeypatch.setattr(recipe, "recipes_dir", lambda: tmp_path)
    monkeypatch.setattr(gui, "recipes_dir", lambda: tmp_path)
    Recipe("dirttest").save(tmp_path / "dirttest.yaml")
    async with user_simulation(root=gui.page) as user:
        await user.open("/?recipe=dirttest&tab=dirt")
        await user.should_see("Dirt on the parts")
        await user.should_see("Dirt on the plate (bed)")
        user.find(marker="switch-scenegen.dirt.plate.enabled").elements.pop().value = True
        oil = user.find(marker="mode-scenegen.dirt.parts.oil").elements.pop()
        oil.value = "fixed"
        user.find(marker="save").click()
    saved = recipe.load_recipe(tmp_path / "dirttest.yaml").config["scenegen"]["dirt"]
    assert saved["plate"]["enabled"] is True
    assert "enabled" not in saved.get("parts", {})  # the parts stay clean (config default: off)
    assert not isinstance(saved["parts"]["oil"], list)
