# CLAUDE.md — screw-synth

screw-synth is a toolkit that generates photorealistic synthetic training data with Blender (through
BlenderProc) for a YOLO model that detects, segments and measures metric fasteners lying on the bed
of a converted Ender 3, seen by a fixed overhead Raspberry Pi Camera Module 3.

The full build guide (one prompt per tool) lives in [docs/prompts/](docs/prompts/). Read the prompt for
the tool you are working on before changing it.

## Workflow
- All code lives on `main`. Each piece of work (a tool from its prompt 01 to 09, a feature, a fix)
  gets a short-lived branch (`tool/<name>`, `feature/<name>`, `fix/<name>`), a pull request into
  `main`, and is deleted after merging. Do not stack branches on each other. Never push to `main`
  directly.
- Run `pytest` and `ruff check .` before every commit.

## Hardware
- Recycled Ender 3, bed area 220 x 220 mm. A 1.5-2 mm matte TPU plate (mid-tone saturated colour,
  e.g. blue or green) sits on the bed. The bed moves in Y, so it is homed/parked before every capture.
- Raspberry Pi Camera Module 3 (standard lens, IMX708, 4608 x 2592), fixed overhead at about 300 mm,
  autofocus LOCKED to a fixed lens position, exposure/gain/white balance LOCKED.
- About 11.7 px/mm at full resolution. The working area is a region of interest (ROI) of roughly
  2560 x 2560 px, defined in calibration.json.
- Diffuse LED lighting with a light shroud.
- Rendering/training PC: NVIDIA RTX 2070 Super, 8 GB VRAM.

## Objects (all are detection targets, each with its own class)
- Screws: DIN 912 (socket head cap), DIN 933 (hex head, fully threaded), ISO 7380 (button head),
  DIN 7991 (countersunk socket). Sizes M2, M2.5, M3, M4, M5, M6, M8, M10.
- Set screws DIN 913 (flat point, M2-M10, up to about 3 x d) and threaded rod pieces DIN 975
  (M2-M10, from twice the longest DIN 913 of the size, so the two never overlap in length). Rod is
  cut to any length: rendered from each size's min_length up to catalogue.rod_max_length_mm in
  5 mm steps, and measured lengths are rounded to 1 mm instead of snapped to standard lengths.
- Hex standoffs, one class per type and size: STANDOFF_MF / _FF / _MM (male-female,
  female-female, male-male), M2-M6, length measured (the hex body).
- Corner brackets for aluminium extrusion: classes BRACKET_2020, BRACKET_3030, BRACKET_4040 (not
  measured), each in two looks (styles): keyed (compact, full-length key) and cast (the common
  "2028" die-cast bracket with pockets and short tabs).
- Nuts, washers, slide-in T-nuts and drop-in T-nuts (for aluminium extrusion slots).
- UNKNOWN: one class for everything else that ends up on the plate (wood screws, lock washers,
  pins, sheet-metal brackets, plates, odd bits), so the machine can reject it. Generated
  from the catalogue's `unknown:` section; a share `scenegen.unknown_fraction` of each scene.
- Finishes: black oxide, zinc plated (clear/yellow), stainless A2/A4, plain steel, brass, nylon,
  aluminium natural and black anodised (brackets, standoffs). No coloured anodising. Finish is a
  visual variation only, never a class.
  Magnetic vs non-magnetic is NOT distinguished.
- Arrangement: a handful of parts thrown onto the plate and spread by hand: clumps, parts
  touching, some parts lying on top of others in small stacks (up to scenegen.clumped.max_layers,
  default 3; no tall piles). A lip just
  outside the camera frame stops parts, so parts are often cut by the image border.
- Operation: a picker removes one part at a time and a new photo is taken after every removal.

## Model design
- Class = family + size, e.g. DIN912_M4. Length is NOT a class; it is measured.
- Stage 1: YOLO segmentation model on the ROI downscaled to 1024 px (detect + segment + classify).
- Stage 2: measurement on full-resolution crops around each detection (keypoints and/or sub-pixel
  edges), parallax-corrected using known part geometry, snapped to the nearest standard length.
  Only parts that are free (nothing on them), lying on the plate and fully visible are measured;
  the others are picked or re-imaged later, once the parts on top are gone (ground truth:
  `free`, `support`, `lying_on_it`, `measurable` in meta.json).
- Required accuracy: exact standard length (adjacent lengths may differ by only 1 mm for M2).

## Pipeline (one tool per step, each in tools/<name> and src/screwsynth/<name>)
| #  | Tool        | Purpose |
|----|-------------|---------|
| 01 | catalogue   | catalogue.yaml of every part + validator + CLI to add parts |
| 02 | partgen     | parametric mesh generator: render mesh, collision proxy, keypoints, part.json |
| 03 | calibration | Pi capture scripts, ChArUco board, intrinsics, extrinsics, sensor profile |
| 04 | restposes   | stable resting poses of each part (precomputed once) |
| 05 | scenegen    | BlenderProc scene generation and rendering (RGB + instance mask + meta.json) |
| 06 | postprocess | lens distortion (RGB and mask), noise, vignetting, compression |
| 07 | labels      | export to YOLO seg/detect/pose/OBB and COCO, splits, preview overlays |
| 08 | measure     | measurement, inference, evaluation (training: model-trainer, see tools/measure/TRAINING.md) |
| 09 | realdata    | label a real validation set, compare real vs synthetic (capture: `screwrig` on the Pi) |

The dataset builder (`src/screwsynth/pipeline`, `screwsynth-dataset` / `screwsynth-gui`) runs 01, 02, 04,
05, 06 and 07 from one recipe (`recipes/<name>.yaml`) as subprocesses. Each step is called with
`--config output/<name>/config.yaml`. When a tool gains a step or a setting, keep it runnable
from that merged config. New config keys appear in the GUI automatically: give them a `#` comment.
Every tool reads the catalogue through `catalogue_from_config()`, so `catalogue.extra_rules`
(the recipe's part selection) applies to all of them.

## Conventions (binding for all tools)
- Units: every length in configs and metadata is in millimetres. Angles in degrees in configs,
  radians in code.
- Plate frame P: z = 0 is the top surface of the TPU plate, z points up towards the camera, x/y lie
  in the plate. The origin and axes are defined by the extrinsic calibration (tool 03).
- Blender scale: Bullet physics is unstable with millimetre-sized objects, so the Blender scene is
  uniformly scaled by WORLD_SCALE (config, default 10): blender_units = mm * 0.001 * WORLD_SCALE.
  Camera intrinsics are unaffected by a uniform scale; only translations are scaled. All metadata
  written to disk is converted back to real millimetres. Use `screwsynth.common.units`.
- Images: renders cover the ROI only, at full ROI resolution; the principal point is shifted
  accordingly. RGB as 8-bit PNG. Instance masks as 16-bit PNG, 0 = background, k = instance id
  listed in meta.json.
- Part id: `<FAMILY>_<SIZE>x<LENGTH>` for screws and standoffs (DIN912_M4x16, STANDOFF_MF_M3x10),
  `<FAMILY>_<SIZE>` for nuts/washers (DIN934_M4, DIN125A_M4), `TNUT_<SLIDE|DROP>_S<slot>_<SIZE>`
  for T-nuts (TNUT_DROP_S6_M4), `<FAMILY>_<PROFILE>_<STYLE>` for extrusion parts (BRACKET_2020_CAST,
  class BRACKET_2020; styles are look-alike variants of one class),
  `UNKNOWN_<TYPE>_<NN>` for unknown parts (UNKNOWN_WOODSCREW_03, class UNKNOWN). Family names may
  have "_"-joined words; the prefixes TNUT_ and UNKNOWN_ are reserved.
  Use `screwsynth.common.naming`.
- Class name: part id without the length (DIN912_M4). Class ids are assigned in a deterministic,
  sorted order and stored in a generated classes.yaml.
- Keypoints: every object has 3 keypoints in this order: head_top, bearing (underside of the head),
  tip. Parts without a length mark all 3 as not visible. Nominal length is bearing-to-tip for
  DIN 912, DIN 933 and ISO 7380, and head_top-to-tip for DIN 7991 (countersunk length includes the
  head). Headless parts (DIN 913: head_top = socket end, DIN 975) and standoffs (the two ends of
  the hex body) are head_top-to-tip with bearing = head_top.
- BlenderProc scripts run inside Blender's own Python. They may only import the standard library,
  numpy, blenderproc/bpy and pure-Python modules from this repo (add src/ to sys.path at the top of
  the script). Anything that needs OpenCV, trimesh, torch etc. runs in normal Python, outside
  Blender. `screwsynth.common` is deliberately pure Python (PyYAML is imported lazily) so it is
  safe to import inside Blender.
- Every tool: a CLI entry point, a README.md, unit tests (pytest), type hints, logging instead of
  print, no hard-coded paths (everything through configs/config.yaml or CLI flags).
- Randomness: every random choice takes a seed; a given seed reproduces the same output.

## Layout
- Two Python packages in this repository:
  - **`screwsynth`** (root, `src/screwsynth`) is the PC side.
  - **`screwrig`** (`rig/`, `rig/src/screwrig`) is the Raspberry Pi side: camera, printer bed, capture sessions,
    later the picker. `screwrig` imports `screwsynth`, never the other way round. Install both with
    `pip install -e . -e rig/`.
- Measurement and inference (stage 2, used on the PC and the Pi) live in `src/screwsynth/measure`.
- **Generic model training is moving to the separate repository
  [model-trainer](https://github.com/ama24/model-trainer).** The two repositories exchange files only:
  - exported datasets go to the trainer (YOLO/COCO plus `groups.csv`: image -> layout);
  - model packages come back (weights plus `model_card.json`).
- `configs/config.yaml` — one section per tool; load with `screwsynth.common.config.load_config()`.
- `recipes/` — dataset recipes (pipeline settings + config overrides).
- `requirements.lock.txt` — exact package versions; regenerate with `pip freeze --exclude-editable`
  when deliberately upgrading.
- `configs/catalogue.yaml` — parts (tool 01); `configs/classes.yaml` is generated from it.
- `data/` — calibration, plate texture, generated assets, rest poses, real captures.
- `output/` — render runs (gitignored). `datasets/` — exported datasets (gitignored).
- `src/screwsynth/common/` — shared helpers: paths, config, units, io, naming, log, seed.
- Tests marked `@pytest.mark.slow` need Blender, a GPU or large downloads; skip with `-m "not slow"`.
