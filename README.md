# supreme-screw-sortering — screw-synth

Synthetic data and models for detecting, classifying and **measuring** metric fasteners (screws, nuts,
washers, T-nuts) lying on the bed of a converted Ender 3, photographed by a fixed overhead Raspberry Pi
Camera Module 3.

Photorealistic training images are rendered with Blender through [BlenderProc](https://github.com/DLR-RM/BlenderProc),
labelled automatically from the render ground truth, and used to train a two-stage YOLO pipeline:

1. **Stage 1:** a segmentation model on the 1024 px downscaled region of interest (ROI) detects each part and
   classifies its family and size (e.g. `DIN912_M4`).
2. **Stage 2:** measurement on full-resolution crops. Keypoints and sub-pixel edges are parallax-corrected and
   snapped to the nearest standard length.

**Quick start:** after installing (below), run `screwsynth-gui`. Choose the camera, parts and dataset size, press
**Run**, and the whole dataset is built: parts, rest poses, renders, post-processing and labels. See
[Build a dataset](#build-a-dataset).

Project conventions are in [CLAUDE.md](CLAUDE.md). The build guide, with one prompt per tool, is in [docs/prompts/](docs/prompts/).

## Pipeline

| #  | Tool        | Purpose | Status |
|----|-------------|---------|--------|
| 01 | catalogue   | `catalogue.yaml` of every part + validator + CLI to add parts | done |
| 02 | partgen     | parametric mesh generator: render mesh, collision proxy, keypoints, `part.json` | done |
| 03 | calibration | Pi capture scripts, ChArUco board, intrinsics, extrinsics, sensor profile | done |
| 04 | restposes   | stable resting poses of each part (precomputed once) | done |
| 05 | scenegen    | BlenderProc scene generation and rendering (RGB + instance mask + `meta.json`) | implemented (Blender run pending) |
| 06 | postprocess | lens distortion (RGB and mask), noise, vignetting, compression | done |
| 07 | labels      | export to YOLO seg/detect/pose/OBB and COCO, splits, preview overlays | done |
| 08 | train       | train stage 1 and stage 2, measurement, inference, evaluation | done (training runs pending) |
| 09 | realdata    | capture and label a real validation set, compare real vs synthetic | done (first real session pending) |
| –  | pipeline    | dataset builder: recipes, one-command runner (`screwsynth-dataset`), browser GUI (`screwsynth-gui`) | done |

Each tool has code in `src/screwsynth/<tool>/` and docs in `tools/<tool>/README.md`.

## Install

### 1. Prerequisites

| What | Version | Needed for | Notes |
|---|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.10–3.12 | everything | On Windows, tick "Add python.exe to PATH" in the installer. 3.12 is what was tested. |
| [Git](https://git-scm.com/downloads) | any | getting the code | |
| NVIDIA driver | recent (≥ 535) | rendering (05), training (08) | Check with `nvidia-smi`. OptiX rendering needs a recent driver. |
| Disk space | ~10 GB | Blender, PyTorch, renders | Rendered runs grow with the number of scenes (about 5–10 MB per full-resolution scene). |
| Free RAM | ≥ 4 GB | Blender, multiprocessing | Close large applications (Fusion 360, KiCad, many Chrome tabs) before rendering or building. |
| [GitHub CLI](https://cli.github.com) | optional | opening pull requests | `winget install --id GitHub.cli -e`, then `gh auth login` |

### 2. Get the code and create the virtual environment

**Windows (PowerShell)**

```powershell
git clone https://github.com/ama24/supreme-screw-sortering.git
cd supreme-screw-sortering
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**Linux**

```bash
git clone https://github.com/ama24/supreme-screw-sortering.git
cd supreme-screw-sortering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Activate the environment (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`) in every new terminal
before using the tools.

### 3. Install the package (tools 01–07 and the GUI)

```bash
pip install -e ".[all]"
```

This installs the `screwsynth` package in editable mode, all the `screwsynth-*` commands, and every dependency
except training.

To get **exactly** the package versions this project was tested with, install from the lock file
instead:

```bash
pip install -r requirements.lock.txt
pip install -e . --no-deps
```

`requirements.lock.txt` lists the exact version of every package (pinned versions). The ranges in
`pyproject.toml` only say which versions should work. Regenerate the lock file after deliberately upgrading
a package:

```bash
python -m pip freeze --exclude-editable > requirements.lock.txt
```

Keep the comment header at the top of the file. Every run also writes `output/<dataset>/environment.txt`, which
records the git commit and the package versions used for that dataset.

You can also install only the groups you need:

| Group | Installs | For tools |
|-------|----------|-----------|
| `catalogue` | ruamel.yaml | 01 |
| `partgen` | numpy, trimesh, manifold3d, Pillow, networkx, scipy | 02, 04 |
| `calib` | opencv-contrib-python, matplotlib, Pillow | 03, 06, 07, 09 |
| `train` | ultralytics (+ PyTorch, see step 5) | 08 |
| `gui` | NiceGUI, ruamel.yaml, Pillow | dataset builder GUI |
| `dev` | pytest, ruff, pre-commit | tests, linting |
| `all` | catalogue + partgen + calib + gui + dev | everything except training |

**OpenCV:** the calibration needs the *contrib* build (`opencv-contrib-python`) for ChArUco. Ultralytics (tool 08)
pulls in plain `opencv-python`. Both packages install the same `cv2` files, so keep them at the **same version**;
the lock file does this. To check, run `python -c "import cv2; print(cv2.aruco.CharucoDetector)"`. If that fails,
run `pip install --force-reinstall --no-deps opencv-contrib-python==<version of opencv-python>`.

### 4. BlenderProc and Blender (tool 05: rendering)

BlenderProc is installed separately and is **not** a dependency of the package, because it runs its own Blender.

```bash
pip install blenderproc
blenderproc quickstart     # first run downloads Blender 4.2 (~400 MB) and renders a test image
```

- **Where Blender goes:** it is downloaded into your home folder (`~/blender` on Windows too) and reused
  afterwards.
- **PyYAML:** the render script needs PyYAML inside Blender's own Python. `generate.py` installs it on first use;
  you can also install it manually with `blenderproc pip install pyyaml`.
- **GPU:** rendering uses Cycles with OptiX. If the log says it fell back to the CPU, update the NVIDIA driver.
- **Test it:**
  `blenderproc run tools/scenegen/generate.py --run preview --preview --nominal-camera` renders 10 small scenes
  into `output/preview/`.

### 5. PyTorch + Ultralytics (tool 08: running the trained models)

Training itself happens in [model-trainer](https://github.com/ama24/model-trainer); see
[tools/measure/TRAINING.md](tools/measure/TRAINING.md) for the settings. This repository only needs Ultralytics to
*use* the models (`screwsynth-infer`, `screwsynth-eval`, the real-data tool).


Install the **CUDA build** of PyTorch first; a plain `pip install ultralytics` would pull in the CPU build. Pick
the command for your CUDA version on [pytorch.org](https://pytorch.org/get-started/locally/). For an RTX 2070
Super with a current driver:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train]"      # ultralytics, onnx, onnxruntime
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

### 6. Optional extras

| Package | Install | Used for |
|---|---|---|
| pybullet | `pip install pybullet` | `screwsynth-restposes compute --validate` (drop simulation). There is no wheel for Python 3.12 on Windows: use Python 3.10/3.11 or Linux. Otherwise the step is skipped. |
| cadquery | `pip install cadquery` | Converting STEP files directly in tool 02. Without it, export the STEP file to STL next to it. |
| coacd | `pip install coacd` | Convex decomposition of STEP parts for collision. Otherwise a single convex hull is used. |
| pycocotools | `pip install pycocotools` | COCO-style evaluation of the exported datasets |

### 7. Raspberry Pi: the `screwrig` package

This repository holds two Python packages:
- **`screwsynth`** (repository root) runs on the PC: dataset creation, calibration solving, labelling and
  measurement.
- **`screwrig`** ([rig/](rig/)) runs on the Raspberry Pi: the locked camera, the printer bed and capture sessions. It
  uses `screwsynth` for the shared parts (catalogue, config, measurement).

On the Pi (Raspberry Pi OS Bookworm, Camera Module 3 connected):

```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-numpy git
git clone https://github.com/ama24/supreme-screw-sortering.git && cd supreme-screw-sortering
python3 -m venv --system-site-packages ~/screwrig     # system packages: Picamera2 comes from apt
source ~/screwrig/bin/activate
pip install --no-deps --require-hashes -r rig/requirements-pi.txt   # pinned: exact versions + hashes
pip install --no-deps --no-build-isolation -e . -e ./rig   # screwsynth + screwrig
rpicam-hello --list-cameras                           # the IMX708 must be listed
screwrig --help
```

Copy the captures to the PC with `scp` or `rsync` (see [tools/calibration/README.md](tools/calibration/README.md)).
Details: [rig/README.md](rig/README.md).

### 8. Check the installation

```bash
pytest -m "not slow"                  # all unit tests (a few are skipped without optional packages)
screwsynth-catalogue validate         # 0 errors
screwsynth-dataset run preview        # the whole pipeline on 10 quick scenes (about 1-2 minutes)
```

Commands installed by the package:

| Command | Tool |
|---|---|
| `screwsynth-gui` | dataset builder in the browser (recipes, every setting, Run, live log and previews) |
| `screwsynth-dataset` | the same from the terminal: `run`, `plan`, `new`, `list` |
| `screwsynth-catalogue` | 01: parts, validation, classes.yaml |
| `screwsynth-partgen` | 02: meshes, collision proxies, keypoints |
| `screwsynth-calib` | 03: board, intrinsics, extrinsic, sensor, validate, recheck, report |
| `screwsynth-restposes` | 04: stable poses |
| `blenderproc run tools/scenegen/generate.py`, `python tools/scenegen/batch.py` | 05: rendering |
| `screwsynth-post` | 06: camera/sensor simulation, realism report |
| `screwsynth-labels` | 07: dataset export, preview, check |
| `screwsynth-infer` | 08: photo → parts, sizes, lengths, plate positions (JSON + image) |
| `screwsynth-eval` | 08: mAP, size confusion, length accuracy, sim-to-real gap |
| `screwsynth-compare-measure` | 08: measurement method A (keypoints) vs B (classical) on true masks |
| `screwsynth-real` | 09: sessions, inventory, pre-annotation, CVAT round trip, length matching, val_real, sim-to-real report |
| `screwrig` (Pi) | 03/09: lock camera settings, calibration captures, session capture with homing/parking, printer status |

### Troubleshooting

- **A command exits silently, or with "Memory allocation failed" / `BrokenProcessPool`.** Windows ran out of
  commit memory, usually because large applications are open. Close them. You can also limit BLAS threads
  (`$env:OPENBLAS_NUM_THREADS=1` in PowerShell) or run builds with `--workers 1`. Partgen and postprocess fall back
  to a single process automatically.
- **`screwsynth-...` is not recognised.** The virtual environment isn't active. Activate it, or reinstall with
  `pip install -e ".[all]"`.
- **`cv2.aruco` has no `CharucoDetector`.** `opencv-python` overwrote the contrib build (see the OpenCV note in
  step 3).
- **BlenderProc says the script "can only be run by blenderproc run".** Start render scripts with
  `blenderproc run ...`, not `python ...`.

## Build a dataset

A dataset is described by a **recipe**, a small file `recipes/<name>.yaml`. It records:
- the camera (nominal Pi Camera 3, or your calibration);
- how many scenes to render and at what quality;
- which parts (families and sizes) are included;
- the export formats;
- any setting that differs from `configs/config.yaml`.

`configs/config.yaml` stays the default for every dataset. A recipe stores only its differences.

**With the GUI:**

```bash
screwsynth-gui            # opens http://localhost:8080 in your browser
```

- **Dataset tab:** name, number of scenes, preview quality, seed, objects per scene, export formats.
- **Camera tab:** nominal Pi Camera 3 values, or the calibration from tool 03 (with its status).
- **Background tab:** what the surface under the parts looks like:
  - your calibrated plate photo;
  - your own surface photos (craft mat, bare table: upload them, then scale them by typing the real width or by
    clicking two ruler marks);
  - solid colours (add or remove them; the default is silicone repair-mat blue);
  - and how often each is used.
- **Parts tab:** tick families and sizes, set how often each finish appears, and see the number of classes and
  parts.
- **All settings tab:** every setting of every tool, generated from `configs/config.yaml`. Hover a name for its
  explanation. Changed values are shown in bold, and the reset button returns to the default.
- **Run tab:** pick the steps, then press Run. It shows checks (memory, BlenderProc, calibration), progress per
  step, the live log, and previews of the parts, rest poses, latest render, post-processing and labels.

**From the terminal:** this runs exactly the same steps as the GUI.

```bash
screwsynth-dataset new my_dataset --from dataset_v1   # create a recipe (or copy one)
screwsynth-dataset plan my_dataset                    # list every command a run would execute
screwsynth-dataset run my_dataset                     # build it
screwsynth-dataset run my_dataset --steps render,postprocess,labels
screwsynth-dataset run my_dataset --fresh             # delete its renders and labels first
screwsynth-dataset list                               # recipes and the state of their last run
```

A run writes:
- `output/<name>/`: `config.yaml` (the exact merged settings), renders, `pipeline.log`, `pipeline_status.json`,
  `environment.txt`;
- `datasets/<name>/`: the exported dataset.

Built parts, rest poses and rendered scenes are reused, so a stopped run continues where it stopped. Two recipes
are included:
- `preview`: 10 quick scenes, to check that everything works;
- `dataset_v1`: 3000 full-quality scenes.

Calibration (tool 03) stays a separate step on the rig. Once `data/calibration/calibration.json` exists, choose
"My calibrated camera" in the Camera tab.

## Development

```bash
pytest                 # all tests
pytest -m "not slow"   # skip Blender/GPU tests
ruff check . && ruff format --check .
pre-commit install
```

## Order of operations

The dataset builder (`screwsynth-gui` / `screwsynth-dataset`) runs steps 1, 2 and 4–7 for you. The tools run in
this order. Calibration (03) can be done in parallel with 01, 02 and 04. Until it exists,
`scenegen --nominal-camera` uses nominal Pi Camera 3 values.

1. **catalogue**: define and validate the parts, then generate `configs/classes.yaml`.
2. **partgen**: build meshes, collision proxies and keypoints into `data/assets/<part_id>/`.
3. **calibration**: on the rig, lock the camera settings, calibrate intrinsics and extrinsics, and build the sensor profile in `data/calibration/`.
4. **restposes**: compute stable poses into `data/rest_poses/`.
5. **scenegen**: render scenes with `blenderproc run tools/scenegen/generate.py ...` into `output/<run>/scenes/`.
6. **postprocess**: apply distortion, sensor noise and compression into `output/<run>/processed/`.
7. **labels**: export YOLO and COCO datasets and measurement crops into `datasets/<name>/`.
8. **train**: train stage 1 and stage 2, then evaluate and run inference.
9. **realdata**: capture and label a real validation set, then produce the sim-to-real report.

End-to-end dataset production:
`catalogue -> partgen -> restposes -> scenegen -> postprocess -> labels -> train`.
