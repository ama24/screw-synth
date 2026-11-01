# screwrig: the Raspberry Pi side

`screwrig` runs on the Raspberry Pi that sits on the rig. It drives:
- the **Pi Camera 3**, with locked exposure, gain, white balance and focus;
- the **printer bed**: homing and parking Y over USB serial.

It records the captures for calibration (tool 03) and for real-data sessions (tool 09). Later it will also run the
sorter loop: photo → detect → measure → pick.

It uses the `screwsynth` package from the repository root for the shared parts: config, catalogue, session files,
calibration recheck and measurement.

## Install (Raspberry Pi OS Bookworm, 64-bit)

```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-numpy git libgl1
git clone https://github.com/ama24/supreme-screw-sortering.git && cd supreme-screw-sortering
python3 -m venv --system-site-packages ~/screwrig     # Picamera2 comes from apt, not pip
source ~/screwrig/bin/activate
pip install -r rig/requirements-pi.txt                # screwsynth + screwrig + OpenCV + PyTorch (CPU) + Ultralytics + NCNN
pip check                                             # "No broken requirements found."
screwrig --help
```

The install takes 10–20 minutes. It needs the 64-bit OS (`uname -m` prints `aarch64`).

**PyTorch must be the CPU build.** The PyPI build for 64-bit ARM is the NVIDIA one (version ending in `+cu…`); on a
Pi 4 it stops with "Illegal instruction". `requirements-pi.txt` takes PyTorch from its CPU-only index, and
`screwrig detect` refuses to start with a `+cu` build and prints the fix. To repair an existing install:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"    # must end in +cpu
```

If even the newest CPU build stops with "Illegal instruction", pin an older one:
`pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu`.

On the PC (for tests or development): `pip install -e . -e rig/`. OpenCV comes from `screwsynth[calib]` there.

## Commands

| Command | What it does |
|---|---|
| `screwrig lock --out camera_settings.json` | Lets exposure and white balance settle, freezes them, runs a focus sweep, and saves the locked settings. Run it once per rig change. |
| `screwrig calib-capture --mode intrinsics\|extrinsic\|sensor\|single --out <dir>` | Calibration captures with the locked settings. It refuses to run if the camera does not apply them exactly. |
| `screwrig capture --session s01 --settings camera_settings.json [--recheck]` | Real-data session. Each Enter homes/parks the bed and saves a photo plus sidecar into `data/real/s01/images`. Create the session first with `screwsynth-real session new`. |
| `screwrig detect` | Loads the model once; each Enter: countdown, photo, segmentation. The result page `http://<pi>:8000` (open it in the PC's browser) always shows the latest picture with outlines and the list of parts. Files `det_0001…` in `data/real/detect`, numbering continues across runs. |
| `screwrig printer park\|status` | Homes Y (and parks it, if configured), then reports the position |

## Run the trained model

1. **Export** in model-trainer: run page > Export > **NCNN**, **fp32** (the fastest format on a Pi; fp16 gives
   little on a Pi 4, whose CPU has no fast 16-bit maths).
2. **Copy** the run's whole package folder into `models/stage1` on the Pi (see [../models/README.md](../models/README.md)).
   From PowerShell on the PC:
   ```powershell
   scp -r "<package folder>" pi@screwrig.local:~/supreme-screw-sortering/models/stage1
   ```
3. **Photo and detection, one command:**
   ```bash
   source ~/screwrig/bin/activate && cd ~/supreme-screw-sortering
   screwrig detect --nominal-camera
   ```
   It prints the address of the result page (e.g. `http://192.168.1.42:8000/`); open it in the PC's browser and
   leave it open. Then press **Enter** in the terminal for every photo: a 3 s countdown (`--countdown`), the photo
   ("photo taken, you can let go"), and the analysis. The page updates by itself. `q` quits.
   - The model is the newest one in `models/` (an NCNN export first), or `--stage1 <path>`.
   - Photos and results go to `data/real/detect/det_NNNN.jpg`, `_vis.jpg`, `_result.json`.
   - With `--settings camera_settings.json` (from `screwrig lock`) it uses the locked camera; without it, the
     automatic one (`rpicam-still`).
   - Leave out `--nominal-camera` once `data/calibration/calibration.json` exists.

   For a single photo file: `screwsynth-infer --image <photo> --stage1 <model> --nominal-camera --save-vis`.

## Printer safety

Only these G-code commands can ever be sent:
- `G28 Y` (home Y);
- `G0 Y<park_y_mm>` (park Y at the configured position);
- `M400` (wait for moves to finish);
- `M114` (report the position).

Anything else is refused before it reaches the serial port. The settings live under `rig.printer` in
`configs/config.yaml`: `port`, `baud`, `park_y_mm`, `feed_mm_min` and `timeout_s`.

## Tests

`pytest rig/tests` runs off the Pi: the camera helpers are tested on synthetic images, and the printer on a fake
serial port.
