# 03 calibration

This tool calibrates the real Raspberry Pi Camera Module 3 so that the Blender camera (tool 05) matches it exactly.
It also measures the sensor and lighting characteristics that tool 06 uses to make renders look like real
captures. It has two halves:

- **Pi side** (the `screwrig` package in [rig/](../../rig/)): `screwrig lock` and `screwrig calib-capture`, using
  Picamera2 on the Raspberry Pi.
- **PC side** (`screwsynth-calib`): solvers using OpenCV ≥ 4.7 ChArUco, run on the PC (or on the Pi).

- **Outputs:** `data/calibration/` contains:
  - `camera_settings.json`: the locked capture settings (written on the Pi).
  - `calibration.json`: intrinsics, extrinsics, ROI and homography. This is the only file tool 05 needs.
  - `sensor_profile.json`, `flatfield.png`, `plate_empty.png`.
  - `report.html`, `charuco_board.pdf`, `board.json`.
- **Guide:** [docs/prompts/03_calibration.txt](../../docs/prompts/03_calibration.txt)

**Status: done.** The code is tested on synthetic captures with known ground truth. The real rig has not been
calibrated yet; follow the procedure below.

## Install

- **PC:** `pip install -e ".[calib]"`
- **Pi (Raspberry Pi OS Bookworm):**

  ```bash
  sudo apt install -y python3-picamera2 python3-numpy
  python3 -m venv --system-site-packages ~/screwcam && source ~/screwcam/bin/activate
  pip install --no-deps --require-hashes -r rig/requirements-pi.txt
  pip install --no-deps --no-build-isolation -e . -e ./rig   # screwsynth + screwrig
  ```

## Step-by-step procedure

Run every `python3 ... .py` command on the Pi (in `~/calib`) and every `screwsynth-calib` command on the PC.

1. **Freeze the rig.**
   - Camera mounted and screwed tight. LED panel on, shroud closed.
   - TPU plate in place. Y axis homed and parked (the bed moves in Y).
   - From now on, don't touch the camera, the focus or the plate.
2. **Lock the camera settings** (Pi), with the empty plate under the camera:

   ```bash
   screwrig lock --out camera_settings.json --distance-mm 300
   ```

   The script:
   - lets auto-exposure and auto white balance settle;
   - freezes exposure, gain and colour gains;
   - sweeps manual focus (variance of the Laplacian in the centre, coarse then fine, with a parabolic peak fit);
   - writes `camera_settings.json`.

   Copy this file to the PC as `data/calibration/camera_settings.json`. Every later capture uses it.
3. **Print the board** (PC):

   ```bash
   screwsynth-calib board --markers
   ```

   This creates `data/calibration/charuco_board.pdf` (A4 landscape): 11 × 8 squares of 18 mm (198 × 144 mm),
   13.5 mm markers, DICT_5X5_100, and a 100 mm scale bar.
   - **Why this size:** it is as large as possible while still lying flat on the 220 mm TPU plate inside the lip
     (needed for the extrinsic step). In each intrinsic view it covers about a third of the image.
   - Print it at **100% / actual size**, without "fit to page". Check the scale bar with calipers.
   - Glue it perfectly flat onto glass or an aluminium plate.
   - Measure several squares with calipers and note the mean (for example 17.96 mm). Every solver takes it as
     `--square-mm 17.96`.
   - `--markers` also writes SVGs for the optional plate-corner ArUco markers (see `corner_marker_ids` in the
     config).
4. **Capture 25–40 intrinsic views** (Pi):

   ```bash
   screwrig calib-capture --settings camera_settings.json --out captures/intrinsics --mode intrinsics --board board.json
   ```

   Press Enter for each view and `q` to finish. After each view the script prints the live coverage map of where
   corners have been seen, and warns about blur.
   - Cover the **whole frame**, including the corners and edges.
   - Rotate the board in-plane, and tilt it by **±10° at most**. With locked focus at 300 mm, the depth of field is
     only about ±20 mm.
   - Hold the board still while capturing.
5. **Solve intrinsics** (PC), after copying the captures (`scp -r pi@<pi-host>:~/calib/captures data/real/calib`):

   ```bash
   screwsynth-calib intrinsics --images data/real/calib/intrinsics --square-mm 17.96
   ```

   - The solver rejects views that are blurry, have too few corners, or don't contain the board.
   - It chooses the standard 5-coefficient model or the rational 8-coefficient model by k-fold **held-out**
     reprojection error.
   - It reports the RMS and the worst views. The target RMS is **< 0.5 px**. If it's above that, retake the worst
     views and the views of uncovered areas, then solve again.
6. **Extrinsic: the plate frame.** Lay the board flat on the TPU plate with Y homed, then capture (Pi) and solve (PC):

   ```bash
   screwrig calib-capture --settings camera_settings.json --out captures/extrinsic --mode extrinsic
   screwsynth-calib extrinsic --image data/real/calib/extrinsic/board_flat.png --square-mm 17.96 \
       --board-thickness-mm 4.1 --area-origin X Y --area-size 220 220
   ```

   - The plate frame's origin is the board's first chessboard corner (the corner next to ChArUco id 0). x runs
     along the board's long side, z points up towards the camera, and z = 0 is the **TPU surface**.
     `--board-thickness-mm` (paper + glue + glass) lowers z = 0 from the board surface to the plate.
   - `--area-origin X Y` is the corner of the 220 × 220 mm working area in that frame, in mm. Measure it from the
     board's first corner while the board lies there. The y values are negative across the board. If the four
     corner ArUco markers are mounted (`corner_marker_ids` in the config), the area is detected from them
     automatically.
   - The solver writes `T_cam_plate`, the camera height, the plate homography, px/mm at the area centre, and the
     ROI (rounded to multiples of 32 px).
   - **Note:** at 300 mm the sensor sees only about 230 mm vertically (2592 px at about 11.3 px/mm). Keep
     `working_area.margin_mm` ≤ about 4 mm, or the ROI is clipped to the image height.
7. **Validate with caliper-measured parts.** Lay flat objects of known length on the plate: gauge blocks, a steel
   rule offcut, washers measured across their diameter, or screws lying flat. Capture with `--mode single`, then run:

   ```bash
   screwsynth-calib validate --image data/real/calib/val.png --lengths 25.02 40.01 12.00 [--order x] [--height-mm 0]
   ```

   - The command segments every object against the plate, maps its outline to the plate through undistortion and
     rays to the plane z = `--height-mm`, and compares the lengths with your caliper values.
   - It matches objects by size, or left to right in the plate frame with `--order x`.
   - The target is a **max error < 0.1–0.2 mm**. Thick objects need `--height-mm` (the height of their outline)
     because of parallax.
8. **Sensor profile.** Capture (Pi), then solve (PC):

   ```bash
   screwrig calib-capture --settings camera_settings.json --out captures/sensor --mode sensor --frames 20
   screwsynth-calib sensor --frames "data/real/calib/sensor/frame_*.png" \
       --white data/real/calib/sensor/white.png --empty data/real/calib/sensor/empty.png
   screwsynth-calib report --extrinsic-image data/real/calib/extrinsic/board_flat.png
   ```

   The capture script prompts for three things:
   - 20 frames of a static scene with a few parts;
   - the empty plate;
   - a matte white sheet covering the whole plate.

   The solver produces:
   - **Noise model:** variance = a · I + b per channel, from the per-pixel mean and variance (fitted in intensity
     bins).
   - **Flat-field:** a degree-4 2D polynomial of the white sheet, normalised to max 1.
   - **Black level:** from the frame metadata.
   - **Empty plate:** colour statistics, and `plate_empty.png`, the undistorted ROI crop that becomes the Blender
     plate texture.

   `report.html` shows every plot: coverage, residual vectors, per-view errors, the extrinsic axes and ROI, the
   noise fit and the flat-field.
9. **When to recheck.** Recheck before every capture session, and always after the camera, focus, light or plate
   has been touched or changed:

   ```bash
   screwsynth-calib recheck --image <flat-board image>
   ```

   - The board can lie anywhere on the plate. The check compares the plate **plane** (tilt, and the camera height
     above it) with the calibration.
   - It exits with code 1 and says **RECALIBRATE** above 0.15° or 0.3 mm. Then redo step 6.
   - Redo steps 2–6 if the focus or the camera mount changed. Redo step 8 if the lighting changed.

Instead of copying captures to the PC, you can also install the package on the Pi
(`pip install -e ".[calib]"`) and run `screwsynth-calib` there.

## `calibration.json` schema (all lengths in mm, pixels in the full 4608 × 2592 image)

| Key | Meaning |
|---|---|
| `image_size` | `[w, h]` of the full sensor image |
| `camera_matrix` | 3×3 K (fx, fy, cx, cy in full-image pixels) |
| `dist_coeffs` | OpenCV distortion coefficients (5 for `standard5`, 8 for `rational8`) |
| `distortion_model` | `standard5` or `rational8` (chosen by held-out error) |
| `rms_px`, `holdout_rms_px` | Reprojection RMS on all views, and k-fold held-out RMS per model |
| `views` | Per image: used / reason, corners, sharpness, `rms_px` |
| `board` | Board spec actually used (with the measured square size) |
| `extrinsic.T_cam_plate` | 4×4 plate → camera (OpenCV camera frame: x right, y down, z forward), mm |
| `extrinsic.board_thickness_mm`, `reprojection_rms_px` | Extrinsic inputs and quality |
| `extrinsic.camera_height_mm`, `camera_centre_plate_mm` | Camera centre in the plate frame |
| `extrinsic.working_area_mm` | `origin`, `size`, `margin` and `source` (config / corner_markers / centred_on_board) |
| `roi` | `{x0, y0, w, h}`: the crop that renders and stage 1 use. Its principal point is `cx - x0`, `cy - y0` |
| `px_per_mm` | At the plate surface, at the working-area centre |
| `plate_homography` | 3×3 map from plate (x, y, 1) mm at z = 0 to **undistorted** full-image pixels |
| `validation` | Last `validate` result |
| `date` | ISO timestamp |

To build the Blender camera, tool 05 takes K, `image_size`, `T_cam_plate` and `roi`, and shifts the principal
point for the ROI with `geometry.roi_camera_matrix`. Distortion is applied later by tool 06.
`screwsynth.calibration.geometry.Camera` provides `project`, `pixel_to_plate(px, height_mm)` (parallax-correct
back-projection), `homography` and `px_per_mm`, and is reused by tools 06 and 08.

## Synthetic sample set (CI)

```bash
screwsynth-calib make-sample output/calib_sample   # 14 board views, a flat board, sensor frames, truth.json
```

The tests render ChArUco boards through a known K, distortion and pose, then check the following:
- K and the undistortion field are recovered, and blurry or blank views are rejected.
- The plate frame, camera height, ROI and px/mm come out right.
- The noise model and flat-field are recovered.
- Lengths measured on a distorted synthetic image are within 0.15 mm.
- Recheck catches a height change and a tilt, but not a moved board.
- The full CLI pipeline runs end to end.
- The Pi helpers work: focus peak, settings mismatch (refuse to capture), coverage map, and capture sidecars.
