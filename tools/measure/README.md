# 08 measure

Measurement and inference use the trained models:
- **find** each part (stage 1: detect, segment, classify);
- **measure** each screw's length (stage 2: keypoints, or classical edges), corrected for parallax and snapped to
  the nearest standard length;
- **evaluate** both against synthetic or real ground truth.

The code is in `src/screwsynth/measure`. The PC uses it for evaluation and the real-data tool. The sorter on the Pi
([sortering-pnp](https://github.com/ama24/sortering-pnp)) has its own copy and reads the per-class table from the
`parts.json` that every dataset export (and so every model package) carries.

- **Commands:** `screwsynth-infer`, `screwsynth-eval`, `screwsynth-compare-measure`
- **Config:** the `measure:` section (`stage1_weights`, `stage2_weights`, `infer.*`)
- **Measured accuracy:** [MEASUREMENT.md](MEASUREMENT.md)

## Length measurement

Both methods give the three keypoints (head_top, bearing, tip) as points on the screw axis in the image, then:

1. **Parallax correction.** Each point is back-projected as a camera ray (with the lens distortion) and intersected
   with the horizontal plane at its height above the plate.
   - The heights come from the class's rest poses (tool 04): the probability-weighted lying poses, interpolated
     between the standard lengths.
   - Since the heights depend on the length, the measurement is iterated: estimate the length, update the heights,
     recompute.
2. **Tilt correction.** The length is the **3D** distance between the two points. A lying screw is tilted, because
   its head lifts the axis.
3. **Nominal length.** Measured bearing → tip for DIN 912, DIN 933 and ISO 7380, and head_top → tip for DIN 7991.
4. **Snapping.** The raw length snaps to the nearest enabled standard length. Exactly between two lengths, the
   shorter one wins.
5. **Output.** The raw mm value, the snapped length and the residual, plus a **confidence**. The confidence is:
   - 1 on a standard length, falling to 0 at half the gap to the next one;
   - reduced when the screw is not measurable (tilted beyond 15°, or cut by the image border);
   - reduced when methods A and B disagree by more than half the gap.
6. **Size check.** The thread diameter is estimated from the shank width at full resolution and compared with the
   stage-1 size class. A mismatch is reported in `size_check`.

**Method A** (`train/infer.py`): YOLO pose on a 384 px crop around the detection, cut exactly like tool 07's
training crops (`measure/crops.py`, shared by both). With `labels.crop_mode: scaled` (the default) a part longer than
about 27 mm gets a bigger window that is shrunk to 384 px, so both ends are in the crop; keypoints are mapped back to
full-resolution pixels. `fixed` always cuts 384 px at full resolution (about 33 mm: long parts are cut off). The
model must be trained on crops of the same mode (`measure.infer.crop_mode` overrides it).

**Method B** (`train/classical.py`):
1. The screw axis is the principal axis of the stage-1 mask; the wider end is the head.
2. The coarse tip, head top and bearing come from the mask's width profile.
3. Each is refined on the image as the strongest intensity edge along lines parallel to the axis (parabolic
   sub-pixel fit).
4. **Rim correction:** the outermost visible point of an end face is its top rim (when the end points away from the
   camera's nadir) or its bottom rim (when it points towards it), not the axis point. From the part geometry, it is
   moved to the axis point behind it. 100 mm off-centre this is worth about 0.5 mm.

`--method auto` (the default) uses A when its keypoint confidence is at least `measure.infer.pose_conf`, and B
otherwise.

### Method comparison on synthetic data

```bash
screwsynth-compare-measure --dataset datasets/<name> [--stage2 runs/pose/.../best.pt] [--split val]
```

This measures every measurable screw in `crops.csv` on the true masks, so detection errors don't mix in. It
compares:
- **A with the true keypoints**, which shows what the geometry alone can reach;
- **A with the keypoint model**, when one is given;
- **B, the classical method.**

The results go to `<dataset>/measurement_comparison/comparison.md`.

Results on the development PC are in [MEASUREMENT.md](MEASUREMENT.md).

## Inference

```bash
screwsynth-infer --image capture.png --stage1 runs/seg/stage1_nano/weights/best.pt \
    --stage2 runs/pose/stage2_pose/weights/best.pt --save-vis
```

1. **Calibration** comes from `paths.calibration_dir` (or `--calibration`; `--nominal-camera` without it). Without a
   calibration the nominal camera is `scenegen.nominal_camera` (focal length, pixel size, resolution and
   **height_mm**: set it to your measured lens-to-plate height until you calibrate). `screwsynth-real` uses the
   same setting.
2. **Input:** a full sensor frame is cropped to the ROI; an image that is already the ROI is used as it is.
3. **Stage 1** runs on the ROI downscaled to 1024 px, and its polygons and boxes are mapped back to full resolution.
4. **Each screw** is measured with A and B.
5. **Output** is `<image>_result.json`. Per object it has:
   - class and size;
   - score, polygon and box;
   - keypoints (A and B);
   - `length` (raw, snapped, residual, confidence, method) and the two raw values;
   - `measurable` and `cut_by_border`;
   - the thread diameter and `size_check`;
   - **`position_mm`** (plate frame) and **`orientation_deg`**, for a pick-and-place head.

   `--save-vis` also writes an annotated image; `--show` writes it and opens it in the image viewer (needs a desktop). Unknown parts (class `UNKNOWN`) are listed but not measured.

`measure.stage1_weights` / `measure.stage2_weights` in the config set the default weights.

### ONNX and CPU latency

```bash
# export in training-synth (run page > Export > ONNX); it re-scores and times the model
```

This writes the ONNX model and `<model>.latency.json`. Measured with onnxruntime on the development PC's
i7-9700K (same architecture, untrained weights; the speed does not depend on the weights):

| Model | Input | PC CPU (mean) | Raspberry Pi 5 (estimate: 5–10× slower) |
|---|---|---|---|
| yolo26n-seg (stage 1) | 1024 px | 236 ms | 1.2–2.4 s |
| yolo26s-seg (stage 1) | 1024 px | 692 ms | 3.5–7 s |
| yolo26n-pose (stage 2) | 384 px crop | 15 ms per screw | 0.07–0.15 s per screw |

The nano stage-1 model can run on a Pi 5 at a few seconds per photo, which may be acceptable when the machine takes
one photo per picked part. On the PC's GPU it is about 10 ms.

## Evaluation

```bash
screwsynth-eval --dataset datasets/val_real --stage1 .../best.pt --stage2 .../best.pt \
    --compare-with datasets/dataset_v1
```

It reports:
1. **Detection.** Mask/box mAP per class (Ultralytics val), and the **size confusion** within each family (true size
   → predicted size: is M2 taken for M2.5?).
2. **Length.** The full pipeline runs on every full-resolution scene, and detections are matched to the true screws
   by mask IoU. Per family and size, it gives:
   - the exact-match rate after snapping;
   - the mean and max absolute error before snapping;
   - the bias.
3. **Worst cases.** The 12 worst length errors, as crops in `worst/`.
4. **Sim-to-real gap.** `--compare-with` runs the same on a second dataset (synthetic val) and puts both side by
   side.

The output goes to `<dataset>/eval_<weights>/report.md` and `evaluation.json`.

