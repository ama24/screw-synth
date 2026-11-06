# Training the screw models with training-synth

Training now happens in [training-synth](https://github.com/ama24/training-synth); `screwsynth-train` is gone. This
page lists the settings that suit this rig. Enter them once in the training-synth wizard, then **Save as
template** to reuse them.

## Which dataset file

The dataset builder's **Run** tab → **Outputs** lists the exported files. In training-synth, go to **Datasets →
Import** and paste the path of the `.yaml`:

| Model | Import | Task in the wizard |
|---|---|---|
| Stage 1: find, outline and classify every part | `datasets/<name>/seg.yaml` | Segmentation |
| Stage 2: keypoints on single-screw crops (length) | `datasets/<name>/crops.yaml` | Pose |

The dataset includes `groups.csv`: all photos of one layout (the removal sequence) stay in the same split. Keep
**Keep the dataset's train/val split**, or, if you re-split, keep *by group* on.

## Stage 1 (seg.yaml)

| Setting | Value | Why |
|---|---|---|
| Base model | YOLO26 nano (fast start) or small (about 2.5x slower, more accurate) | 8 GB GPU at 1024 px |
| Image size | 1024 | M2 screws must stay several pixels wide |
| Epochs / patience | 100 (small: 120) / 30 | |
| Batch | -1 (auto) | |
| Cache | ram (disk if RAM is short) | |
| Workers | 4 | |
| Rotation (degrees) | 180 | Overhead camera: any orientation is real |
| Horizontal / vertical flip | 0.5 / 0.5 | Seen from above, mirrored parts look like real parts |
| Scale | 0.1 | The camera height is fixed: sizes barely change, and size *is* the class |
| Translate | 0.1 | |
| Shear / perspective | 0 / 0 | The camera looks straight down |
| Hue / saturation / brightness | 0.015 / 0.4 / 0.3 | Finishes and lighting vary; colour is not a class |
| Mosaic / close mosaic | 1.0 / 10 | |
| Mixup / copy-paste | 0 / 0 | |

**Scale must stay small.** M3 and M4 differ by only about 30 % in size. Strong zoom augmentation would teach the
model to ignore exactly the difference it has to see.

## Stage 2 (crops.yaml)

| Setting | Value | Why |
|---|---|---|
| Base model | YOLO26 nano pose | |
| Image size | 384 | The crops are 384 px at full resolution |
| Epochs / patience | 150 / 40 | |
| Rotation | 180 | |
| Scale | 0.05 | Full-resolution crops: the scale is exact |
| Mosaic / close mosaic | 0 / 0 | One screw per crop: the context does not matter |
| Other augmentation | As stage 1 | |

## Using the trained models here

The model package is in the run's `package/` folder: `best.pt` plus `model_card.json`. Set the paths in
`configs/config.yaml`:

```yaml
measure:
  stage1_weights: C:/Users/.../model-trainer-workspace/projects/<project>/runs/<run>/package/best.pt
  stage2_weights: .../package/best.pt   # or null: classical measurement only
```

`screwsynth-infer` and `screwsynth-eval` check the model card's classes against the catalogue before use.
