# 07 labels

Nobody labels anything by hand. This tool converts the rendered ground truth into training datasets:
- for stage 1: YOLO segmentation, detection, OBB and pose, plus COCO;
- for stage 2: full-resolution measurement crops.

It runs in normal Python (OpenCV).

- **Inputs:** `output/<run>/processed/*/{image, instance.png, meta.json}` (tool 06) and `configs/classes.yaml`
- **Outputs:** `datasets/<name>/`
- **Guide:** [docs/prompts/07_labels.txt](../../docs/prompts/07_labels.txt)

**Status: done.**

## Usage

```bash
screwsynth-labels export --run run01 [run02 ...] --out datasets/run01 [--formats seg,pose,obb,det,coco,crops] [--imgsz 1024]
screwsynth-labels preview --dataset datasets/run01 --n 20 [--split val]   # -> preview_val.jpg (polygons, boxes, names, keypoints)
screwsynth-labels check --dataset datasets/run01    # parse, class ids, ranges, polygon area vs mask area
screwsynth-labels stats --dataset datasets/run01    # per class / finish / size (and per length in stats.json)
screwsynth-labels merge datasets/run01 datasets/run02 --out datasets/all
```

The default formats are `seg,pose,coco`. Add `crops` for stage 2.

## Per object

1. **Visible mask:** `instance == id`.
   - The object is dropped when `visible_pixel_fraction` < `min_visible_fraction` (0.25) or when the mask has fewer
     than `min_area_px` pixels. Dropped objects are listed in `dropped.csv`.
   - Objects cut by the image border are kept if enough of them remains visible.
2. **Polygon:** the outer contours (`RETR_EXTERNAL`) of the visible mask.
   - If an occluder splits the mask into pieces, they are **merged** into one polygon with zero-width bridges, as
     Ultralytics expects. `multi_piece: largest` keeps only the biggest piece instead.
   - The polygon is simplified with Douglas-Peucker at `polygon_tolerance_px` (0.7 px at full resolution). The
     tolerance is relaxed automatically so thin parts keep at least `polygon_min_points`.
   - The round-trip IoU is > 0.97, including a 2 mm-wide object at 11.7 px/mm.
3. **Axis-aligned box** from the visible mask, and an **oriented box** from `cv2.minAreaRect` (pixel extents
   included).
4. **Keypoints:** taken from `meta.json`, already distorted by tool 06, with COCO visibility (0 = not labelled,
   1 = occluded, 2 = visible). Non-screws and keypoints outside the image are 0.

## Formats (each line is one object)

YOLO coordinates are normalised by the image size (pixel index / size, as Ultralytics rasterises them):

| Format | Line | Example |
|---|---|---|
| seg (`seg/`) | `class x1 y1 x2 y2 ...` | `5 0.412109 0.300781 0.437500 0.301758 ...` |
| det (`det/`) | `class cx cy w h` | `5 0.425781 0.312500 0.031250 0.025391` |
| obb (`obb/`) | `class x1 y1 x2 y2 x3 y3 x4 y4` | `5 0.41 0.30 0.44 0.30 0.44 0.32 0.41 0.32` |
| pose (`pose/`) | `class cx cy w h kx1 ky1 v1 kx2 ky2 v2 kx3 ky3 v3` | `5 0.4258 0.3125 0.0313 0.0254 0.4121 0.3110 2 0.4170 0.3115 2 0.4385 0.3140 2` |
| COCO (`coco/{train,val}.json`) | Polygons, `bbox [x, y, w, h]`, `keypoints [x, y, v]*3`, `num_keypoints`, `area`; categories carry the keypoint names and skeleton | for pycocotools and other tools |

- Each YOLO format is its own Ultralytics root (`<fmt>/images/{train,val}`, `<fmt>/labels/{train,val}`). The
  images are hard-linked between formats, or copied if hard links aren't possible.
- `<fmt>.yaml` is the Ultralytics dataset file (path, train, val, names). `pose.yaml` adds
  `kpt_shape: [3, 3]`, `flip_idx: [0, 1, 2]` and `kpt_names`.
- With `--val-real PATH` (or `labels.val_real`), each yaml also gets a `val_real` entry, and a `<fmt>_real.yaml`
  is written whose `val` is the real set (tool 09). Real images are never mixed into training.
- COCO coordinates are continuous pixel coordinates of the exported (stage-1) images. Instance and part ids are
  kept on each annotation.

## Resolution

Stage 1 trains on the ROI downscaled so its longest side is `stage1_imgsz` (1024). Images are written at that
size: the labels are normalised, so they stay valid, and this saves disk space and loader time. `sources.csv` maps
every exported image to its processed scene and full-resolution image, so crops can be regenerated.

## Measurement-crop dataset (stage 2, `--formats crops`)

For every **measurable** screw (`measurable` in `meta.json`, with labelled keypoints), the tool writes a
full-resolution crop centred on the object:
- **Size:** `crop_size_px` (384). With `crop_mode: scaled` (the default), the side is
  `max(384, crop_margin × object size)`, resized to 384, so long parts fit whole. `fixed` always cuts 384 px at
  full resolution (about 33 mm). Inference cuts its crops the same way (`measure/crops.py`).
- **Clamping:** crops are shifted, not padded, to stay inside the image.

Outputs:
- `crops/images/{split}/<scene>_<instance>.png`, plus YOLO-pose labels in `crops/labels/` (single class
  `0: screw`) and `crops.yaml`.
- `crops.csv` with one row per crop:
  - identity: crop file, split, scene dir, instance id, part id, class, size;
  - ground truth: **nominal length**, length mode, axis tilt, and **axis heights** (head, tip, head_top) for the
    parallax correction;
  - placement: the crop's **offset** in the full ROI image, its side and its scale;
  - keypoints: `<name>_x`, `<name>_y`, `<name>_v` in crop pixels.

To get back to the ROI image: `full = (crop + 0.5) / scale - 0.5 + offset`.

## Splits and statistics

- **Splits:** by **layout**, so lighting variants of one arrangement and the objects of one scene never end up in
  different sets. The default is `split: {train: 0.9, val: 0.1}`, deterministic from the seed.
- **`stats.json`:** images and instances per split, and instances per class, finish, size and length.
  `classes_below_min` lists the classes with fewer than `min_instances_per_class` (3000) instances: render more
  scenes, or check the class balance in tool 05.

## Tests

`pytest tests/labels` covers:
- polygon round trip IoU > 0.97 (several angles, a 2 mm-wide object, a threaded saw-tooth outline);
- merging of split masks; OBB and bbox; RLE;
- deterministic splits that never leak a layout across sets (including lighting variants);
- all formats, crops whose keypoints land on the features, drop rules, the checker catching bad labels, and the CLI
  (export, preview, check, stats, merge).

The slow test loads a pose dataset with the Ultralytics dataset loader. It is skipped when ultralytics is not
installed.
