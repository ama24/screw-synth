# 09 realdata

Synthetic metrics are not trustworthy on their own: the project is judged on **real photos from the rig**. This tool
captures, labels and manages a small real dataset:
- **size:** about 100–200 images for validation, plus optionally a few hundred more from *other* sessions for
  fine-tuning;
- **use:** it feeds tool 06 (realism comparison), tool 07 (the `val_real` split) and tool 08 (evaluation and
  fine-tuning).

Inputs and outputs:
- **Inputs:** Pi captures, inventory sheets (caliper lengths), corrected CVAT annotations, models from tool 08.
- **Outputs:** `data/real/<session>/`, `datasets/val_real/`, `sim_to_real.md`.
- **Guide:** [docs/prompts/09_realdata.txt](../../docs/prompts/09_realdata.txt)

**Status: done.** It has been tested with fixtures. The Pi side (camera, printer) still needs a first real session.

## The procedure, end to end

### 1. Create a session (PC or Pi)

```bash
screwsynth-real session new --name s01 --operator alex --camera-settings camera_settings.json
```

This creates `data/real/s01/` with `session.json` (fingerprints of the camera settings and the calibration) and
`inventory.csv`.

### 2. Fill in the inventory BEFORE capturing

For each kind of part you put on the plate, write one line in `data/real/s01/inventory.csv`:

```
part_id,finish,length_mm,count,notes
DIN912_M4x16,black_oxide,15.95,3,
DIN912_M4x20,black_oxide,19.9,2,
DIN934_M4,zinc_clear,,5,
UNKNOWN,,,4,wood screws and a bracket
```

- **`length_mm`** is what the **calipers** say. It is the length truth, not pixels.
- **Empty `length_mm`:** for a screw, the nominal length from the part id is used.
- **Junk:** `UNKNOWN` covers parts that are not in the catalogue.

Tip: use different lengths of one class in a session only when they are easy to tell apart, or put one length per
class per session. That keeps the matching (step 6) automatic.

### 3. Take the photos on the sorter

The photos are taken with [sortering-pnp](https://github.com/ama24/sortering-pnp) on the Pi (Photos tab): it parks
the bed, uses the locked camera settings and saves `images/img_NNNN.png` with a JSON sidecar (session, time,
settings, frame metadata). Workflow: take a photo, remove a part, take a photo, and so on, as the machine will do.
Download the session as a zip and unpack it into `data/real/<session>/`.

### 4. Pre-annotate (once a stage-1 model exists)

```bash
screwsynth-real preannotate --session s01 --stage1 runs/seg/stage1_nano/weights/best.pt \
    [--stage2 runs/pose/stage2_pose/weights/best.pt]
screwsynth-real export-cvat --session s01
```

1. **Crop:** the photos are cropped to the calibrated ROI (`annotations/images/`).
2. **Pre-annotate:** the tool 08 pipeline proposes polygons with classes. For screws it finds lying flat, it also
   proposes the three keypoints.
3. **Export:** `export-cvat` writes:
   - `annotations/cvat.xml` (CVAT for images 1.1);
   - `annotations/cvat_labels.json` (label definitions).

   Without a model, the export contains the images with no annotations.

### 5. Correct in CVAT

Start CVAT once with Docker (Docker Desktop on Windows):

```bash
git clone https://github.com/cvat-ai/cvat && cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
# open http://localhost:8080
```

1. **Project:** create a project, open *Raw* in the label constructor, and paste `cvat_labels.json`. That gives one
   polygon label per class plus the point labels `head_top`, `bearing` and `tip`.
2. **Task:** create a task in the project with the images from `data/real/s01/annotations/images/`.
3. **Pre-annotations:** use *Upload annotations*, choose **CVAT 1.1**, and pick `cvat.xml`.
4. **Correct:**
   - every part gets a polygon of its **visible** outline with the right class (`UNKNOWN` for junk);
   - **only screws lying flat** also get the three points: `head_top` (centre of the head's top face), `bearing`
     (centre of the head's underside, on the axis) and `tip` (centre of the tip end). Keep each screw's points in
     the same **group** as its polygon (select them, press G). Points outside any group are attached to the screw
     polygon that contains them;
   - parts lying on others or cut by the border: polygon and class only.
5. **Export:** use *Export task dataset* with **CVAT for images 1.1**.

### 6. Import and match lengths

```bash
screwsynth-real import --session s01 --xml path/to/annotations.xml
screwsynth-real match --session s01
```

`match` links every labelled screw to the inventory, per image and class:
- **One length:** if the inventory has only one length for the class, every screw of that class gets it
  automatically.
- **Several lengths:** each screw with labelled keypoints is measured with tool 08's geometry. It gets the nearest
  inventory length when that is clearly closer than the next one (within 40 % of the gap) and the inventory count
  allows it.
- **Ambiguous:** everything else is listed with its candidate lengths. Write `length_mm` and set `status` to
  `manual` in `matches.csv`. Manual rows survive re-running `match`.

### 7. Build the real dataset

```bash
screwsynth-real build-dataset --name val_real                  # every labelled session -> val only
screwsynth-real build-dataset --name real_ft --sessions s05 s06 --split train   # fine-tuning set
```

1. **Scenes:** each labelled image becomes a processed-format scene in `data/real/<session>/scenes/`:
   - `image.png` (the ROI);
   - `instance.png`, drawn from the polygons, largest first;
   - `meta.json` with:
     - the calibrated camera, with its real lens distortion;
     - part ids with the matched lengths;
     - finishes from the inventory (when a class has one finish);
     - keypoints;
     - `measurable` (screws with all three keypoints and a known length).
2. **Export:** tool 07's exporter turns the scenes into the same formats: YOLO seg/pose, COCO, crops.
3. **Splits:** all photos of a session share one layout id, so they always stay in the same split.
   - A **validation** set is `val` only. Real images are never mixed into training.
   - Use **different sessions** for a fine-tuning set (`--split train`).

To attach the real set to a synthetic dataset, use tool 07's `--val-real`:
`screwsynth-labels export --run run01 --out datasets/run01 --val-real datasets/val_real/seg/images/val`.

### 8. Sim-to-real report

```bash
screwsynth-real report --dataset datasets/val_real --stage1 .../best.pt --stage2 .../best.pt \
    --synthetic-dataset datasets/dataset_v1 --run dataset_v1
```

This writes `datasets/val_real/sim_to_real/sim_to_real.md` (and `.json`):
1. **Evaluation:** tool 08 on the real set, side by side with synthetic val (mAP, size confusion, length accuracy,
   worst cases).
2. **Image statistics:** tool 06's compare (rendered vs real) with its suggestions, also as `realism.html`.
3. **Top error sources:** the lowest-mAP classes, and the length exact rate by size, family, **finish** (from the
   inventory) and session.
4. **Suggested changes:** for example "zinc_clear parts often wrong: check that finish's material", "large mAP gap:
   fix realism, then fine-tune", "N screws missed: lower measure.infer.conf".

## Tests

`pytest tests/realdata` covers:
- sessions and inventory validation;
- the printer whitelist (homing and parking only, refused before writing);
- the CVAT export/import round trip, including points without a group;
- the matching cases: unambiguous, ambiguous, over count, not in the inventory, manual rows kept;
- an end-to-end run: a session with an image, CVAT XML, the inventory, an automatic length match, then
  `build-dataset` → a val-only dataset with a measurement crop of the true length.
