# 01 catalogue

The single source of truth for every part a dataset can contain. [configs/catalogue.yaml](../../configs/catalogue.yaml)
holds the families, a per-size dimension table using the standard's symbols, length lists and allowed finishes.
Code expands these into the flat part list (family × size × length).
[configs/classes.yaml](../../configs/classes.yaml) is generated from the enabled parts.

- **Inputs:** `configs/catalogue.yaml`, `catalogue.max_length_mm` in `configs/config.yaml`
- **Outputs:** `configs/classes.yaml`, CSV exports, the `Catalogue` Python API used by tools 02–08
- **Guide:** [docs/prompts/01_catalogue.txt](../../docs/prompts/01_catalogue.txt)
- **Values to check:** [VERIFY.md](VERIFY.md)

**Status: done.** The shipped catalogue has 680 parts and 105 classes (plus UNKNOWN), with a maximum
length of 80 mm.

## Contents

| Families | Kind | Sizes |
|---|---|---|
| DIN912 (DIN 912 / ISO 4762), DIN933 (DIN 933 / ISO 4017) | screw | M2–M10 |
| ISO7380, DIN7991 (DIN 7991 / ISO 10642) | screw | M3–M10 |
| DIN913 (set screw, ISO 4026), DIN975 (threaded rod) | screw | M2–M10 |
| STANDOFF_MF, STANDOFF_FF, STANDOFF_MM (hex standoffs) | standoff | M2–M6 |
| BRACKET (corner bracket for extrusion) | bracket | 2020, 3030, 4040 |
| DIN934 (hex nut) | nut | M2–M10 |
| DIN985 (nyloc) | nut | M3–M10 |
| DIN125A (DIN 125 A / ISO 7089) | washer | M2–M10 |
| DIN9021 (large washer) | washer | M3–M10 |
| TNUT_SLIDE, TNUT_DROP | tnut | slot 6: M3/M4/M5, slot 8: M4/M5/M6 |

Finishes are `black_oxide`, `zinc_clear`, `zinc_yellow`, `stainless_a2`, `stainless_a4`, `plain_steel`, `brass`,
`nylon_white`, `nylon_black`, `aluminium` and `aluminium_black` (anodised). They are visual only and never a class.

**DIN 913 vs DIN 975.** Lying on their side, a set screw and a piece of threaded rod look alike from above (the
socket is hidden), so their lengths never overlap. Set screws go up to about 3 × d (the next standard length). Rod
pieces start at twice the longest set screw of the same size: M3 3–10 / 20–80, M5 5–16 / 35–80,
M10 10–30 / 60–80. A test checks the gap.

## File format

```yaml
version: 1
finishes:
  black_oxide: {label: Black oxide, material: metal}       # material: metal | plastic
families:
  DIN912:                        # family name: upper-case letters/digits (T-nuts: TNUT_<...>)
    title: Hexagon socket head cap screw
    standard: DIN 912 / ISO 4762
    kind: screw                  # screw | standoff | nut | washer | tnut | bracket
    shape: socket_cap            # which builder tool 02 uses (see "Shapes")
    nominal_length: bearing_to_tip   # screws only: bearing_to_tip | head_top_to_tip
    enabled: true
    verified: false              # set true once checked with calipers (also per size)
    dims_source: ISO 4762:2004 table 1
    common_dims: {angle: 90}     # optional: merged into every size's dims
    finishes:                    # finish -> "all" or a list of size keys
      black_oxide: all
      nylon_white: [M2.5, M3, M4, M5, M6]
    sizes:
      M4:                        # size key; T-nuts use S<slot>_<thread>, e.g. S6_M4
        pitch: 0.7               # coarse pitch (threaded parts)
        dims: {dk: 7.0, k: 4.0, s: 3.0, t: 2.0, b: 20, full_thread_max_length: 25}
        lengths: [6, 8, 10, 12, 16, 20, 25, 30, 35, 40]   # screws only
        uncommon: [35]           # optional: kept, but marked common: false
        # optional per size: enabled: false, common: false, verified: true
rules:                           # ordered overrides; the last matching rule wins
  - {match: DIN912_M2, enabled: false}
```

Expansion rules:
- Screws produce one part per length, up to `catalogue.max_length_mm` (default 80, set in `configs/config.yaml`).
  Longer lengths stay in the file but are not expanded. Raise the limit to use them. Every command also takes
  `--max-length MM` to override it once.
- `d` (nominal diameter) is filled in from the size. For `socket_cap`, `ds` defaults to `d`.
- A part is enabled when its family and size are enabled. Then each rule is applied in order. A rule's `match`
  is a shell glob tested against the part id, class name, family and size. For example, `DIN985` disables the
  whole family, `M2` every M2 part, and `DIN912_M3x*` all M3 DIN 912 lengths.
- `common` is false when the size has `common: false` or the length is listed in `uncommon`.
- `fully_threaded` (screws) is true when the family has no `b`, or when length ≤ `full_thread_max_length`.

### Shapes and required dimensions (mm)

| shape | dims |
|---|---|
| `socket_cap` (DIN 912) | d, dk, k, s, t, b, ds, optional `full_thread_max_length` |
| `hex` (DIN 933) | d, s, k, e, optional dw, c |
| `button` (ISO 7380) | d, dk, k, s, t |
| `countersunk` (DIN 7991) | d, dk, k, s, t, angle, optional b, `full_thread_max_length` |
| `hex_nut` (DIN 934) | d, s, m, e |
| `nyloc_nut` (DIN 985) | d, s, h (total), m (metal part), e |
| `washer` | d1 (inner), d2 (outer), h |
| `tnut_slide`, `tnut_drop` | d, length (along slot), width (across slot), height, neck_width, neck_height |
| `set_screw` (DIN 913) | d, s (socket across flats), t (socket depth) |
| `threaded_rod` (DIN 975) | d |
| `standoff_mf`, `standoff_ff`, `standoff_mm` | d, s (across flats), e (across corners), ls (male stud), lf (female thread depth); MF needs both, FF lf, MM ls |
| `corner_bracket` | a (leg), w (width), t (leg thickness), ts (side wall), dh (hole), slot (hole elongation), key_w, key_h (alignment key) |

### STEP-based parts
A family can set `source: step` instead of parametric dimensions. For screws, each size's `step` entry is keyed
by length, and each entry needs a `path` and `keypoints` (`head_top`, `bearing`, `tip` as `[x, y, z]`, measured in
the STEP file's own coordinates; optional `units`, default mm). Tool 02 aligns the part from the keypoints. Other kinds give a single `step: {path: ...}`. For now these fields are only stored and
validated here; tool 02 converts the files. See [templates/family_template.yaml](templates/family_template.yaml).

### Unknown parts (`unknown:`)
Junk that ends up on the plate but is not a catalogue part all gets one class, `UNKNOWN`, so the
machine can reject it.
- **Types:** `wood_screw`, `split_washer`, `tooth_washer`, `pin`, `bracket` (bent sheet metal),
  `plate` and `blob`. `threaded_rod` and `standoff` still exist but are left out of the shipped
  catalogue, because DIN975 and the STANDOFF families are classes now.
- **Variants:** each type gets `variants` shapes (default 6), with ids `UNKNOWN_WOODSCREW_01`, …
- **Parameters:** each variant's parameters are drawn from the type's ranges (mm), seeded by the part
  id, so an id always gives the same part. The defaults are in `UNKNOWN_TYPES` in `schema.py`.
- **Finishes:** set per section (`finishes`) or per type.
- **Switching off:** use `enabled: false` (the whole section, or one type) or a rule:
  `{match: UNKNOWN}` for all, `{match: "UNKNOWN_BLOB_*"}` for one type.

The class `UNKNOWN` sorts last, so adding it does not change the ids of the real classes.

## CLI: `screwsynth-catalogue`

Every command accepts `--config`, `--catalogue`, `--max-length MM` and `-v`.

| Command | What it does |
|---|---|
| `list [--family F] [--size M4] [--enabled-only]` | Table of parts with class, length, enabled, common and finishes |
| `stats` | Counts per family, per class and per finish |
| `validate` | Runs every check. Exit code 1 on errors. Warns if `classes.yaml` is stale |
| `enable <pattern>` / `disable <pattern>` | Adds a rule (replacing one with the same pattern) |
| `add-length <family> <size> <length> [--uncommon]` | Inserts a length in sorted order |
| `add-family --from-template FILE [--replace]` | Copies families from a template file |
| `export-csv PATH [--family F] [--size M4] [--enabled-only]` | One row per part, dims as `dim_<symbol>` columns |
| `classes [--out PATH]` | Regenerates `configs/classes.yaml` |

Examples:

```bash
screwsynth-catalogue stats
screwsynth-catalogue list --family DIN912 --size M3
screwsynth-catalogue add-length DIN912 M3 35 --uncommon
screwsynth-catalogue disable DIN933_M2          # one class
screwsynth-catalogue disable "DIN912_M10x*"     # glob (quote it in the shell)
screwsynth-catalogue enable DIN933_M2
screwsynth-catalogue add-family --from-template tools/catalogue/templates/family_template.yaml
screwsynth-catalogue export-csv output/parts.csv --enabled-only
screwsynth-catalogue validate && screwsynth-catalogue classes
```

Editing commands use ruamel.yaml, so comments and layout are preserved. They validate the result and **refuse to
save** if it would no longer validate. After any change, run `classes` so that `configs/classes.yaml` matches.

## Validation
Structural problems stop loading with a `CatalogueError`: unknown keys, wrong types, bad size keys, unknown
kind or shape, bad family names, or a missing `tnut_kind`. Semantic checks report issues with a code:

- **Ids and classes:** `duplicate-id`, `part-id`, `class-mismatch`.
- **Dimensions:** `missing-dim`, `dim-positive`, `d-mismatch`, `dim-sanity`. Examples of `dim-sanity`: dk > d,
  s < dk, t < k, e in (s, s/cos30], d2 > d1, countersunk k ≥ cone height, nyloc h > m, T-nut neck
  fits the thread.
- **Threads and lengths:** `pitch`, `no-lengths`, `length`, `uncommon`, `thread-length` (b ≤ length for
  partially threaded screws).
- **Finishes:** `unknown-finish`, `finish-size`, `no-finishes`, `finish-material`.
- **Family settings:** `shape-kind`, `nominal-length`, `no-sizes`, `step-source`.
- **Warnings:** `rule-unmatched`, `no-classes`.

## Python API

```python
from screwsynth.catalogue import load_catalogue

cat = load_catalogue("configs/catalogue.yaml", max_length_mm=80)
cat.enabled_parts  # list[Part]: id, class_name, dims, finishes, length, pitch, ...
cat.classes()  # ordered class names (same order as classes.yaml)
cat.part("DIN912_M4x16").hash  # geometry hash, used by tool 02 for incremental builds
```

`schema.py` is pure Python. Tool 05 can import it inside Blender, but loading the YAML there needs PyYAML.
