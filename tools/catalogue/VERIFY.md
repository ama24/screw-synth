# Catalogue values to verify

Every entry in `configs/catalogue.yaml` has `verified: false`. All dimensions were filled in from
memory of the published DIN/ISO tables, not copied from the documents. The values below are the
ones I am **least** sure of. Check them first, with calipers or against the standard or your
supplier's datasheet.

After checking a size, correct the values and set `verified: true` on that size entry. Setting it on
the family means the whole family is checked.

Use `screwsynth-catalogue export-csv parts.csv` to get a sheet you can compare against supplier
catalogues.

## How the values were chosen
- Head dimensions use the standard's **maximum or nominal** value (dk max, k max, s nominal).
  Socket depth `t` uses the **minimum**. Real parts are usually slightly smaller than dk/k max.
  Caliper readings of your own stock are the best reference for rendering.
- Across-corners `e` is the standard's **minimum**. The theoretical value is s / cos 30°.
- Lengths are the standard's preferred lengths within each size's range, cut at
  `catalogue.max_length_mm`. The ranges are the least certain part (see below).

## Uncertain values

| Part / family | Field | Why it is uncertain |
|---|---|---|
| DIN 933 M10, DIN 934 M10, DIN 985 M10 | `s` = 16, `e` = 17.77 | ISO 4017/4032/10511 use 16 mm across flats. The old DIN 933/934/985 use **17 mm** (e ≈ 18.9). Parts sold as "DIN" in Europe can be either. Measure yours. |
| DIN 912 (all sizes) | `full_thread_max_length` | Longest length that is threaded up to the head. Taken as roughly the shaded area of ISO 4762 table 1. Values vary by edition and supplier. |
| DIN 912 (all sizes) | `b` | Reference thread length for l ≤ 125 mm (2d + 12). The standard defines it this way, but suppliers' partially threaded lengths vary. |
| DIN 912 (all sizes) | `lengths` | ISO 4762 ranges were used. Old DIN 912 and trade ranges are often wider, e.g. M3x35–50, M2.5x30. Add them with `add-length` if you have them. |
| DIN 933 M2, M2.5 | all dims, `lengths` | Rare sizes. dw/c come from ISO 4017 small-size rows I am not sure of. Marked `common: false`. |
| DIN 933 (all sizes) | `dw`, `c` | Washer-face diameter (min) and height (max) from ISO 4017. Many DIN 933 parts have no visible washer face. |
| DIN 933 (all sizes) | `lengths` | Start of each range (e.g. M3 from 5, M6 from 8). ISO 4017 starts slightly longer than trade DIN 933. |
| ISO 7380 (all sizes) | `lengths` | Trade ranges were used (suppliers sell far more lengths than ISO 7380-1 lists). Long lengths are marked uncommon. |
| ISO 7380 (all sizes) | `t` | Socket depth minimum. Recalled values, low confidence. |
| ISO 7380 | sizes M2/M2.5 | Not included because ISO 7380-1 starts at M3. Some vendors sell M2/M2.5 button heads; add them as sizes if you have them. |
| DIN 7991 (all sizes) | `dk`, `k` | DIN 7991 values used: dk 6/8/10/12/16/20, k 1.7–5.5. ISO 10642 has larger *theoretical* dk (6.72/8.96/11.2/13.44/17.92/22.4) and k (1.86–6.2). The real head is between the two. Measure it. |
| DIN 7991 (all sizes) | `t`, `b`, `full_thread_max_length` | Recalled values, low confidence. |
| DIN 7991 | sizes M2/M2.5 | Not included because DIN 7991 starts at M3. Vendor versions exist. |
| DIN 934 M5, M6, M8, M10 | `m` | ISO 4032 heights were used (4.7/5.2/6.8/8.4). Old DIN 934 heights are lower (4/5/6.5/8). |
| DIN 985 (all sizes) | `m` (metal part height) | Estimated, not from the standard. It only sets where the nylon insert starts in the mesh. |
| DIN 985 (all sizes) | `h` | Total height, maximum. Recalled values. |
| DIN 985 | sizes M2/M2.5 | Not included (DIN 985 is usually listed from M3). |
| DIN 125 A M2 | `h` = 0.3 | Recalled value. Some tables give 0.3, others 0.35. |
| DIN 9021 (all sizes) | all dims | Recalled from ISO 7093-1. M2/M2.5 are not included. |
| TNUT_SLIDE (all) | all dims | **Not standardised.** Vendor-typical values (`dims_source: vendor-typical, measure your own`). 2020 profiles: 10x10 mm body, 4 mm high. 3030/4040: 12x15 mm, 6 mm high. Measure your own. |
| TNUT_DROP (all) | all dims | **Not standardised.** Hammer-head shape: 11x6 mm (slot 6), 16x8 mm (slot 8). Roll-in nuts with a spring ball look different, so add a separate family if you use them. |
| T-nuts (all) | `pitch` | Coarse pitch was assumed. Correct as long as the thread is standard metric coarse. |
| DIN 913 (all sizes) | `s`, `t` | ISO 4026 socket sizes (s nominal) and depths (t min, for the longer lengths). Recalled values; short set screws have shallower sockets. |
| DIN 913 (all sizes) | `lengths` | Deliberately cut at about 3 × d so they never overlap with DIN 975 (see README). Longer set screws exist; adding them breaks that gap. |
| DIN 975 (all sizes) | `min_length` | Cut to any length: only the shortest piece is set (twice the longest DIN 913 of the size); the rest comes from `length_step` and `catalogue.rod_max_length_mm`. |
| STANDOFF_* (all) | all dims | **Not standardised.** Vendor-typical: across flats 4/5/5.5/7/8/10 for M2–M6, male stud about 1–1.7 × d, female thread depth about 1.5 × d. Brass M3 standoffs come with s = 5 or 5.5. Measure yours. |
| STANDOFF_FF (all) | `through` | Threaded all the way through, like the user's own FF standoffs (2026-11-01). Hex coupling nuts (DIN 6334) look the same from above and are sorted into this class. |
| STANDOFF_* (all) | `lengths` | Common vendor lengths (body only, studs excluded). |
| BRACKET (all) | all dims | **Not standardised.** Die-cast bracket shape guessed from typical 2020/3030/4040 parts: legs 20/28/36 mm, slotted hole, triangular side walls, alignment key. Measure yours. |
| All screws | `common` / `uncommon` | My guess at what is uncommon in the trade (mostly lengths 45, 55 and 65, and very long button heads). |
| All families | `finishes` | Typical availability, not a hard rule. Example: nylon is listed only for small DIN 912/933/934/125 sizes. |

## Coarse pitch (high confidence)
M2 0.4, M2.5 0.45, M3 0.5, M4 0.7, M5 0.8, M6 1.0, M8 1.25, M10 1.5 (ISO 261 coarse series).
