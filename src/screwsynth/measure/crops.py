"""Where the stage-2 crop around an object goes (shared by the dataset export and inference).

The keypoint model only works when inference cuts its crops the way the training crops were cut:
- fixed:  a crop_size x crop_size window at full resolution (objects longer than the window are
          cut off: at 11.7 px/mm, 384 px is about 33 mm);
- scaled: the window grows to crop_margin x the object's longer box side when that is bigger, and
          is resized down to crop_size, so long screws, rods and standoffs fit whole.
"""

from __future__ import annotations

from collections.abc import Sequence

CROP_MODES = ("fixed", "scaled")


def crop_window(
    box: Sequence[float],
    image_size: tuple[int, int],
    crop_size: int,
    mode: str = "scaled",
    margin: float = 1.2,
) -> tuple[int, int, int, float]:
    """(x0, y0, side, scale) of the square window around box (x0, y0, x1, y1) inside image_size
    (w, h); scale = crop_size / side maps window pixels to crop pixels."""
    if mode not in CROP_MODES:
        raise ValueError(f"crop mode {mode!r} is not one of {CROP_MODES}")
    w, h = image_size
    bx0, by0, bx1, by1 = box
    side = crop_size
    if mode == "scaled":
        side = int(max(crop_size, margin * max(bx1 - bx0, by1 - by0)))
    side = min(side, w, h)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0 = int(min(max(round(cx - side / 2), 0), w - side))
    y0 = int(min(max(round(cy - side / 2), 0), h - side))
    return x0, y0, side, crop_size / side
