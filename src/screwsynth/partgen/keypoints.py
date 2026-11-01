"""Keypoints (head_top, bearing, tip) in the part's local frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from screwsynth.catalogue.schema import LENGTH_KINDS, Part
from screwsynth.common.naming import KEYPOINTS


def nominal_length(part: Part, keypoints: Mapping[str, Sequence[float]]) -> float | None:
    """Nominal length from keypoints: bearing-to-tip, or head_top-to-tip (countersunk and headless
    screws, standoffs)."""
    if part.kind not in LENGTH_KINDS:
        return None
    start = "head_top" if part.nominal_length == "head_top_to_tip" else "bearing"
    return float(np.linalg.norm(np.subtract(keypoints[start], keypoints["tip"])))


def keypoints_json(keypoints: Mapping[str, Sequence[float]], labelled: bool) -> dict:
    """Ordered keypoint dict for part.json; non-screws are marked labelled: false."""
    return {
        name: {"xyz": [round(float(v), 6) for v in keypoints[name]], "labelled": labelled}
        for name in KEYPOINTS
    }
