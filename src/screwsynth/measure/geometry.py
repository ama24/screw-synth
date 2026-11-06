"""Length measurement geometry: axis heights, parallax and tilt correction, snapping.

The pixels of a keypoint are the image of a point on the screw axis. Its height above the plate is
known from the part's resting geometry (tool 04: the axis rests on the head and on the thread, so
it is tilted), and depends on the class (size) and slightly on the length. Each keypoint is
back-projected as a camera ray and intersected with the horizontal plane at its height, which
removes the parallax; the length is the 3D distance between the two points, which removes the
tilt. Heights depend on the length being measured, so the estimate is iterated once.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from screwsynth.calibration.geometry import Camera
from screwsynth.common.naming import KEYPOINTS

log = logging.getLogger(__name__)


@dataclass
class ClassGeometry:
    """What the measurement needs to know about one screw class (family + size)."""

    class_name: str
    family: str
    size: str
    mode: str  # bearing_to_tip | head_top_to_tip
    lengths: list[float]  # enabled standard lengths, sorted
    d: float  # nominal (thread) diameter
    head_radius: float  # radius the head rests on when lying (dk/2, or s/2 for hex heads)
    # length -> keypoint heights (mm above the plate) in the usual lying pose
    heights_by_length: dict[float, dict[str, float]] = field(default_factory=dict)
    kind: str = "screw"  # screw | standoff (the classical thread-width check is for screws)
    cut_to_length: bool = False  # threaded rod: any length, rounded to 1 mm instead of snapped

    def heights(self, length: float) -> dict[str, float]:
        """Keypoint heights for a (raw) length: interpolated between the standard lengths."""
        table = self.heights_by_length
        if not table:
            return analytic_heights(self.d, self.head_radius)
        keys = sorted(table)
        if length <= keys[0]:
            return dict(table[keys[0]])
        if length >= keys[-1]:
            return dict(table[keys[-1]])
        i = bisect_left(keys, length)
        a, b = keys[i - 1], keys[i]
        w = (length - a) / (b - a)
        return {k: (1 - w) * table[a][k] + w * table[b][k] for k in KEYPOINTS}

    @property
    def anchor(self) -> str:
        """Keypoint the nominal length is measured from (to the tip)."""
        return "head_top" if self.mode == "head_top_to_tip" else "bearing"


def analytic_heights(d: float, head_radius: float) -> dict[str, float]:
    """Fallback without rest poses: the head rests on its radius, the thread on d/2."""
    return {"head_top": head_radius, "bearing": head_radius, "tip": d / 2}


def side_heights(rest: Mapping[str, Any]) -> dict[str, float] | None:
    """Probability-weighted keypoint heights of a part's lying ("side") poses."""
    poses = [p for p in rest.get("poses", []) if p.get("category") == "side"]
    total = sum(float(p["probability"]) for p in poses)
    if not poses or total <= 0:
        return None
    return {
        k: sum(float(p["probability"]) * float(p["keypoint_heights_mm"][k]) for p in poses) / total
        for k in KEYPOINTS
    }


class Geometry:
    """Per-class screw geometry for the measurement, from the catalogue and the rest poses."""

    def __init__(self, classes: Mapping[str, ClassGeometry]) -> None:
        self.classes = dict(classes)

    def __contains__(self, class_name: str) -> bool:
        return class_name in self.classes

    def __getitem__(self, class_name: str) -> ClassGeometry:
        return self.classes[class_name]

    @classmethod
    def from_config(cls, cfg: Any) -> Geometry:
        from screwsynth.catalogue.schema import LENGTH_KINDS, catalogue_from_config
        from screwsynth.common.io import read_json
        from screwsynth.common.paths import resolve

        cat = catalogue_from_config(cfg)
        rest_dir = resolve(cfg.get("paths.rest_poses_dir", "data/rest_poses"))
        out: dict[str, ClassGeometry] = {}
        missing = 0
        for p in cat.enabled_parts:
            if p.kind not in LENGTH_KINDS or p.length is None:
                continue
            g = out.get(p.class_name)
            if g is None:
                dims = p.dims
                hex_body = p.shape == "hex" or p.kind == "standoff"
                head_r = dims["s"] / 2 if hex_body else dims.get("dk", dims["d"]) / 2
                g = out[p.class_name] = ClassGeometry(
                    class_name=p.class_name,
                    family=p.family,
                    size=p.size,
                    mode=p.nominal_length or "bearing_to_tip",
                    lengths=[],
                    d=float(dims["d"]),
                    head_radius=float(head_r),
                    kind=p.kind,
                    cut_to_length=p.cut_to_length,
                )
            g.lengths.append(float(p.length))
            path = Path(rest_dir) / f"{p.id}.json"
            h = side_heights(read_json(path)) if path.is_file() else None
            if h is None:
                missing += 1
                h = analytic_heights(g.d, g.head_radius)
            g.heights_by_length[float(p.length)] = h
        for g in out.values():
            g.lengths = sorted(set(g.lengths))
        if missing:
            log.warning("%d screw(s) without rest poses: using analytic axis heights", missing)
        return cls(out)


PARTS_FORMAT = "screwsynth-parts"  # parts.json: what a sorter needs to measure and grip each class
PARTS_VERSION = 1  # change sortering-pnp's reader (measure/geometry.py) together with this


def parts_table(cfg: Any) -> dict[str, Any]:
    """parts.json: per class with a length (screws, standoffs) the standard lengths a measurement
    snaps to, thread and head sizes (the grip point) and the axis heights of the lying poses.

    It goes into every exported dataset next to classes.yaml and from there into the model package,
    so a sorter (sortering-pnp) measures with exactly the catalogue the model was trained on."""
    from screwsynth.catalogue.schema import LENGTH_KINDS, catalogue_from_config

    cat = catalogue_from_config(cfg)
    geo = Geometry.from_config(cfg)
    first: dict[str, Any] = {}
    for p in cat.enabled_parts:
        if p.kind in LENGTH_KINDS and p.length is not None:
            first.setdefault(p.class_name, p)
    classes: dict[str, Any] = {}
    for name in sorted(geo.classes):
        g, p = geo.classes[name], first[name]
        headless = p.kind == "standoff" or p.shape in ("set_screw", "threaded_rod")
        classes[name] = {
            "family": g.family,
            "size": g.size,
            "kind": g.kind,
            "shape": p.shape,
            "mode": g.mode,
            "d": g.d,
            "head_radius": g.head_radius,
            "head_height": 0.0 if headless else float(p.dims.get("k", 0.0)),
            "headless": headless,
            "socket": p.shape in ("socket_cap", "button", "countersunk"),
            "cut_to_length": g.cut_to_length,
            "lengths": g.lengths,
            "heights": {
                f"{length:g}": {k: round(v, 4) for k, v in h.items()}
                for length, h in sorted(g.heights_by_length.items())
            },
        }
    return {
        "format": PARTS_FORMAT,
        "version": PARTS_VERSION,
        "catalogue_hash": cat.hash,
        "units": "mm",
        "classes": classes,
    }


@dataclass
class Measurement:
    raw_mm: float
    snapped_mm: float | None
    residual_mm: float | None
    confidence: float
    anchor_xyz: list[float]
    tip_xyz: list[float]
    tilt_deg: float
    iterations: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_raw_mm": round(self.raw_mm, 3),
            "length_mm": self.snapped_mm,
            "residual_mm": None if self.residual_mm is None else round(self.residual_mm, 3),
            "confidence": round(self.confidence, 3),
            "anchor_xyz_mm": [round(v, 3) for v in self.anchor_xyz],
            "tip_xyz_mm": [round(v, 3) for v in self.tip_xyz],
            "tilt_deg": round(self.tilt_deg, 2),
            "notes": list(self.notes),
        }


def axis_points(
    cam: Camera,
    anchor_px: Sequence[float],
    tip_px: Sequence[float],
    heights: Mapping[str, float],
    anchor: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project the two keypoints onto their horizontal planes (plate frame, mm)."""
    a = cam.pixel_to_plate(np.asarray([anchor_px], float), heights[anchor])[0]
    t = cam.pixel_to_plate(np.asarray([tip_px], float), heights["tip"])[0]
    return a, t


def measure_length(
    cam: Camera,
    geo: ClassGeometry,
    anchor_px: Sequence[float],
    tip_px: Sequence[float],
    iterations: int = 2,
    heights: Mapping[str, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """3D length between the anchor keypoint and the tip, parallax- and tilt-corrected.

    pixels: distorted image coordinates of `cam`. heights: known keypoint heights (then no
    iteration); otherwise they come from the class geometry at the current length estimate,
    starting from the middle standard length.
    """
    if heights is not None:
        a, t = axis_points(cam, anchor_px, tip_px, heights, geo.anchor)
        return float(np.linalg.norm(a - t)), a, t
    L = geo.lengths[len(geo.lengths) // 2] if geo.lengths else 10.0
    a = t = np.zeros(3)
    for _ in range(max(1, iterations)):
        a, t = axis_points(cam, anchor_px, tip_px, geo.heights(L), geo.anchor)
        L = float(np.linalg.norm(a - t))
    return L, a, t


def snap(length: float, lengths: Sequence[float]) -> tuple[float | None, float | None, float]:
    """Nearest standard length, the residual (raw - snapped) and half the local gap.

    Exactly between two lengths, the shorter one wins (deterministic). Lengths outside the list
    snap to the nearest end; the half gap there is the one of the nearest pair.
    """
    ls = sorted(float(x) for x in lengths)
    if not ls:
        return None, None, math.inf
    best = min(ls, key=lambda x: (abs(x - length), x))
    i = ls.index(best)
    if len(ls) == 1:
        half_gap = max(1.0, 0.1 * best)
    elif length >= best:
        half_gap = ((ls[i + 1] if i + 1 < len(ls) else best + (best - ls[i - 1])) - best) / 2
    else:
        half_gap = (best - (ls[i - 1] if i > 0 else best - (ls[i + 1] - best))) / 2
    return best, length - best, half_gap


def confidence(
    residual: float | None,
    half_gap: float,
    measurable: bool = True,
    disagreement_mm: float | None = None,
) -> tuple[float, list[str]]:
    """1 when the raw value sits on a standard length, falling to 0 at half the gap to the next
    one; scaled down when the part is not measurable or two methods disagree."""
    notes = []
    if residual is None or not math.isfinite(half_gap) or half_gap <= 0:
        return 0.0, ["no standard lengths for this class"]
    c = max(0.0, 1.0 - abs(residual) / half_gap)
    if abs(residual) > 0.6 * half_gap:
        notes.append("raw length close to the middle between two standard lengths")
    if not measurable:
        c *= 0.3
        notes.append("not measurable (covered, tilted, cut by the border or not lying flat)")
    if disagreement_mm is not None and disagreement_mm > half_gap:
        c *= 0.5
        notes.append(f"methods disagree by {disagreement_mm:.2f} mm")
    return c, notes


def measure(
    cam: Camera,
    geo: ClassGeometry,
    keypoints_px: Mapping[str, Sequence[float]],
    measurable: bool = True,
    other_raw_mm: float | None = None,
) -> Measurement:
    """Full measurement of one screw from its keypoints (distorted pixels of `cam`)."""
    raw, a, t = measure_length(cam, geo, keypoints_px[geo.anchor], keypoints_px["tip"])
    disagreement = abs(raw - other_raw_mm) if other_raw_mm is not None else None
    if geo.cut_to_length:  # any length: the nearest mm, no standard lengths to snap to
        snapped, residual = float(round(raw)), raw - round(raw)
        conf, notes = confidence(0.0, 1.0, measurable, disagreement)
        notes.append("cut to length: rounded to 1 mm")
    else:
        snapped, residual, half_gap = snap(raw, geo.lengths)
        conf, notes = confidence(residual, half_gap, measurable, disagreement)
    horiz = float(np.linalg.norm(a[:2] - t[:2]))
    tilt = math.degrees(math.atan2(float(a[2] - t[2]), horiz)) if horiz > 0 else 90.0
    return Measurement(raw, snapped, residual, conf, a.tolist(), t.tolist(), tilt, 2, notes)


def thread_diameter_mm(
    widths_px: Sequence[float], cam: Camera, centre_px: Sequence[float], height_mm: float
) -> float:
    """Shank width in mm from widths (px) measured across the axis at full resolution: the
    median width, scaled by the image scale at the axis height around centre_px."""
    if len(widths_px) == 0:
        return float("nan")
    c = np.asarray(centre_px, float)
    p = cam.pixel_to_plate(np.array([c, c + [1.0, 0.0], c + [0.0, 1.0]]), height_mm)
    mm_per_px = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2
    return float(np.median(widths_px) * mm_per_px)


def nearest_size(diameter_mm: float, sizes: Sequence[float]) -> float:
    return min(sizes, key=lambda s: abs(s - diameter_mm))


def camera_from_meta(camera: Mapping[str, Any], distorted: bool | None = None) -> Camera:
    """Camera of a rendered/processed scene (meta.json `camera`): ROI pixel coordinates."""
    K = np.asarray(camera["K"], float)
    dist = np.asarray(camera.get("dist_coeffs", np.zeros(5)), float).ravel()
    use_dist = camera.get("distorted", False) if distorted is None else distorted
    w, h = camera.get("resolution", camera.get("image_size_full"))
    return Camera(
        K=K,
        dist=dist if use_dist else np.zeros(5),
        image_size=(int(w), int(h)),
        T_cam_plate=np.asarray(camera["T_cam_plate"], float),
    )
