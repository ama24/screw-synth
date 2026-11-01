"""Stable resting poses of a part on a flat plate.

Method: trimesh.poses.compute_stable_poses on the render mesh (convex hull + centre of mass, exact
solid-angle probabilities when com_samples == 1). Poses that only differ by a yaw (rotation about the
plate normal) or by a symmetry of the part (roll about its own axis, 180 deg flip of symmetric
nuts/washers) are merged; rare poses are dropped; per-family overrides reweight categories.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from screwsynth.catalogue.schema import LENGTH_KINDS
from screwsynth.common.naming import KEYPOINTS

GENERATOR_VERSION = "1"

# roll symmetry about the part's z axis ("continuous" or n-fold) and 180 deg flip symmetry
SYMMETRY: dict[str, tuple[str | int, bool]] = {
    "socket_cap": ("continuous", False),
    "button": ("continuous", False),
    "countersunk": ("continuous", False),
    "hex": (6, False),
    "set_screw": ("continuous", False),
    "threaded_rod": ("continuous", True),
    "standoff_mf": (6, False),
    "standoff_ff": (6, True),
    "standoff_mm": (6, True),
    "hex_nut": (6, True),
    "nyloc_nut": (6, False),
    "washer": ("continuous", True),
    "tnut_slide": (2, False),
    "tnut_drop": (2, False),
    # unknown (junk) parts: round ones merge their lying poses like the screws do
    "unknown_wood_screw": ("continuous", False),
    "unknown_threaded_rod": ("continuous", True),
    "unknown_pin": ("continuous", True),
    "unknown_standoff": (6, True),
    "unknown_tooth_washer": ("continuous", True),
}

CATEGORIES_SCREW = ("side", "standing_on_head", "standing_on_tip")
CATEGORIES_OTHER = ("flat", "edge")


@dataclass(frozen=True)
class Settings:
    com_samples: int = 1
    com_sigma_mm: float = 0.0
    min_probability: float = 0.02
    measurable_tilt_deg: float = 15.0
    merge_tilt_deg: float = 1.0
    overrides: tuple[Mapping[str, Any], ...] = ()
    seed: int = 0

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None, seed: int = 0) -> Settings:
        s = dict(section or {})
        return cls(
            com_samples=int(s.get("com_samples", 1)),
            com_sigma_mm=float(s.get("com_sigma_mm", 0.0)),
            min_probability=float(s.get("min_probability", 0.02)),
            measurable_tilt_deg=float(s.get("measurable_tilt_deg", 15.0)),
            merge_tilt_deg=float(s.get("merge_tilt_deg", 1.0)),
            overrides=tuple(dict(o) for o in (s.get("overrides") or [])),
            seed=seed,
        )


@dataclass
class Pose:
    transform: np.ndarray  # 4x4, part local frame -> plate frame (mm), lowest point at z = 0
    probability: float
    raw_probability: float
    category: str
    axis_tilt_deg: float  # elevation of the part's +z axis (towards the head) above the plate
    keypoint_heights: dict[str, float] = field(default_factory=dict)
    measurable: bool = False
    merged: int = 1

    def to_json(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "transform": np.round(self.transform, 9).tolist(),
            "probability": round(self.probability, 6),
            "raw_probability": round(self.raw_probability, 6),
            "category": self.category,
            "axis_tilt_deg": round(self.axis_tilt_deg, 4),
            "keypoint_heights_mm": {k: round(v, 6) for k, v in self.keypoint_heights.items()},
            "axis_height_head_mm": round(self.keypoint_heights.get("bearing", 0.0), 6),
            "axis_height_tip_mm": round(self.keypoint_heights.get("tip", 0.0), 6),
            "measurable": self.measurable,
            "merged_raw_poses": self.merged,
        }


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


_RX180 = np.diag([1.0, -1.0, -1.0])


def symmetry_group(shape: str) -> tuple[str | int, bool, list[np.ndarray]]:
    order, flip = SYMMETRY.get(shape, (1, False))
    n = 1 if order == "continuous" else int(order)
    group = [_rz(2 * math.pi * k / n) for k in range(n)]
    if flip:
        group += [g @ _RX180 for g in group]
    return order, flip, group


def axis_tilt_deg(R: np.ndarray) -> float:
    return math.degrees(math.asin(float(np.clip(R[2, 2], -1.0, 1.0))))


def equivalent(R1: np.ndarray, R2: np.ndarray, shape: str, tilt_tol_deg: float) -> bool:
    """True when two resting rotations differ only by yaw and a symmetry of the part."""
    order, flip, group = symmetry_group(shape)
    if order == "continuous":
        t1, t2 = axis_tilt_deg(R1), axis_tilt_deg(R2)
        if flip:
            t1, t2 = abs(t1), abs(t2)
        return abs(t1 - t2) <= tilt_tol_deg
    cos_tol = math.cos(math.radians(tilt_tol_deg))
    return any((R1 @ g.T @ R2.T)[2, 2] >= cos_tol for g in group)


def canonical_yaw(T: np.ndarray) -> np.ndarray:
    """Rotate about the plate normal so the part axis (or part x if vertical) points along +x."""
    R = T[:3, :3]
    v = R[:, 2] if math.hypot(R[0, 2], R[1, 2]) > 1e-6 else R[:, 0]
    out = np.eye(4)
    out[:3, :3] = _rz(-math.atan2(v[1], v[0]))
    return out @ T


def category(kind: str, tilt_deg: float) -> str:
    if kind in LENGTH_KINDS:
        if tilt_deg < -60:
            return "standing_on_head"
        if tilt_deg > 60:
            return "standing_on_tip"
        return "side"
    return "flat" if abs(tilt_deg) > 60 else "edge"


def _override_weight(
    overrides: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], cat: str
) -> float:
    w = 1.0
    names = (meta["id"], meta["class_name"], meta["family"])
    for o in overrides:
        if o.get("category") not in (None, cat):
            continue
        if any(fnmatch.fnmatchcase(n, str(o.get("match", "*"))) for n in names):
            w = float(o.get("weight", 1.0))  # the last matching override wins
    return w


def raw_stable_poses(mesh: trimesh.Trimesh, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    T, p = trimesh.poses.compute_stable_poses(
        mesh,
        sigma=settings.com_sigma_mm,
        n_samples=max(1, settings.com_samples),
        seed=settings.seed,
    )
    return np.asarray(T), np.asarray(p, dtype=float)


def compute_poses(mesh: trimesh.Trimesh, meta: Mapping[str, Any], settings: Settings) -> list[Pose]:
    """Merged, thresholded, reweighted stable poses of an asset (meta = its part.json)."""
    shape, kind = meta["shape"], meta["kind"]
    T_raw, p_raw = raw_stable_poses(mesh, settings)
    order = np.argsort(-p_raw)
    clusters: list[list[int]] = []
    for i in order:
        for cl in clusters:
            if equivalent(T_raw[cl[0]][:3, :3], T_raw[i][:3, :3], shape, settings.merge_tilt_deg):
                cl.append(int(i))
                break
        else:
            clusters.append([int(i)])

    kp = {k: np.asarray(meta["keypoints"][k]["xyz"], dtype=float) for k in KEYPOINTS}
    poses = []
    for cl in clusters:
        T = canonical_yaw(T_raw[cl[0]])
        verts = mesh.vertices @ T[:3, :3].T + T[:3, 3]
        T[2, 3] -= verts[:, 2].min()  # lowest point exactly on the plate
        tilt = axis_tilt_deg(T[:3, :3])
        cat = category(kind, tilt)
        heights = {k: float((T[:3, :3] @ v + T[:3, 3])[2]) for k, v in kp.items()}
        prob = float(p_raw[cl].sum())
        poses.append(
            Pose(
                transform=T,
                probability=prob,
                raw_probability=prob,
                category=cat,
                axis_tilt_deg=tilt,
                keypoint_heights=heights,
                measurable=kind in LENGTH_KINDS and abs(tilt) <= settings.measurable_tilt_deg,
                merged=len(cl),
            )
        )

    total = sum(p.raw_probability for p in poses) or 1.0
    for p in poses:
        p.raw_probability /= total
        p.probability = p.raw_probability * _override_weight(settings.overrides, meta, p.category)
    poses = [p for p in poses if p.probability >= settings.min_probability]
    norm = sum(p.probability for p in poses)
    if norm <= 0:
        raise ValueError(f"{meta['id']}: overrides removed every pose")
    for p in poses:
        p.probability /= norm
    poses.sort(key=lambda p: -p.probability)
    return poses


def rest_pose_json(
    meta: Mapping[str, Any], poses: list[Pose], settings: Settings
) -> dict[str, Any]:
    order, flip, _ = symmetry_group(meta["shape"])
    return {
        "part_id": meta["id"],
        "class_name": meta["class_name"],
        "family": meta["family"],
        "kind": meta["kind"],
        "shape": meta["shape"],
        "units": "mm",
        "frame": "transform maps the part local frame to the plate frame (z = 0 plate, yaw = 0)",
        "roll_symmetry": order,
        "flip_symmetric": flip,
        "poses": [p.to_json(i) for i, p in enumerate(poses)],
        "method": {
            "algorithm": "trimesh.poses.compute_stable_poses (convex hull, quasi-static)",
            "com_samples": settings.com_samples,
            "com_sigma_mm": settings.com_sigma_mm,
            "min_probability": settings.min_probability,
            "merge_tilt_deg": settings.merge_tilt_deg,
            "measurable_tilt_deg": settings.measurable_tilt_deg,
            "overrides": [dict(o) for o in settings.overrides],
        },
        "catalogue_hash": meta.get("catalogue_hash"),
        "asset_generator_version": meta.get("generator_version"),
        "generator_version": GENERATOR_VERSION,
    }
