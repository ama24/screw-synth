"""Ground truth per object: projected keypoints with visibility, silhouette areas, discard rules.

Pure numpy (usable inside Blender). Plate-frame millimetres throughout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .camera import RenderCamera
from .placement import convex_hull_2d
from .variation import view_angle_deg

KEYPOINTS = ("head_top", "bearing", "tip")
NOT_LABELLED, OCCLUDED, VISIBLE = 0, 1, 2


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def _fill_convex(poly: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Boolean mask (h, w) of pixel centres (x0 + i, y0 + j) inside a CCW convex polygon."""
    xs = np.arange(x0, x0 + w, dtype=float)
    ys = np.arange(y0, y0 + h, dtype=float)
    X, Y = np.meshgrid(xs, ys)
    inside = np.ones((h, w), dtype=bool)
    n = len(poly)
    for k in range(n):
        ax, ay = poly[k]
        bx, by = poly[(k + 1) % n]
        inside &= (bx - ax) * (Y - ay) - (by - ay) * (X - ax) >= 0
    return inside


def silhouette_mask(
    cam: RenderCamera, pieces_plate: Sequence[np.ndarray]
) -> tuple[np.ndarray, int, int] | None:
    """The object rendered alone (union of projected convex collision pieces) as a boolean mask
    and the pixel position (x0, y0) of its top-left corner; None if nothing projects."""
    polys = [convex_hull_2d(cam.project(p)) for p in pieces_plate if len(p) >= 3]
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return None
    allp = np.vstack(polys)
    x0, y0 = np.floor(allp.min(axis=0)).astype(int)
    x1, y1 = np.ceil(allp.max(axis=0)).astype(int)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for p in polys:
        px0, py0 = np.floor(p.min(axis=0)).astype(int)
        px1, py1 = np.ceil(p.max(axis=0)).astype(int)
        sub = _fill_convex(p, px0, py0, px1 - px0 + 1, py1 - py0 + 1)
        mask[py0 - y0 : py1 - y0 + 1, px0 - x0 : px1 - x0 + 1] |= sub
    return mask, int(x0), int(y0)


def silhouette_area(cam: RenderCamera, pieces_plate: Sequence[np.ndarray]) -> tuple[int, int]:
    """Pixels covered by the object rendered alone: (total ignoring the image border, inside it)."""
    sil = silhouette_mask(cam, pieces_plate)
    if sil is None:
        return 0, 0
    mask, x0, y0 = sil
    rw, rh = cam.resolution
    ys, xs = np.nonzero(mask)
    xs, ys = xs + x0, ys + y0
    in_img = int(((xs >= 0) & (xs < rw) & (ys >= 0) & (ys < rh)).sum())
    return int(mask.sum()), in_img


def occluders(
    silhouette: tuple[np.ndarray, int, int] | None,
    instance: np.ndarray,
    inst_id: int,
    min_px: int,
) -> dict[int, int]:
    """Instances seen inside this object's alone-silhouette: {id: pixels}, those with >= min_px.

    Seen from above, a part showing where this one would be lies on top of it (or leans over it).
    min_px ignores the thin slivers left by the convex collision pieces around touching parts."""
    if silhouette is None:
        return {}
    mask, x0, y0 = silhouette
    h, w = instance.shape
    ys, xs = np.nonzero(mask)
    xs, ys = xs + x0, ys + y0
    keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    ids, counts = np.unique(instance[ys[keep], xs[keep]], return_counts=True)
    return {
        int(i): int(c)
        for i, c in zip(ids, counts, strict=True)
        if i not in (0, inst_id) and c >= min_px
    }


def keypoint_visibility(
    uv: np.ndarray, instance: np.ndarray, inst_id: int, window: int = 2
) -> list[int]:
    """COCO-style visibility from the instance mask: 2 visible, 1 occluded, 0 outside the image."""
    h, w = instance.shape
    out = []
    for u, v in uv:
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            out.append(NOT_LABELLED)
            continue
        patch = instance[max(0, y - window) : y + window + 1, max(0, x - window) : x + window + 1]
        centre = int(instance[y, x])
        if centre == inst_id or (centre == 0 and (patch == inst_id).any()):
            out.append(VISIBLE)
        elif centre != 0 and centre != inst_id:
            out.append(OCCLUDED)  # another part covers the keypoint
        else:
            out.append(OCCLUDED if not (patch == inst_id).any() else VISIBLE)
    return out


def axis_tilt_deg(T: np.ndarray) -> float:
    return math.degrees(math.asin(float(np.clip(T[2, 2], -1.0, 1.0))))


def discard_reason(
    T_final: np.ndarray,
    T_initial: np.ndarray,
    verts_local: np.ndarray,
    area: Sequence[float],
    max_lift_mm: float = 0.3,
    max_move_mm: float = 15.0,
    below_plate_mm: float = 1.0,
    allow_stacked: bool = False,
    max_stack_mm: float = 25.0,
) -> str | None:
    """Why an object must be discarded after the physics settle (None = keep).

    allow_stacked: parts resting on other parts are kept (clumped scenes); otherwise a part whose
    lowest point is more than max_lift_mm above the plate is discarded as stacked."""
    v = transform_points(T_final, verts_local)
    zmin = float(v[:, 2].min())
    if zmin < -below_plate_mm:
        return f"fell through/off the plate (lowest z {zmin:.2f} mm)"
    x0, y0, x1, y1 = area
    c = v.mean(axis=0)
    if not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
        return "left the working area"
    if zmin > max_lift_mm and not allow_stacked:
        return f"stacked or floating (lowest z {zmin:.2f} mm)"
    if zmin > max_stack_mm:
        return f"balanced high up on other parts (lowest z {zmin:.2f} mm)"
    move = float(np.linalg.norm(T_final[:2, 3] - T_initial[:2, 3]))
    if move > max_move_mm:
        return f"moved {move:.1f} mm during the settle"
    return None


def object_meta(
    *,
    inst_id: int,
    part: Mapping[str, Any],
    class_id: int,
    finish: str,
    material: Mapping[str, Any],
    T_plate: np.ndarray,
    rest_pose_index: int,
    measurable_tilt_deg: float,
    cam: RenderCamera,
    instance: np.ndarray,
    alone_area: tuple[int, int],
    window: int = 2,
    silhouette: tuple[np.ndarray, int, int] | None = None,
    occlusion_min_px: int = 0,
    on_plate_mm: float = 0.3,
    lowest_z_mm: float | None = None,
    lying_on_it: Sequence[int] | None = None,
    rig_view: bool = True,
) -> dict[str, Any]:
    """Ground-truth record of one object (all lengths in mm, pixels in the undistorted render).

    support: "plate" when the lowest point is within on_plate_mm of the plate, else "parts" (it
    rests on other parts). lying_on_it: parts physically resting on this one (from the 3D poses);
    free: nothing rests on it (a candidate for the picker). occluded_by: parts covering some of it
    in the image (also neighbours leaning over it in perspective). measurable: a screw lying on
    the plate, nearly flat, free, inside the image and with all three keypoints visible, so its
    length can be measured exactly; only in the rig camera's view (rig_view), since the
    measurement relies on the calibrated camera looking straight down. view_angle_deg: angle between
    the camera ray to the part and the vertical (0 = seen from straight above).
    """
    kps_local = {k: np.asarray(part["keypoints"][k]["xyz"], dtype=float) for k in KEYPOINTS}
    labelled = bool(part["keypoints"]["tip"]["labelled"])
    kp_plate = {k: transform_points(T_plate, v)[0] for k, v in kps_local.items()}
    uv = cam.project(np.array([kp_plate[k] for k in KEYPOINTS]))
    vis = keypoint_visibility(uv, instance, inst_id, window) if labelled else [NOT_LABELLED] * 3
    mask_area = int((instance == inst_id).sum())
    tilt = axis_tilt_deg(T_plate)
    is_screw = part["kind"] in ("screw", "standoff")  # parts with a measured length
    over = occluders(silhouette, instance, inst_id, occlusion_min_px)
    on_it = sorted(int(i) for i in lying_on_it) if lying_on_it is not None else sorted(over)
    zmin = 0.0 if lowest_z_mm is None else float(lowest_z_mm)
    on_plate = zmin <= on_plate_mm
    truncated = alone_area[1] < alone_area[0]
    return {
        "instance_id": inst_id,
        "part_id": part["id"],
        "class_name": part["class_name"],
        "class_id": class_id,
        "kind": part["kind"],
        "family": part["family"],
        "size": part["size"],
        "length": part.get("length"),
        "finish": finish,
        "material": dict(material),
        "pose": np.round(T_plate, 9).tolist(),
        "rest_pose": rest_pose_index,
        "axis_tilt_deg": round(tilt, 4),
        "keypoints": {
            k: {
                "xyz": np.round(kp_plate[k], 6).tolist(),
                "uv": np.round(uv[i], 4).tolist(),
                "visibility": vis[i],
            }
            for i, k in enumerate(KEYPOINTS)
        },
        "keypoints_labelled": labelled,
        "axis_height_head_mm": round(float(kp_plate["bearing"][2]), 6) if is_screw else None,
        "axis_height_tip_mm": round(float(kp_plate["tip"][2]), 6) if is_screw else None,
        "axis_height_head_top_mm": round(float(kp_plate["head_top"][2]), 6) if is_screw else None,
        "measurable": bool(
            rig_view
            and is_screw
            and abs(tilt) <= measurable_tilt_deg
            and on_plate
            and not on_it
            and not truncated
            and labelled
            and all(v_ == VISIBLE for v_ in vis)
        ),
        "view_angle_deg": round(view_angle_deg(cam, np.asarray(T_plate)[:3, 3]), 3),
        "support": "plate" if on_plate else "parts",
        "lowest_z_mm": round(zmin, 4),
        "occluded_by": sorted(over),
        "lying_on_it": on_it,
        "free": not on_it,
        "nominal_length_mode": part.get("nominal_length_mode"),
        "mask_area_px": mask_area,
        "alone_area_px": alone_area[0],
        "alone_area_in_image_px": alone_area[1],
        "visible_pixel_fraction": round(min(1.0, mask_area / alone_area[0]), 4)
        if alone_area[0]
        else 0.0,
        "truncated": truncated,
    }
