"""Real captures -> labelling images -> processed-format scenes -> datasets (tool 07 formats).

The scenes look exactly like tool 06 output (image.png, instance.png, meta.json), so tool 07's
exporter turns them into the same YOLO/COCO/crops datasets. Every image of a session shares one
layout id: the photos taken after each removal show the same parts and must never be split across
train and val.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.calibration.geometry import Camera, nominal_camera, roi_camera_matrix, roi_for_area
from screwsynth.common.io import write_json
from screwsynth.common.naming import UNKNOWN, parse_part_id, screw_id

from .cvat import ImageAnn
from .matching import Match
from .session import InventoryRow, finish_of_class

log = logging.getLogger("screwsynth.realdata")


def roi_camera(
    calibration: Mapping[str, Any] | None, nominal: Camera | None = None
) -> tuple[Camera, dict[str, int], dict[str, Any]]:
    """ROI camera (pixel coordinates of the labelling images), its ROI and meta.json `camera`.
    Without a calibration: `nominal` (nominal_from_config), or the 300 mm default."""
    if calibration is not None:
        full = Camera.from_calibration(calibration)
        roi = dict(
            calibration.get("roi")
            or {"x0": 0, "y0": 0, "w": full.image_size[0], "h": full.image_size[1]}
        )
        source = "calibration"
    else:
        full = nominal if nominal is not None else nominal_camera()
        roi = roi_for_area(full, (0.0, 0.0), (220.0, 220.0), 5.0)
        source = "nominal"
    K = roi_camera_matrix(full.K, roi)
    cam = Camera(K, full.dist, (roi["w"], roi["h"]), full.T_cam_plate)
    meta = {
        "source": source,
        "K": K.tolist(),
        "K_full": full.K.tolist(),
        "dist_coeffs": np.asarray(full.dist).ravel().tolist(),
        "image_size_full": list(full.image_size),
        "roi": roi,
        "resolution": [roi["w"], roi["h"]],
        "resolution_scale": 1.0,
        "T_cam_plate": np.asarray(full.T_cam_plate).tolist(),
        "distorted": source == "calibration",  # real photos carry the real lens distortion
    }
    return cam, roi, meta


def prepare_images(
    session_dir: Path, calibration: Mapping[str, Any] | None, nominal: Camera | None = None
) -> list[Path]:
    """Crop the captured full frames to the ROI for labelling: annotations/images/<name>.png."""
    cam, roi, _ = roi_camera(calibration, nominal)
    out_dir = session_dir / "annotations" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted((session_dir / "images").glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if (w, h) != (roi["w"], roi["h"]):
            if w < roi["x0"] + roi["w"] or h < roi["y0"] + roi["h"]:
                raise ValueError(f"{p.name}: {w}x{h} does not contain the ROI {roi}")
            img = img[roi["y0"] : roi["y0"] + roi["h"], roi["x0"] : roi["x0"] + roi["w"]]
        dst = out_dir / p.name
        cv2.imwrite(str(dst), img)
        out.append(dst)
    return out


def measured_lengths(ann: ImageAnn, cam: Camera, geometry: Any) -> dict[int, float | None]:
    """Length of every screw with labelled keypoints (tool 08 measurement), for the matching."""
    from screwsynth.measure.geometry import measure_length

    out: dict[int, float | None] = {}
    for o in ann.objects:
        if o.label not in geometry:
            continue
        g = geometry[o.label]
        if g.anchor in o.keypoints and "tip" in o.keypoints:
            out[o.id] = measure_length(cam, g, o.keypoints[g.anchor], o.keypoints["tip"])[0]
        else:
            out[o.id] = None
    return out


def _family_size(cls: str) -> tuple[str, str]:
    if cls == UNKNOWN:
        return UNKNOWN, "-"
    try:
        p = parse_part_id(cls)
        return p.family if not p.is_tnut else "TNUT", p.size
    except ValueError:
        return cls.rsplit("_", 1)[0], cls.rsplit("_", 1)[-1]


def build_scenes(
    session_dir: Path,
    anns: Sequence[ImageAnn],
    matches: Mapping[tuple[str, int], Match],
    inventory: Sequence[InventoryRow],
    calibration: Mapping[str, Any] | None,
    class_ids: Mapping[str, int],
    catalogue: Any,
    nominal: Camera | None = None,
) -> list[Path]:
    """One processed-format scene per labelled image, in <session>/scenes/<image stem>/."""
    _, _, cam_meta = roi_camera(calibration, nominal)
    finishes = finish_of_class(inventory)
    kinds = {p.class_name: p for p in catalogue.parts}
    session = session_dir.name
    out = []
    for ann in anns:
        src = session_dir / "annotations" / "images" / ann.name
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("%s: image missing, skipped", src)
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint16)
        objs = [o for o in ann.objects if o.label in class_ids and len(o.polygon) >= 3]
        for o in ann.objects:
            if o.label not in class_ids:
                log.warning(
                    "%s: label %r is not a class of this dataset, skipped", ann.name, o.label
                )
        # larger parts first, so parts lying on top (usually smaller) win the overlap
        objs.sort(key=lambda o: -cv2.contourArea(np.asarray(o.polygon, np.float32)))
        records = []
        for k, o in enumerate(objs, start=1):
            cv2.fillPoly(mask, [np.round(np.asarray(o.polygon) * 8).astype(np.int32)], k, shift=3)
            ref = kinds.get(o.label)
            kind = ref.kind if ref is not None else ("unknown" if o.label == UNKNOWN else "screw")
            family, size = (ref.family, ref.size) if ref is not None else _family_size(o.label)
            m = matches.get((ann.name, o.id))
            length = m.length_mm if (m is not None and kind == "screw") else None
            part_id = (
                screw_id(family, size, length)
                if (kind == "screw" and length is not None)
                else o.label
            )
            labelled = kind == "screw" and bool(o.keypoints)
            kps = {
                name: {
                    "xyz": None,
                    "uv": list(o.keypoints[name]) if name in o.keypoints else [0.0, 0.0],
                    "visibility": 2 if (labelled and name in o.keypoints) else 0,
                }
                for name in ("head_top", "bearing", "tip")
            }
            records.append(
                {
                    "instance_id": k,
                    "part_id": part_id,
                    "class_name": o.label,
                    "class_id": int(class_ids[o.label]),
                    "kind": kind,
                    "family": family,
                    "size": size,
                    "length": length,
                    "finish": finishes.get(o.label, "unknown"),
                    "keypoints": kps,
                    "keypoints_labelled": labelled,
                    "measurable": labelled and len(o.keypoints) == 3 and length is not None,
                    "nominal_length_mode": ref.nominal_length if ref is not None else None,
                    "axis_tilt_deg": None,
                    "axis_height_head_mm": None,
                    "axis_height_tip_mm": None,
                    "axis_height_head_top_mm": None,
                    "match_status": m.status if m is not None else None,
                }
            )
        for r in records:
            area = int((mask == r["instance_id"]).sum())
            r["mask_area_px"] = area
            r["alone_area_px"] = area  # unknown for real photos: count what is visible
            r["visible_pixel_fraction"] = 1.0
            r["truncated"] = False
        d = session_dir / "scenes" / Path(ann.name).stem
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "image.png"), img)
        cv2.imwrite(str(d / "instance.png"), mask)
        write_json(
            d / "meta.json",
            {
                "scene_id": f"{session}_{Path(ann.name).stem}",
                "layout_id": session,  # all photos of a session stay in one split
                "source": "real",
                "session": session,
                "image": "image.png",
                "instance_mask": "instance.png",
                "camera": cam_meta,
                "objects": records,
            },
        )
        out.append(d)
    return out
