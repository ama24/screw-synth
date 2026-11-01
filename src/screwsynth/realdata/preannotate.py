"""Pre-annotate the labelling images with the current best models (tool 08), to save clicking."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2

from .build import roi_camera
from .cvat import ImageAnn, Obj

log = logging.getLogger("screwsynth.realdata")


def preannotate(
    images: list[Path],
    stage1: Path,
    stage2: Path | None,
    calibration: Mapping[str, Any] | None,
    cfg: Any,
) -> list[ImageAnn]:
    """Detections as polygons (class labels); keypoints for screws the pipeline found measurable."""
    from screwsynth.calibration.geometry import nominal_from_config
    from screwsynth.measure.geometry import Geometry
    from screwsynth.measure.infer import Pipeline, Scene, Settings

    nominal = nominal_from_config(cfg)
    cam, roi, meta = roi_camera(calibration, nominal)
    pipe = Pipeline(
        stage1,
        stage2,
        Geometry.from_config(cfg),
        Settings.from_config(cfg.to_dict()),
        calibration,
        nominal=nominal,
    )
    out = []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        scene = Scene(img, cam, roi, meta["source"])
        result, _ = pipe.run_scene(scene, p.name)
        objs = []
        for o in result["objects"]:
            kp: dict[str, tuple[float, float]] = {}
            # only screws lying flat get keypoints (see the labelling guide)
            if o.get("measurable"):
                found = o.get("keypoints") or {}
                src = found.get("keypoints") or found.get("classical")
                if src:
                    kp = {k: (float(v[0]), float(v[1])) for k, v in src.items()}
            objs.append(
                Obj(o["class"], [tuple(p) for p in o["polygon"]], kp, o["score"], len(objs) + 1)
            )
        out.append(ImageAnn(p.name, img.shape[1], img.shape[0], objs))
        log.info("%s: %d pre-annotated object(s)", p.name, len(objs))
    return out
