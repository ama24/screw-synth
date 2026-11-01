"""Inference: photo -> ROI -> stage 1 at 1024 px -> per screw stage 2 + classical -> JSON.

Per object: class, size, detection confidence, polygon and box (ROI pixels, full resolution),
keypoints, length (raw / snapped / residual / confidence / method), position on the plate (mm)
and orientation (deg) for a future pick-and-place head, and consistency flags.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.calibration.geometry import Camera, nominal_camera, roi_camera_matrix, roi_for_area
from screwsynth.common.naming import UNKNOWN

from .classical import classical_keypoints, to_gray
from .geometry import ClassGeometry, Geometry, measure, nearest_size, thread_diameter_mm

log = logging.getLogger("screwsynth.infer")

METRIC_SIZES = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


@dataclass
class Detection:
    """One stage-1 detection in full-resolution ROI pixel coordinates."""

    class_id: int
    class_name: str
    score: float
    polygon: np.ndarray  # (N, 2)
    box: tuple[float, float, float, float]  # x0, y0, x1, y1

    def mask_window(self, shape: tuple[int, int], margin: int = 8) -> tuple[np.ndarray, int, int]:
        """Boolean mask of the polygon inside its bounding window (+margin): (mask, x0, y0)."""
        h, w = shape
        x0 = max(0, int(math.floor(self.box[0])) - margin)
        y0 = max(0, int(math.floor(self.box[1])) - margin)
        x1 = min(w, int(math.ceil(self.box[2])) + margin + 1)
        y1 = min(h, int(math.ceil(self.box[3])) + margin + 1)
        m = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
        pts = np.round((self.polygon - [x0, y0]) * 8).astype(np.int32)  # 3 bits of sub-pixel
        cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=3)
        return m.astype(bool), x0, y0


@dataclass
class Settings:
    stage1_imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.6
    max_det: int = 300
    crop_size_px: int = 384
    pose_conf: float = 0.5  # keypoint confidence needed to trust the keypoint model
    measurable_tilt_deg: float = 15.0
    border_px: int = 3  # masks touching the ROI border within this are "cut by the border"
    method: str = "auto"  # auto (A when confident, else B) | keypoints | classical

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Settings:
        d = cls()
        m = dict(cfg.get("measure", {}) or {}).get("infer", {}) or {}
        return cls(
            stage1_imgsz=int(
                m.get("stage1_imgsz", cfg.get("labels", {}).get("stage1_imgsz", d.stage1_imgsz))
            ),
            conf=float(m.get("conf", d.conf)),
            iou=float(m.get("iou", d.iou)),
            max_det=int(m.get("max_det", d.max_det)),
            crop_size_px=int(
                m.get("crop_size_px", cfg.get("labels", {}).get("crop_size_px", d.crop_size_px))
            ),
            pose_conf=float(m.get("pose_conf", d.pose_conf)),
            measurable_tilt_deg=float(
                m.get(
                    "measurable_tilt_deg",
                    cfg.get("restposes", {}).get("measurable_tilt_deg", d.measurable_tilt_deg),
                )
            ),
            border_px=int(m.get("border_px", d.border_px)),
            method=str(m.get("method", d.method)),
        )


@dataclass
class Scene:
    """The ROI image and its camera (ROI pixel coordinates, with the lens distortion)."""

    image: np.ndarray  # BGR, full-resolution ROI
    cam: Camera
    roi: dict[str, int]
    camera_source: str
    gray: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gray is None:
            self.gray = to_gray(self.image)


def load_scene(image_path: Path, calibration: Mapping[str, Any] | None) -> Scene:
    """Read a photo (full sensor frame, or already the ROI) and build the ROI camera."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    if calibration is not None:
        cam_full = Camera.from_calibration(calibration)
        roi = dict(
            calibration.get("roi")
            or {"x0": 0, "y0": 0, "w": cam_full.image_size[0], "h": cam_full.image_size[1]}
        )
        source = "calibration"
    else:
        cam_full = nominal_camera()
        roi = roi_for_area(cam_full, (0.0, 0.0), (220.0, 220.0), 5.0)
        source = "nominal"
    h, w = img.shape[:2]
    if (w, h) == tuple(cam_full.image_size):
        img = img[roi["y0"] : roi["y0"] + roi["h"], roi["x0"] : roi["x0"] + roi["w"]]
    elif (w, h) != (roi["w"], roi["h"]):
        raise ValueError(
            f"{image_path}: {w}x{h} is neither the full frame {cam_full.image_size} nor the ROI "
            f"{roi['w']}x{roi['h']} of the {source} camera"
        )
    cam = Camera(
        roi_camera_matrix(cam_full.K, roi),
        cam_full.dist,
        (roi["w"], roi["h"]),
        cam_full.T_cam_plate,
    )
    return Scene(np.ascontiguousarray(img), cam, roi, source)


def run_stage1(model: Any, scene: Scene, names: Mapping[int, str], st: Settings) -> list[Detection]:
    """Stage 1 on the ROI downscaled to stage1_imgsz; polygons and boxes back at full resolution."""
    h, w = scene.image.shape[:2]
    s = st.stage1_imgsz / max(h, w)
    small = cv2.resize(scene.image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    res = model.predict(
        small, imgsz=st.stage1_imgsz, conf=st.conf, iou=st.iou, max_det=st.max_det, verbose=False
    )[0]
    dets = []
    if res.masks is None or res.boxes is None:
        return dets
    for poly, box, cls, score in zip(
        res.masks.xy,
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
        strict=True,
    ):
        if len(poly) < 3:
            continue
        full = (np.asarray(poly, float) + 0.5) / s - 0.5  # same convention as tool 07
        b = (np.asarray(box, float) + 0.5) / s - 0.5
        cid = int(cls)
        dets.append(Detection(cid, names.get(cid, str(cid)), float(score), full, tuple(b.tolist())))
    return dets


def run_pose(
    model: Any, scene: Scene, det: Detection, st: Settings
) -> tuple[dict[str, np.ndarray], float] | None:
    """Stage 2 keypoints on a full-resolution crop centred on the detection (tool 07's crops)."""
    h, w = scene.image.shape[:2]
    side = st.crop_size_px
    cx, cy = (det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2
    x0 = int(np.clip(round(cx - side / 2), 0, max(0, w - side)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(0, h - side)))
    crop = scene.image[y0 : y0 + side, x0 : x0 + side]
    res = model.predict(crop, imgsz=side, conf=0.1, verbose=False)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
    ious = [_iou(b, det.box) for b in boxes]
    i = int(np.argmax(ious))
    if ious[i] < 0.3:
        return None
    xy = res.keypoints.xy.cpu().numpy()[i] + [x0, y0]
    conf = res.keypoints.conf.cpu().numpy()[i] if res.keypoints.conf is not None else np.ones(3)
    return {k: xy[j] for j, k in enumerate(("head_top", "bearing", "tip"))}, float(np.min(conf))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def measure_screw(
    scene: Scene,
    det: Detection,
    geo: ClassGeometry,
    st: Settings,
    pose: tuple[dict[str, np.ndarray], float] | None = None,
    mask: tuple[np.ndarray, int, int] | None = None,
) -> dict[str, Any]:
    """Both methods on one screw, the chosen result, and consistency checks."""
    out: dict[str, Any] = {}
    m, mx0, my0 = mask if mask is not None else det.mask_window(scene.image.shape[:2])
    h, w = scene.image.shape[:2]
    cut = (
        mx0 <= st.border_px
        or my0 <= st.border_px
        or mx0 + m.shape[1] >= w - st.border_px
        or my0 + m.shape[0] >= h - st.border_px
    ) and _touches_border(m, mx0, my0, w, h, st.border_px)
    # method B: classical, on the full-resolution window
    gray = scene.gray[my0 : my0 + m.shape[0], mx0 : mx0 + m.shape[1]]
    win_cam = Camera(
        roi_camera_matrix(scene.cam.K, {"x0": mx0, "y0": my0}),
        scene.cam.dist,
        (m.shape[1], m.shape[0]),
        scene.cam.T_cam_plate,
    )
    b = classical_keypoints(gray, m, win_cam, geo)
    kp_b = {k: v + [mx0, my0] for k, v in b[0].items()} if b else None
    meas_b = measure(scene.cam, geo, kp_b) if kp_b else None
    if meas_b is not None and not _plausible(meas_b.raw_mm, geo):
        meas_b, kp_b = None, None  # degenerate edge result (e.g. a merged or broken mask)
    # method A: keypoint model
    kp_a = pose[0] if pose else None
    pose_ok = bool(pose and pose[1] >= st.pose_conf)
    meas_a = measure(scene.cam, geo, kp_a) if kp_a else None
    if meas_a is not None and not _plausible(meas_a.raw_mm, geo):
        meas_a, kp_a, pose_ok = None, None, False
    tilt = (meas_a or meas_b).tilt_deg if (meas_a or meas_b) else 90.0
    measurable = not cut and abs(tilt) <= st.measurable_tilt_deg
    use_a = meas_a is not None and (st.method == "keypoints" or (st.method == "auto" and pose_ok))
    if st.method == "classical" or meas_a is None or not use_a:
        chosen, method, other = meas_b, "classical", meas_a
    else:
        chosen, method, other = meas_a, "keypoints", meas_b
    if chosen is not None:
        final = measure(
            scene.cam,
            geo,
            kp_a if method == "keypoints" else kp_b,
            measurable,
            other.raw_mm if other else None,
        )
        out["length"] = {**final.to_dict(), "method": method}
    out["length_keypoints_mm"] = round(meas_a.raw_mm, 3) if meas_a else None
    out["length_classical_mm"] = round(meas_b.raw_mm, 3) if meas_b else None
    out["keypoints"] = {
        "keypoints": {k: np.round(v, 2).tolist() for k, v in kp_a.items()} if kp_a else None,
        "keypoint_confidence": round(pose[1], 3) if pose else None,
        "classical": {k: np.round(v, 2).tolist() for k, v in kp_b.items()} if kp_b else None,
    }
    out["measurable"] = measurable
    out["cut_by_border"] = bool(cut)
    if b is not None and b[1].shank_widths_px and geo.kind == "screw":  # standoffs: hex body
        h_mid = geo.heights((geo.lengths or [10])[len(geo.lengths or [10]) // 2])["tip"]
        centre = kp_b["tip"] if kp_b else np.mean(det.polygon, axis=0)
        dia = thread_diameter_mm(b[1].shank_widths_px, scene.cam, centre, h_mid)
        size = nearest_size(dia, METRIC_SIZES)
        out["thread_diameter_mm"] = round(dia, 2)
        out["size_check"] = (
            "ok"
            if abs(size - geo.d) < 1e-6
            else f"mask width suggests M{size:g}, class is {geo.size}"
        )
    return out


def _plausible(length_mm: float, geo: ClassGeometry) -> bool:
    longest = max(geo.lengths) if geo.lengths else 100.0
    return bool(
        np.isfinite(length_mm) and 0.3 * min(geo.lengths or [1.0]) <= length_mm <= 2 * longest + 20
    )


def _touches_border(m: np.ndarray, x0: int, y0: int, w: int, h: int, b: int) -> bool:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return False
    xs, ys = xs + x0, ys + y0
    return bool(xs.min() <= b or ys.min() <= b or xs.max() >= w - 1 - b or ys.max() >= h - 1 - b)


def plate_pose(scene: Scene, det: Detection, height_mm: float) -> tuple[list[float], float]:
    """Centre (mm) and orientation (deg, principal axis in the plate frame) of a detection."""
    pts = scene.cam.pixel_to_plate(det.polygon, height_mm)[:, :2]
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T) if len(pts) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    ax = evecs[:, 1]
    angle = math.degrees(math.atan2(ax[1], ax[0])) % 180.0
    return [round(float(c[0]), 2), round(float(c[1]), 2)], round(angle, 1)


class Pipeline:
    """Loaded models + geometry; call `run(image)`."""

    def __init__(
        self,
        stage1: str | Path,
        stage2: str | Path | None,
        geometry: Geometry,
        settings: Settings,
        calibration: Mapping[str, Any] | None,
        classes_yaml: str | Path | None = None,
    ) -> None:
        from ultralytics import YOLO

        from .modelcard import check_classes

        self.stage1 = YOLO(str(stage1))
        self.stage2 = YOLO(str(stage2)) if stage2 else None
        self.names = {int(k): v for k, v in self.stage1.names.items()}
        errors, warnings = check_classes(stage1, self.names, classes_yaml)
        if errors:
            raise ValueError("; ".join(errors))
        for w in warnings:
            log.warning("stage-1 model: %s", w)
        self.geo = geometry
        self.st = settings
        self.calibration = calibration

    def run(self, image_path: Path) -> tuple[dict[str, Any], Scene, list[Detection]]:
        t0 = time.perf_counter()
        scene = load_scene(image_path, self.calibration)
        result, dets = self.run_scene(scene, str(image_path), t0)
        return result, scene, dets

    def run_scene(
        self, scene: Scene, name: str = "", t0: float | None = None
    ) -> tuple[dict[str, Any], list[Detection]]:
        """Everything after loading: stage 1, then stage 2 and the measurement per screw."""
        t0 = time.perf_counter() if t0 is None else t0
        dets = run_stage1(self.stage1, scene, self.names, self.st)
        t1 = time.perf_counter()
        objects = []
        for i, det in enumerate(dets, start=1):
            obj: dict[str, Any] = {
                "id": i,
                "class": det.class_name,
                "class_id": det.class_id,
                "unknown": det.class_name == UNKNOWN,
                "score": round(det.score, 3),
                "box": [round(v, 1) for v in det.box],
                "polygon": np.round(det.polygon, 1).tolist(),
            }
            geo = self.geo.classes.get(det.class_name)
            height = 1.5
            if geo is not None:
                obj["size"] = geo.size
                pose = run_pose(self.stage2, scene, det, self.st) if self.stage2 else None
                obj.update(measure_screw(scene, det, geo, self.st, pose))
                height = geo.d / 2
            obj["position_mm"], obj["orientation_deg"] = plate_pose(scene, det, height)
            objects.append(obj)
        result = {
            "image": name,
            "camera": scene.camera_source,
            "roi": scene.roi,
            "objects": objects,
            "timing_s": {"stage1": round(t1 - t0, 3), "total": round(time.perf_counter() - t0, 3)},
        }
        return result, dets


def draw(scene: Scene, result: Mapping[str, Any], max_side: int = 2048) -> np.ndarray:
    """Visualisation: outlines, classes, lengths and keypoints on the ROI image."""
    img = scene.image.copy()
    for o in result["objects"]:
        poly = np.round(np.asarray(o["polygon"])).astype(np.int32)
        colour = (
            (180, 0, 180)
            if o.get("unknown")
            else (0, 200, 0)
            if o.get("length", {}).get("confidence", 0) > 0.5
            else (0, 160, 255)
        )
        cv2.polylines(img, [poly], True, colour, 3)
        label = o["class"]
        if o.get("length"):
            L = o["length"]
            label += (
                f" {L['length_mm']:g} mm"
                if L["length_mm"] is not None
                else f" ~{L['length_raw_mm']:.1f}"
            )
        x0, y0 = int(o["box"][0]), int(o["box"][1])
        cv2.putText(
            img, label, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA
        )
        kp = (o.get("keypoints") or {}).get("keypoints") or (o.get("keypoints") or {}).get(
            "classical"
        )
        if kp:
            for name, c in (
                ("head_top", (0, 0, 255)),
                ("bearing", (0, 255, 0)),
                ("tip", (255, 0, 0)),
            ):
                cv2.circle(img, (int(round(kp[name][0])), int(round(kp[name][1]))), 6, c, -1)
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(
            img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA
        )
    n = len(result["objects"])
    summary = f"{n} object(s), {sum(1 for o in result['objects'] if o.get('length'))} measured"
    for colour, thick in (((0, 0, 0), 6), ((255, 255, 255), 2)):  # outlined: readable on any plate
        cv2.putText(
            img, summary, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, thick, cv2.LINE_AA
        )
    return img
