"""Camera geometry shared by calibration, rendering and measurement (numpy + OpenCV).

Frames: plate frame P (mm, z = 0 on the TPU surface, z up towards the camera) and camera frame C
(OpenCV: x right, y down, z forward). T_cam_plate maps P -> C: X_c = R X_p + t.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Camera:
    K: np.ndarray  # 3x3, full-image pixel coordinates
    dist: np.ndarray  # OpenCV distortion coefficients (5, 8, ...)
    image_size: tuple[int, int]  # (width, height) of the full image
    T_cam_plate: np.ndarray | None = None  # 4x4

    @classmethod
    def from_calibration(cls, calib: Mapping[str, Any]) -> Camera:
        T = calib.get("extrinsic", {}).get("T_cam_plate")
        return cls(
            K=np.asarray(calib["camera_matrix"], dtype=float),
            dist=np.asarray(calib["dist_coeffs"], dtype=float).ravel(),
            image_size=tuple(calib["image_size"]),
            T_cam_plate=None if T is None else np.asarray(T, dtype=float),
        )

    @property
    def R(self) -> np.ndarray:
        assert self.T_cam_plate is not None, "camera has no extrinsic calibration"
        return self.T_cam_plate[:3, :3]

    @property
    def t(self) -> np.ndarray:
        assert self.T_cam_plate is not None, "camera has no extrinsic calibration"
        return self.T_cam_plate[:3, 3]

    @property
    def centre_in_plate(self) -> np.ndarray:
        """Camera centre in the plate frame (mm)."""
        return -self.R.T @ self.t

    def project(self, X_plate: np.ndarray, distort: bool = True) -> np.ndarray:
        """Plate-frame points (N, 3) mm -> pixels (N, 2) in the full image."""
        X = np.asarray(X_plate, dtype=float).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        dist = self.dist if distort else np.zeros(5)
        px, _ = cv2.projectPoints(X, rvec, self.t.reshape(3, 1), self.K, dist)
        return px.reshape(-1, 2)

    def undistort_points(self, px: np.ndarray) -> np.ndarray:
        """Distorted pixels -> undistorted pixels (same K)."""
        pts = np.asarray(px, dtype=float).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def rays(self, px: np.ndarray, distorted: bool = True) -> np.ndarray:
        """Unit ray directions in the plate frame for pixels."""
        pts = np.asarray(px, dtype=float).reshape(-1, 1, 2)
        if distorted:
            n = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        else:
            n = cv2.undistortPoints(pts, self.K, np.zeros(5)).reshape(-1, 2)
        d_cam = np.c_[n, np.ones(len(n))]
        d = d_cam @ self.R  # R^T d for each row
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def pixel_to_plate(
        self, px: np.ndarray, height_mm: float | np.ndarray = 0.0, distorted: bool = True
    ) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane z = height_mm (parallax-correct)."""
        d = self.rays(px, distorted)
        c = self.centre_in_plate
        s = (np.asarray(height_mm, dtype=float) - c[2]) / d[:, 2]
        return c + s[:, None] * d

    def homography(self) -> np.ndarray:
        """Plate (x, y) mm at z = 0 -> undistorted full-image pixels."""
        H = self.K @ np.c_[self.R[:, 0], self.R[:, 1], self.t]
        return H / H[2, 2]

    def px_per_mm(self, at_plate_xy: Sequence[float] = (0.0, 0.0)) -> float:
        x, y = at_plate_xy
        pts = np.array([[x, y, 0.0], [x + 1, y, 0.0], [x, y + 1, 0.0]])
        p = self.project(pts, distort=False)
        return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2)


def roi_camera_matrix(K: np.ndarray, roi: Mapping[str, int]) -> np.ndarray:
    """Camera matrix of a crop: the principal point moves by the ROI offset."""
    Kr = np.array(K, dtype=float)
    Kr[0, 2] -= roi["x0"]
    Kr[1, 2] -= roi["y0"]
    return Kr


def nominal_camera(
    focal_mm: float = 4.74,
    pixel_um: float = 1.4,
    size: tuple[int, int] = (4608, 2592),
    height_mm: float = 300.0,
    plate_centre: tuple[float, float] = (110.0, 110.0),
) -> Camera:
    """Ideal Pi Camera 3 looking straight down at the plate centre (no distortion)."""
    f = focal_mm / (pixel_um * 1e-3)
    w, h = size
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    # camera x = plate x, camera y = -plate y, camera z = -plate z (looking down)
    R = np.diag([1.0, -1.0, -1.0])
    C = np.array([plate_centre[0], plate_centre[1], height_mm])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return Camera(K, np.zeros(5), size, T)


def nominal_from_config(cfg: Any) -> Camera:
    """The nominal camera as configured in scenegen.nominal_camera (focal length, pixel size,
    resolution, height): the same camera --nominal-camera renders with."""
    nc = dict(cfg.get("scenegen.nominal_camera", {}) or {})
    return nominal_camera(
        focal_mm=float(nc.get("focal_length_mm", 4.74)),
        pixel_um=float(nc.get("pixel_size_um", 1.4)),
        size=tuple(int(v) for v in nc.get("resolution", (4608, 2592))),
        height_mm=float(nc.get("height_mm", 300.0)),
    )


def roi_for_area(
    cam: Camera,
    origin_mm: Sequence[float],
    size_mm: Sequence[float],
    margin_mm: float,
    multiple: int = 32,
) -> dict[str, int]:
    """Pixel ROI covering a plate rectangle (+ margin), rounded out to a multiple, clipped."""
    x0, y0 = origin_mm[0] - margin_mm, origin_mm[1] - margin_mm
    x1, y1 = origin_mm[0] + size_mm[0] + margin_mm, origin_mm[1] + size_mm[1] + margin_mm
    xs, ys = np.linspace(x0, x1, 21), np.linspace(y0, y1, 21)
    border = np.array(
        [[x, y0, 0] for x in xs]
        + [[x, y1, 0] for x in xs]
        + [[x0, y, 0] for y in ys]
        + [[x1, y, 0] for y in ys]
    )
    px = cam.project(border)
    w_img, h_img = cam.image_size
    u0, v0 = np.floor(px.min(axis=0)).astype(int)
    u1, v1 = np.ceil(px.max(axis=0)).astype(int)
    w = int(np.ceil((u1 - u0) / multiple) * multiple)
    h = int(np.ceil((v1 - v0) / multiple) * multiple)
    w, h = min(w, w_img // multiple * multiple), min(h, h_img // multiple * multiple)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    x = int(np.clip(round(cu - w / 2), 0, w_img - w))
    y = int(np.clip(round(cv - h / 2), 0, h_img - h))
    return {"x0": x, "y0": y, "w": w, "h": h}
