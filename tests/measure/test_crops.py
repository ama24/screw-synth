import cv2
import numpy as np
import pytest

from screwsynth.calibration.geometry import nominal_camera, nominal_from_config
from screwsynth.measure.crops import crop_window
from screwsynth.measure.infer import Detection, Scene, Settings, load_scene, run_pose


def test_crop_window_fixed_and_scaled():
    long_box = (1000.0, 1000.0, 1900.0, 1100.0)  # a 77 mm rod at 11.7 px/mm
    x0, y0, side, scale = crop_window(long_box, (2560, 2560), 384, "fixed")
    assert (side, scale) == (384, 1.0)  # cut off: only the middle third fits
    x0, y0, side, scale = crop_window(long_box, (2560, 2560), 384, "scaled", 1.2)
    assert side == 1080 and scale == pytest.approx(384 / 1080)
    assert x0 <= long_box[0] and x0 + side >= long_box[2]  # the whole part is in the window
    short = (500.0, 500.0, 600.0, 540.0)
    assert crop_window(short, (2560, 2560), 384, "scaled")[2:] == (384, 1.0)  # short: unchanged
    assert crop_window((0, 0, 50, 50), (2560, 2560), 384, "fixed")[:2] == (0, 0)  # clamped
    with pytest.raises(ValueError):
        crop_window(short, (2560, 2560), 384, "zoomed")


class _T:
    """Stand-in for a torch tensor: .cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakePose:
    """Keypoint model that 'finds' known full-resolution points, seen through the crop."""

    def __init__(self, full_kps, full_box, window):
        self.full_kps, self.full_box, self.window = np.asarray(full_kps), full_box, window

    def predict(self, crop, imgsz, conf, verbose):
        x0, y0, side, scale = self.window
        assert crop.shape[:2] == (imgsz, imgsz)

        def to_crop(v):  # the export's mapping (labels.make_crop)
            return (v + 0.5 - [x0, y0]) * scale - 0.5

        b = self.full_box
        box = np.r_[to_crop(np.array(b[:2])), to_crop(np.array(b[2:]))]
        res = type("R", (), {})()
        res.boxes = type("B", (), {"xyxy": _T([box]), "__len__": lambda s: 1})()
        res.keypoints = type(
            "K", (), {"xy": _T([to_crop(self.full_kps)]), "conf": _T([[0.9] * 3])}
        )()
        return [res]


@pytest.mark.parametrize("mode", ["fixed", "scaled"])
def test_run_pose_maps_scaled_crops_back(mode):
    img = np.zeros((2560, 2560, 3), np.uint8)
    scene = Scene(img, nominal_camera(), {"x0": 0, "y0": 0, "w": 2560, "h": 2560}, "nominal")
    box = (1000.0, 1000.0, 1900.0, 1100.0)
    kps = [[1010.0, 1050.0], [1010.0, 1050.0], [1890.0, 1050.0]]  # both rod ends, 880 px apart
    det = Detection(0, "DIN975_M5", 0.9, np.array([[1000, 1000], [1900, 1100]]), box)
    st = Settings(crop_mode=mode)
    window = crop_window(box, (2560, 2560), st.crop_size_px, mode, st.crop_margin)
    found = run_pose(_FakePose(kps, box, window), scene, det, st)
    assert found is not None
    pts, conf = found
    assert np.allclose(pts["head_top"], kps[0], atol=1e-6)
    assert np.allclose(pts["tip"], kps[2], atol=1e-6)  # in fixed mode it lies outside the crop


def test_nominal_camera_height_comes_from_the_config(tmp_path):
    class Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    cam = nominal_from_config(Cfg({"scenegen.nominal_camera": {"height_mm": 340}}))
    assert cam.T_cam_plate[2, 3] == pytest.approx(340)
    assert nominal_from_config(Cfg()).T_cam_plate[2, 3] == pytest.approx(300)
    photo = tmp_path / "frame.png"
    cv2.imwrite(str(photo), np.zeros((2592, 4608, 3), np.uint8))
    near, far = load_scene(photo, None), load_scene(photo, None, cam)
    # higher camera: each mm on the plate covers fewer pixels, so the same area needs a smaller ROI
    assert far.roi["w"] < near.roi["w"]
    assert far.cam.T_cam_plate[2, 3] == pytest.approx(340)
