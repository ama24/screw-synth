import json
import math

import numpy as np
import pytest

from screwsynth.calibration.geometry import Camera, nominal_camera
from screwsynth.measure.geometry import (
    ClassGeometry,
    Geometry,
    analytic_heights,
    camera_from_meta,
    confidence,
    measure,
    measure_length,
    side_heights,
    snap,
    thread_diameter_mm,
)


def _camera(distortion=True) -> Camera:
    cam = nominal_camera()
    if distortion:  # a realistic barrel distortion, like the Pi Camera 3 lens
        cam = Camera(
            cam.K, np.array([-0.08, 0.03, 0.0005, -0.0003, 0.0]), cam.image_size, cam.T_cam_plate
        )
    return cam


def _screw_points(length, z_anchor, z_tip, xy=(200.0, 40.0), yaw_deg=33.0):
    """Anchor and tip of a lying screw: 3D distance `length`, tilted by the height difference."""
    dz = z_anchor - z_tip
    horiz = math.sqrt(length**2 - dz**2)
    u = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    a = np.array([*xy, z_anchor])
    t = np.array([*(np.asarray(xy) + horiz * u), z_tip])
    return a, t


@pytest.mark.parametrize("length,z_a,z_t", [(16.0, 3.47, 1.91), (40.0, 6.5, 3.0), (6.0, 2.0, 1.0)])
@pytest.mark.parametrize("distortion", [False, True])
def test_parallax_and_tilt_within_005mm(length, z_a, z_t, distortion):
    """A screw of known length at known heights, projected with the camera, is measured back."""
    cam = _camera(distortion)
    a, t = _screw_points(length, z_a, z_t)  # 90 mm off the image centre: strong parallax
    pa, pt = cam.project(np.array([a, t]))
    geo = ClassGeometry("DIN912_M4", "DIN912", "M4", "bearing_to_tip", [length], 4.0, 3.5)
    heights = {"head_top": z_a, "bearing": z_a, "tip": z_t}
    L, a2, t2 = measure_length(cam, geo, pa, pt, heights=heights)
    assert pytest.approx(length, abs=0.05) == L
    assert np.allclose(a2, a, atol=0.05) and np.allclose(t2, t, atol=0.05)
    # ignoring parallax (both points on the plate) would be visibly wrong
    flat, _, _ = measure_length(cam, geo, pa, pt, heights={"head_top": 0, "bearing": 0, "tip": 0})
    assert abs(flat - length) > 0.1 or length < 10


def test_iteration_with_length_dependent_heights():
    """Heights from the class geometry (they change with length) converge on the true length."""
    cam = _camera()
    table = {
        L: {"head_top": 3.5, "bearing": 3.5 - 0.01 * L, "tip": 1.9} for L in (10, 16, 20, 25, 30)
    }
    geo = ClassGeometry(
        "DIN912_M4", "DIN912", "M4", "bearing_to_tip", sorted(table), 4.0, 3.5, table
    )
    true_L = 20.0
    h = geo.heights(true_L)
    a, t = _screw_points(true_L, h["bearing"], h["tip"])
    pa, pt = cam.project(np.array([a, t]))
    L, _, _ = measure_length(cam, geo, pa, pt)
    assert pytest.approx(true_L, abs=0.05) == L
    m = measure(cam, geo, {"bearing": pa, "tip": pt, "head_top": pa})
    assert m.snapped_mm == 20.0 and m.confidence > 0.9 and abs(m.tilt_deg) > 1


def test_countersunk_measures_from_head_top():
    geo = ClassGeometry("DIN7991_M4", "DIN7991", "M4", "head_top_to_tip", [16.0], 4.0, 4.0)
    assert geo.anchor == "head_top"
    cam = _camera()
    a, t = _screw_points(16.0, 3.0, 1.4)
    pa, pt = cam.project(np.array([a, t]))
    h = {"head_top": 3.0, "bearing": 2.8, "tip": 1.4}
    assert measure_length(cam, geo, pa, pt, heights=h)[0] == pytest.approx(16.0, abs=0.05)


def test_snap_edge_cases():
    lengths = [8, 10, 12, 16, 20]
    assert snap(10.2, lengths)[:2] == (10, pytest.approx(0.2))
    s, r, half = snap(11.0, lengths)  # exactly between 10 and 12: the shorter one, deterministic
    assert s == 10 and r == pytest.approx(1.0) and half == pytest.approx(1.0)
    s, r, half = snap(14.5, lengths)  # gap 12 -> 16 is 4 mm
    assert s == 16 and half == pytest.approx(2.0)
    assert snap(5.0, lengths)[0] == 8  # below the list: nearest end
    assert snap(31.0, lengths)[0] == 20  # length not in the enabled list: nearest end
    assert snap(7.0, [7])[0] == 7  # single length
    assert snap(3.0, []) == (None, None, math.inf)


def test_confidence_rules():
    assert confidence(0.0, 1.0)[0] == pytest.approx(1.0)
    c, notes = confidence(0.9, 1.0)
    assert c == pytest.approx(0.1) and notes  # near the middle between two lengths
    assert confidence(0.0, 1.0, measurable=False)[0] < 0.5
    c, notes = confidence(0.1, 1.0, disagreement_mm=1.5)
    assert c < 0.5 and any("disagree" in n for n in notes)
    assert confidence(None, math.inf)[0] == 0.0


def test_side_heights_and_class_table(tmp_path):
    rest = {
        "poses": [
            {
                "category": "side",
                "probability": 0.75,
                "keypoint_heights_mm": {"head_top": 4, "bearing": 3, "tip": 2},
            },
            {
                "category": "side",
                "probability": 0.25,
                "keypoint_heights_mm": {"head_top": 8, "bearing": 7, "tip": 2},
            },
            {
                "category": "standing_on_head",
                "probability": 0.5,
                "keypoint_heights_mm": {"head_top": 0, "bearing": 5, "tip": 20},
            },
        ]
    }
    assert side_heights(rest) == {"head_top": 5.0, "bearing": 4.0, "tip": 2.0}
    assert side_heights({"poses": []}) is None
    geo = ClassGeometry(
        "X",
        "X",
        "M4",
        "bearing_to_tip",
        [10, 20],
        4,
        3.5,
        {10: {"head_top": 4, "bearing": 3, "tip": 2}, 20: {"head_top": 4, "bearing": 5, "tip": 2}},
    )
    assert geo.heights(15)["bearing"] == pytest.approx(4.0)  # interpolated
    assert geo.heights(99)["bearing"] == 5  # clamped
    assert analytic_heights(4, 3.5) == {"head_top": 3.5, "bearing": 3.5, "tip": 2.0}


def test_geometry_from_shipped_catalogue(tmp_path):
    from screwsynth.common.config import Config, load_config

    cfg = Config(
        {
            **load_config().to_dict(),
            "paths": {**load_config().to_dict()["paths"], "rest_poses_dir": str(tmp_path)},
        }
    )
    (tmp_path / "DIN912_M4x16.json").write_text(
        json.dumps(
            {
                "poses": [
                    {
                        "category": "side",
                        "probability": 1,
                        "keypoint_heights_mm": {"head_top": 3.8, "bearing": 3.5, "tip": 1.9},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    geo = Geometry.from_config(cfg)
    g = geo["DIN912_M4"]
    assert 16.0 in g.lengths and g.mode == "bearing_to_tip" and g.d == 4.0
    assert g.heights(16.0)["bearing"] == pytest.approx(3.5)  # from the rest poses
    assert geo["DIN7991_M4"].mode == "head_top_to_tip"
    assert geo["DIN933_M8"].head_radius == pytest.approx(13.0 / 2)  # hex: rests on a flat
    assert "DIN934_M4" not in geo  # nuts are not measured


def test_thread_diameter_and_scene_camera():
    cam = nominal_camera()
    centre = cam.project(np.array([[110.0, 110.0, 2.0]]))[0]
    px_per_mm = 3385.714 / (300 - 2.0)
    d = thread_diameter_mm([4.0 * px_per_mm] * 5, cam, centre, 2.0)
    assert d == pytest.approx(4.0, abs=0.02)
    meta_cam = {
        "K": [[846.4, 0, 319.5], [0, 846.4, 319.5], [0, 0, 1]],
        "dist_coeffs": [-0.1, 0, 0, 0, 0],
        "resolution": [640, 640],
        "T_cam_plate": np.eye(4).tolist(),
        "distorted": False,
    }
    assert np.all(camera_from_meta(meta_cam).dist == 0)  # rendered, undistorted image
    assert camera_from_meta({**meta_cam, "distorted": True}).dist[0] == -0.1  # processed image


def test_cut_to_length_rounds_instead_of_snapping():
    """Threaded rod: any length, so the result is the nearest mm, not the nearest list entry."""
    cam = _camera()
    geo = ClassGeometry(
        "DIN975_M5",
        "DIN975",
        "M5",
        "head_top_to_tip",
        [35.0, 40.0, 45.0],
        5.0,
        2.5,
        cut_to_length=True,
    )
    a, t = _screw_points(37.2, 2.5, 2.5)
    pa, pt = cam.project(np.array([a, t]))
    m = measure(cam, geo, {"head_top": pa, "bearing": pa, "tip": pt})
    assert m.snapped_mm == 37.0  # a list snap would give 35
    assert m.raw_mm == pytest.approx(37.2, abs=0.1)
    assert m.confidence > 0.9 and "rounded to 1 mm" in " ".join(m.notes)
