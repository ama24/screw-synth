"""Mesh validity and dimension checks against the catalogue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import trimesh

from screwsynth.catalogue.schema import LENGTH_KINDS, Part

from .collision import MAX_TRIANGLES, contained_fraction, total_triangles
from .keypoints import nominal_length


def _close(name: str, measured: float, expected: float, tol: float, out: list[str]) -> None:
    if abs(measured - expected) > tol:
        out.append(f"{name}: measured {measured:.4f}, expected {expected:.4f} (tol {tol})")


def _radial(v: np.ndarray) -> np.ndarray:
    return np.hypot(v[:, 0], v[:, 1])


def check_mesh(mesh: trimesh.Trimesh, max_triangles: int) -> list[str]:
    out = []
    if not mesh.is_watertight:
        out.append("mesh is not watertight")
    if not mesh.is_winding_consistent:
        out.append("mesh winding is inconsistent")
    if mesh.is_watertight and mesh.volume <= 0:
        out.append("mesh volume is not positive (normals point inwards)")
    if len(mesh.faces) > max_triangles:
        out.append(f"{len(mesh.faces)} triangles exceed the budget of {max_triangles}")
    return out


def check_dimensions(part: Part, mesh: trimesh.Trimesh, tol: float = 0.01) -> list[str]:
    """Compare the mesh with the catalogue: overall size, head, flats, thread major diameter."""
    if part.source == "step":
        return []
    d = part.dims
    v = mesh.vertices
    out: list[str] = []
    zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
    if part.shape in ("set_screw", "threaded_rod"):
        _close("head_top z", zmax, 0.0, tol, out)
        _close("overall length", zmax - zmin, part.length or 0.0, tol, out)
        _close("thread major diameter", float(_radial(v).max()) * 2, d["d"], tol, out)
    elif part.kind == "standoff":
        L, ls = part.length or 0.0, d.get("ls", 0.0)
        top = ls if part.shape == "standoff_mm" else 0.0
        bottom = -L if part.shape == "standoff_ff" else -L - ls
        _close("top z", zmax, top, tol, out)
        _close("bottom z", zmin, bottom, tol, out)
        hexa = v[(v[:, 2] > -L + 0.02) & (v[:, 2] < -0.02)]
        _close("across flats s", float(np.abs(hexa[:, 1]).max()) * 2, d["s"], tol, out)
    elif part.kind == "bracket":
        ext = v.max(axis=0) - v.min(axis=0)
        _close("leg x (with key)", float(ext[0]), d["a"] + d["key_h"], tol, out)
        _close("width", float(ext[1]), d["w"], tol, out)
        _close("leg z (with key)", float(ext[2]), d["a"] + d["key_h"], tol, out)
    elif part.kind == "screw":
        L = part.length or 0.0
        k = d["k"]
        _close("head top z", zmax, k, tol, out)
        overall = L if part.nominal_length == "head_top_to_tip" else L + k
        _close("overall length", zmax - zmin, overall, tol, out)
        head = v[v[:, 2] > 0.02]
        if part.shape == "hex":
            _close("across flats s", float(np.abs(head[:, 1]).max()) * 2, d["s"], tol, out)
            corners = float(_radial(head).max()) * 2
            if corners < d["e"] - tol:
                out.append(f"across corners {corners:.3f} < e min {d['e']}")
        else:
            _close("head diameter dk", float(_radial(head).max()) * 2, d["dk"], tol, out)
        shank = v[v[:, 2] < -0.02]
        _close("thread major diameter", float(_radial(shank).max()) * 2, d["d"], tol, out)
    elif part.kind == "nut":
        height = d["h"] if part.shape == "nyloc_nut" else d["m"]
        _close("height", zmax - zmin, height, tol, out)
        _close("across flats s", float(np.abs(v[:, 1]).max()) * 2, d["s"], tol, out)
        mid = v[np.abs(v[:, 2] - (zmin + min(d["m"], height) / 2)) < (part.pitch or 1)]
        minor = d["d"] - 2 * 0.54127 * (part.pitch or 0)
        _close(
            "thread minor diameter",
            float(_radial(mid).min()) * 2,
            minor + 0.04 * (part.pitch or 0),
            0.05,
            out,
        )
    elif part.kind == "washer":
        r = _radial(v)
        _close("outer diameter d2", float(r.max()) * 2, d["d2"], tol, out)
        _close("inner diameter d1", float(r.min()) * 2, d["d1"], tol, out)
        _close("thickness h", zmax - zmin, d["h"], tol, out)
    elif part.kind == "tnut":
        ext = v.max(axis=0) - v.min(axis=0)
        _close("length", float(ext[0]), d["length"], tol, out)
        _close("width", float(ext[1]), d["width"], tol, out)
        _close("height", float(ext[2]), d["height"], tol, out)
    return out


def check_keypoints(
    part: Part, keypoints: Mapping[str, Sequence[float]], tol: float = 0.01
) -> list[str]:
    if part.kind not in LENGTH_KINDS or part.length is None:
        return []
    out: list[str] = []
    _close("keypoint nominal length", nominal_length(part, keypoints) or 0.0, part.length, tol, out)
    return out


def check_collision(
    mesh: trimesh.Trimesh, pieces: list[trimesh.Trimesh], min_fraction: float = 0.95
) -> list[str]:
    out = []
    n = total_triangles(pieces)
    if n > MAX_TRIANGLES:
        out.append(f"collision proxy has {n} triangles (> {MAX_TRIANGLES})")
    if any(not p.is_convex for p in pieces):
        out.append("collision piece is not convex")
    frac = contained_fraction(mesh, pieces)
    if frac < min_fraction:
        out.append(f"collision proxy contains only {frac:.1%} of the render volume")
    return out
