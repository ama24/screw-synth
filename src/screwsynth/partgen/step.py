"""STEP / external CAD route.

STEP files are converted with cadquery (or OCP) when installed. Without them, export the STEP file
to STL from your CAD program (FreeCAD: select the body > File > Export > STL, fine tessellation)
and put it next to the STEP file with the same name: data/step/foo.step -> data/step/foo.stl.
OBJ and PLY work too.

Alignment: for screws the keypoints in the catalogue entry are given in the STEP file's own
coordinates. The mesh is moved so the axis tip -> head_top becomes +z and the bearing lands at the
origin; no axis guessing is needed. Other parts may give `axis: [x, y, z]` (default +z) and are
centred on the origin. `units: mm | cm | m | inch` rescales the file (default mm).
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from screwsynth.catalogue.schema import LENGTH_KINDS, Part
from screwsynth.common.paths import resolve

from .builders import BuildError, BuiltPart, directional_hull

log = logging.getLogger(__name__)

UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4, "in": 25.4}
MESH_SUFFIXES = (".stl", ".obj", ".ply")


def _convert_step(path: Path) -> trimesh.Trimesh | None:
    try:
        import cadquery as cq  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        shape = cq.importers.importStep(str(path))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "part.stl"
            cq.exporters.export(shape, str(out), tolerance=0.005, angularTolerance=0.1)
            return trimesh.load_mesh(out, process=True)
    except Exception as exc:  # OCCT raises plain ValueError/RuntimeError on unreadable files
        log.warning("cadquery could not convert %s (%s)", path.name, exc)
        return None


def load_source_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() in MESH_SUFFIXES and path.is_file():
        return trimesh.load_mesh(path, process=True)
    if path.suffix.lower() in (".step", ".stp"):
        if path.is_file():
            mesh = _convert_step(path)
            if mesh is not None:
                return mesh
        for suffix in MESH_SUFFIXES:
            alt = path.with_suffix(suffix)
            if alt.is_file():
                log.info("using %s (pre-converted mesh) for %s", alt.name, path.name)
                return trimesh.load_mesh(alt, process=True)
        raise BuildError(
            f"cannot convert {path}: cadquery is missing or failed to read it (pip install "
            f"cadquery), and there is no {path.with_suffix('.stl').name} next to it to fall back on"
        )
    raise BuildError(f"unsupported or missing CAD file: {path}")


def _rotation_to_z(axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return trimesh.geometry.align_vectors(axis, [0.0, 0.0, 1.0])


def _convex_pieces(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    try:
        import coacd  # type: ignore[import-not-found]

        parts = coacd.run_coacd(
            coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.05, max_convex_hull=6
        )
        pieces = [directional_hull(np.asarray(v), 48) for v, _ in parts]
        if pieces:
            return pieces
    except ImportError:
        log.info("coacd not installed; using a single convex hull as collision proxy")
    return [directional_hull(mesh.vertices)]


def build_step(part: Part, coll_n: int = 16) -> BuiltPart:
    entry = dict(part.step or {})
    if not entry.get("path"):
        raise BuildError(f"{part.id}: no STEP path in the catalogue")
    mesh = load_source_mesh(resolve(entry["path"]))
    scale = UNITS.get(str(entry.get("units", "mm")).lower())
    if scale is None:
        raise BuildError(f"{part.id}: unknown units {entry.get('units')!r}")
    mesh.apply_scale(scale)

    T = np.eye(4)
    if part.kind in LENGTH_KINDS:
        kp = {
            k: np.asarray(v, dtype=float) * scale for k, v in (entry.get("keypoints") or {}).items()
        }
        if set(kp) != {"head_top", "bearing", "tip"}:
            raise BuildError(f"{part.id}: STEP screws need keypoints head_top, bearing, tip")
        T = _rotation_to_z(kp["head_top"] - kp["tip"])
        T[:3, 3] = -(T[:3, :3] @ kp["bearing"])
        mesh.apply_transform(T)
        keypoints = {k: (T[:3, :3] @ p + T[:3, 3]).tolist() for k, p in kp.items()}
        labelled = True
    else:
        axis = np.asarray(entry.get("axis", [0, 0, 1]), dtype=float)
        mesh.apply_transform(_rotation_to_z(axis))
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        lo, hi = mesh.bounds
        keypoints = {"head_top": [0, 0, hi[2]], "bearing": [0, 0, lo[2]], "tip": [0, 0, 0]}
        labelled = False

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    radius = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    info = {"source_file": str(entry["path"]), "units": entry.get("units", "mm")}
    if part.kind in LENGTH_KINDS:
        info["axis_height_when_lying"] = {"head": radius, "tip": part.dims.get("d", 2 * radius) / 2}
        info["axis_check_deg"] = math.degrees(math.acos(min(1.0, abs(T[2, 2]))))
    return BuiltPart(
        mesh=mesh,
        slots={"body": (0, len(mesh.faces))},
        keypoints=keypoints,
        labelled=labelled,
        collision=_convex_pieces(mesh),
        info=info,
    )
