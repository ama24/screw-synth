"""One builder per catalogue shape. Each returns a watertight render mesh, its material slots,
keypoints and convex collision pieces, all in the part's local frame (mm).

Local frame: screws have their axis along +z with the bearing face (underside of the head) at
z = 0, head towards +z and tip towards -z. For countersunk screws z = 0 is where the cone meets
the shank and head_top is the top face at z = k. Headless parts (set screws, threaded rod) and
standoffs run from z = -L (tip) to z = 0 (head_top = bearing); standoff studs come on top of that.
Nuts, washers, T-nuts and brackets are centred on the origin (bores along z).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, Mesh

from screwsynth.catalogue.schema import LENGTH_KINDS, Part

from .profiles import (
    CREST,
    EXT_DEPTH,
    EXT_ROOT,
    INT_DEPTH,
    INT_ROOT,
    TAN30,
    RodSpec,
    arc,
    ccw,
    hexagon,
    regular_polygon,
    rod_mesh,
    rod_triangles,
    rounded_rect_rz,
)

SLOTS = ("body", "insert")


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class Lod:
    thread_segments: int
    axial_per_pitch: int
    round_segments: int


@dataclass
class BuiltPart:
    mesh: trimesh.Trimesh  # faces sorted by material slot
    slots: dict[str, tuple[int, int]]  # slot -> [first_face, end_face)
    keypoints: dict[str, list[float]]
    labelled: bool  # keypoints are meaningful (screws) or only placeholders
    collision: list[trimesh.Trimesh]
    info: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- manifold helpers


def _revolve(points_rz: np.ndarray | list, segments: int) -> Manifold:
    return CrossSection([ccw(points_rz)]).revolve(segments)


def _prism(points_xy: np.ndarray, z0: float, z1: float) -> Manifold:
    return CrossSection([ccw(points_xy)]).extrude(z1 - z0).translate([0, 0, z0])


def _from_arrays(vertices: np.ndarray, faces: np.ndarray) -> Manifold:
    m = Manifold(
        Mesh(
            vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
        )
    )
    if m.is_empty():
        raise BuildError(f"invalid input mesh for manifold: {m.status()}")
    return m


def _rod(spec: RodSpec, lod: Lod) -> Manifold:
    return _from_arrays(*rod_mesh(spec, lod.thread_segments, lod.axial_per_pitch))


def _to_trimesh(
    m: Manifold, slot_ids: dict[str, int] | None = None
) -> tuple[trimesh.Trimesh, dict]:
    """Convert to trimesh; faces sorted by slot using manifold's per-run original ids."""
    if m.is_empty():
        raise BuildError(f"boolean result is empty: {m.status()}")
    mesh = m.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    faces = np.asarray(mesh.tri_verts).astype(np.int64)
    slot_of_face = np.zeros(len(faces), dtype=np.int64)
    if slot_ids:
        run_index = np.asarray(mesh.run_index)
        run_orig = np.asarray(mesh.run_original_id)
        id_to_slot = {oid: SLOTS.index(name) for name, oid in slot_ids.items()}
        for k, oid in enumerate(run_orig):
            t0, t1 = run_index[k] // 3, run_index[k + 1] // 3
            slot_of_face[t0:t1] = id_to_slot.get(int(oid), 0)
    order = np.argsort(slot_of_face, kind="stable")
    faces, slot_of_face = faces[order], slot_of_face[order]
    slots = {}
    for idx, name in enumerate(SLOTS):
        hit = np.nonzero(slot_of_face == idx)[0]
        if len(hit):
            slots[name] = (int(hit[0]), int(hit[-1]) + 1)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False), slots


def _hull(points: np.ndarray) -> trimesh.Trimesh:
    return trimesh.convex.convex_hull(np.asarray(points, dtype=float))


def _cyl_points(r: float, z0: float, z1: float, n: int) -> np.ndarray:
    ring = regular_polygon(r, n)
    return np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]])


def _revolved_points(points_rz: list[tuple[float, float]], n: int) -> np.ndarray:
    pts = []
    for r, z in points_rz:
        ring = regular_polygon(r, n) if r > 0 else np.zeros((1, 2))
        pts.append(np.c_[ring, np.full(len(ring), z)])
    return np.vstack(pts)


def directional_hull(vertices: np.ndarray, n_dirs: int = 96) -> trimesh.Trimesh:
    """Low-poly convex hull: the extreme vertex in each of n_dirs directions."""
    k = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * k / n_dirs)
    theta = math.pi * (1 + 5**0.5) * k
    dirs = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    idx = np.unique(np.argmax(vertices @ dirs.T, axis=0))
    return _hull(vertices[idx])


# ------------------------------------------------------------------------------------- screws


def _thread_lod(part: Part, lod: Lod, max_triangles: int, rod_length: float) -> Lod:
    """Lower the axial thread resolution if the rod alone would exceed ~85% of the budget."""
    pitch = part.pitch or 1.0
    k = lod.axial_per_pitch
    while k > 4 and rod_triangles(rod_length, pitch, lod.thread_segments, k) > 0.85 * max_triangles:
        k -= 1
    seg = lod.thread_segments
    while seg > 16 and rod_triangles(rod_length, pitch, seg, k) > 0.85 * max_triangles:
        seg -= 4
    return Lod(seg, k, lod.round_segments)


def _screw_rod(part: Part, z_tip: float, z_top: float, thread_top: float, lod: Lod) -> Manifold:
    d = part.dims
    p = part.pitch or 0.0
    R = d["d"] / 2
    depth = EXT_DEPTH * p
    if part.fully_threaded is False:
        thread_end, runout = z_tip + d["b"], 2.0 * p
    else:
        thread_end, runout = thread_top - 0.5 * p, 1.5 * p
    spec = RodSpec(
        radius=R,
        pitch=p,
        z0=z_tip,
        z1=z_top,
        depth=depth,
        crest=CREST,
        root=EXT_ROOT,
        thread_end=thread_end,
        runout=runout,
        tip_radius=R - 1.05 * depth,  # chamfer from just below the thread root, 45 deg
        tip_slope=1.0,
    )
    return _rod(spec, lod)


def _socket(s: float, depth: float, z_top: float) -> Manifold:
    """Hexagon socket (across flats s) with a 118 deg drill-point bottom, open at z_top."""
    hexa = hexagon(s)
    z_bottom = z_top - depth
    apex = z_bottom - (s / 2) / math.tan(math.radians(59))
    pts = np.vstack(
        [np.c_[hexa, np.full(6, z_bottom)], np.c_[hexa, np.full(6, z_top + 1.0)], [[0, 0, apex]]]
    )
    return Manifold.hull_points(pts.astype(float))


def _screw_common(
    part: Part, head: Manifold, z_tip: float, lod: Lod, rod_top: float, socket: Manifold | None
) -> Manifold:
    body = head + _screw_rod(part, z_tip, rod_top, 0.0, lod)
    if socket is not None:
        body = body - socket
    return body


def _screw_result(
    part: Part,
    body: Manifold,
    z_head_top: float,
    z_tip: float,
    head_proxy: trimesh.Trimesh,
    coll_n: int,
    head_contact_r: float,
) -> BuiltPart:
    mesh, slots = _to_trimesh(body)
    R = part.dims["d"] / 2
    shank = _hull(_cyl_points(R, z_tip, min(0.0, z_head_top) + 0.05 * R, coll_n))
    return BuiltPart(
        mesh=mesh,
        slots=slots,
        keypoints={
            "head_top": [0.0, 0.0, z_head_top],
            "bearing": [0.0, 0.0, 0.0],
            "tip": [0.0, 0.0, z_tip],
        },
        labelled=True,
        collision=[head_proxy, shank],
        info={"axis_height_when_lying": {"head": head_contact_r, "tip": R}},
    )


def build_socket_cap(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    L, k, dk = part.length or 0.0, d["k"], d["dk"]
    ct, cb = 0.08 * k, 0.04 * k
    head = _revolve(
        [(0, 0), (dk / 2 - cb, 0), (dk / 2, cb), (dk / 2, k - ct), (dk / 2 - ct, k), (0, k)],
        lod.round_segments,
    )
    body = _screw_common(part, head, -L, lod, 0.4 * k, _socket(d["s"], d["t"], k))
    proxy = _hull(_cyl_points(dk / 2, 0.0, k, coll_n))
    return _screw_result(part, body, k, -L, proxy, coll_n, dk / 2)


def build_button(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    L, k, dk = part.length or 0.0, d["k"], d["dk"]
    e0 = 0.12 * k
    n = max(8, lod.round_segments // 4)
    # dome from the edge (height e0) to a flat top ring just outside the socket corners
    r_flat = 1.15 * d["s"] / 2 / math.cos(math.radians(30))
    dome = [
        (r_flat + (dk / 2 - r_flat) * math.cos(a), e0 + (k - e0) * math.sin(a))
        for a in np.linspace(0, math.pi / 2, n)
    ]
    head = _revolve([(0, 0), (dk / 2, 0), *dome, (0, k)], lod.round_segments)
    body = _screw_common(part, head, -L, lod, 0.4 * k, _socket(d["s"], d["t"], k))
    prof = [(dk / 2, 0.0)] + [dome[i] for i in range(0, n, max(1, n // 4))] + [dome[-1], (0.0, k)]
    proxy = _hull(_revolved_points(prof, coll_n))
    return _screw_result(part, body, k, -L, proxy, coll_n, dk / 2)


def build_countersunk(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    L, k, dk, R = part.length or 0.0, d["k"], d["dk"], d["d"] / 2
    k_cone = min((dk / 2 - R) / math.tan(math.radians(d["angle"] / 2)), 0.97 * k)
    prof = [(0, 0), (R, 0), (dk / 2, k_cone), (dk / 2, k), (0, k)]
    head = _revolve(prof, lod.round_segments)
    z_tip = k - L  # nominal length includes the head
    body = _screw_common(part, head, z_tip, lod, 0.4 * k_cone, _socket(d["s"], d["t"], k))
    proxy = _hull(_revolved_points(prof[1:4], coll_n))
    return _screw_result(part, body, k, z_tip, proxy, coll_n, dk / 2)


def build_hex(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    L, k, s = part.length or 0.0, d["k"], d["s"]
    c = d.get("c", 0.0)
    dw = d.get("dw", 0.0)
    rc = s / 2 / math.cos(math.radians(30))
    r_ch = 0.95 * s / 2  # top chamfer starts just inside the flats (30 deg)
    chamfer = _revolve(
        [(0, c), (rc + 0.5, c), (rc + 0.5, k - (rc + 0.5 - r_ch) * TAN30), (r_ch, k), (0, k)],
        lod.round_segments,
    )
    head = _prism(hexagon(s), c, k) ^ chamfer
    if c > 0 and dw > 0:
        head = head + Manifold.cylinder(c + 0.01, dw / 2, circular_segments=lod.round_segments)
    body = _screw_common(part, head, -L, lod, 0.4 * k, None)
    proxy = _hull(np.vstack([np.c_[hexagon(s), np.zeros(6)], np.c_[hexagon(s), np.full(6, k)]]))
    return _screw_result(part, body, k, -L, proxy, coll_n, s / 2)


# ------------------------------------------------------------------ headless (set screw, rod)


def _end_chamfer(R: float, r_end: float, z_end: float, z_far: float, segs: int) -> Manifold:
    """Solid to intersect with: a 45 deg chamfer from r_end at z_end, open towards z_far < z_end."""
    rise = R + 1.0 - r_end
    return _revolve(
        [(0, z_far), (R + 1.0, z_far), (R + 1.0, z_end - rise), (r_end, z_end), (0, z_end)], segs
    )


def rod_end_styles(part_id: str) -> tuple[str, str]:
    """(tip end, head_top end) of a cut rod piece: "chamfer" (factory end, 45 deg) or "cut" (sawn
    off flat, thread profile up to the edge). Drawn from the part id, so a part always looks the
    same; across lengths and sizes all four combinations occur."""
    h = hashlib.sha256(part_id.encode()).digest()
    return tuple("chamfer" if b % 2 == 0 else "cut" for b in h[:2])  # type: ignore[return-value]


def _headless(
    part: Part, lod: Lod, coll_n: int, socket: bool, ends: tuple[str, str] = ("chamfer", "chamfer")
) -> BuiltPart:
    """Fully threaded rod from z = -L (tip) to z = 0 (head_top); each end chamfered or cut flat.

    Set screws (DIN 913) have a hexagon socket at z = 0 and both ends chamfered; threaded rod has no
    socket and its ends vary (rod_end_styles). With no head the nominal length is the overall
    length, so head_top and bearing are the same point.
    """
    d = part.dims
    L, p = part.length or 0.0, part.pitch or 0.0
    R = d["d"] / 2
    depth = EXT_DEPTH * p
    r_end = R - 1.05 * depth  # chamfer from just below the thread root, as on the screw tips
    tip, top = ends
    spec = RodSpec(
        radius=R,
        pitch=p,
        z0=-L,
        z1=0.3 if top == "chamfer" else 0.0,
        depth=depth,
        crest=CREST,
        root=EXT_ROOT,
        tip_radius=r_end if tip == "chamfer" else None,
        tip_slope=1.0,
    )
    body = _rod(spec, lod)
    if top == "chamfer":
        body = body ^ _end_chamfer(R, r_end, 0.0, -L - 1.0, lod.round_segments)
    if socket:
        body = body - _socket(d["s"], d["t"], 0.0)
    mesh, slots = _to_trimesh(body)
    return BuiltPart(
        mesh=mesh,
        slots=slots,
        keypoints={"head_top": [0.0, 0.0, 0.0], "bearing": [0.0, 0.0, 0.0], "tip": [0.0, 0.0, -L]},
        labelled=True,
        collision=[_hull(_cyl_points(R, -L, 0.0, coll_n))],
        info={"axis_height_when_lying": {"head": R, "tip": R}, "end_styles": list(ends)},
    )


def build_set_screw(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    return _headless(part, lod, coll_n, socket=True)


def build_threaded_rod(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    return _headless(part, lod, coll_n, socket=False, ends=rod_end_styles(part.id))


# ------------------------------------------------------------------------------- hex standoffs


def _stud(part: Part, lod: Lod, z_body: float, up: bool) -> Manifold:
    """Male thread of length ls leaving the body face at z_body, upwards or downwards."""
    d = part.dims
    p = part.pitch or 0.0
    R, ls = d["d"] / 2, d["ls"]
    depth = EXT_DEPTH * p
    spec = RodSpec(
        radius=R,
        pitch=p,
        z0=-ls,
        z1=0.3,
        depth=depth,
        crest=CREST,
        root=EXT_ROOT,
        thread_end=-0.5 * p,
        runout=p,
        tip_radius=R - 1.05 * depth,
        tip_slope=1.0,
    )
    stud = _rod(spec, lod)  # from -ls (chamfered end) to the body face at 0
    if up:
        stud = stud.rotate([180, 0, 0])  # z -> -z: from the body face up to +ls
    return stud.translate([0, 0, z_body])


def _standoff(part: Part, lod: Lod, coll_n: int, top: str, bottom: str) -> BuiltPart:
    """Hex body from z = -L to 0 (the nominal length); each end is "male" (stud) or "female"."""
    d = part.dims
    L, s, segs = part.length or 0.0, d["s"], lod.round_segments
    R = d["d"] / 2
    body = _prism(hexagon(s), -L, 0.0) ^ _double_chamfer(s, -L, 0.0, segs)
    pieces = [_hull(np.vstack([np.c_[hexagon(s), np.full(6, -L)], np.c_[hexagon(s), np.zeros(6)]]))]
    through = d.get("through", 0) > 0 or 2 * d["lf"] >= L - 0.6  # short ones are always through
    if top == "female" and bottom == "female" and through:
        body = body - _tap(part, -L - 0.5, 0.5, lod)
    else:
        lf = min(d.get("lf", 0.0), L - 0.5)  # leave a wall when a stud is on the other side
        if top == "female":
            body = body - _tap(part, -lf, 0.5, lod)
        if bottom == "female":
            body = body - _tap(part, -L - 0.5, -L + lf, lod)
    for end, z, up in ((top, 0.0, True), (bottom, -L, False)):
        if end == "male":
            body = body + _stud(part, lod, z, up)
            z0, z1 = (z, z + d["ls"]) if up else (z - d["ls"], z)
            pieces.append(_hull(_cyl_points(R, z0, z1, coll_n)))
    mesh, slots = _to_trimesh(body)
    return BuiltPart(
        mesh=mesh,
        slots=slots,
        keypoints={"head_top": [0.0, 0.0, 0.0], "bearing": [0.0, 0.0, 0.0], "tip": [0.0, 0.0, -L]},
        labelled=True,
        collision=pieces,
        info={"axis_height_when_lying": {"head": s / 2, "tip": s / 2}},
    )


def build_standoff_mf(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    return _standoff(part, lod, coll_n, top="female", bottom="male")


def build_standoff_ff(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    return _standoff(part, lod, coll_n, top="female", bottom="female")


def build_standoff_mm(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    return _standoff(part, lod, coll_n, top="male", bottom="male")


# --------------------------------------------------------------------------------- nuts etc.


def _tap(part: Part, z0: float, z1: float, lod: Lod) -> Manifold:
    """Internal thread void (to subtract) from z0 to z1."""
    p = part.pitch or 0.0
    spec = RodSpec(
        radius=part.dims["d"] / 2 + 0.02 * p,
        pitch=p,
        z0=z0,
        z1=z1,
        depth=INT_DEPTH * p,
        crest=CREST,
        root=INT_ROOT,
    )
    return _rod(spec, lod)


def _bore_countersinks(R: float, z_bottom: float, z_top: float, segs: int) -> Manifold:
    h = 0.15 * R
    r_face = 1.12 * R
    low = Manifold.cylinder(h + 0.01, r_face, r_face - h - 0.01, segs).translate(
        [0, 0, z_bottom - 0.01]
    )
    high = Manifold.cylinder(h + 0.01, r_face - h - 0.01, r_face, segs).translate([0, 0, z_top - h])
    return low + high


def _double_chamfer(s: float, z0: float, z1: float, segs: int, top: bool = True) -> Manifold:
    rc = s / 2 / math.cos(math.radians(30)) + 0.5
    r_ch = 0.95 * s / 2
    dz = (rc - r_ch) * TAN30
    prof = [(0, z0), (r_ch, z0), (rc, z0 + dz)]
    prof += [(rc, z1 - dz), (r_ch, z1)] if top else [(rc, z1)]
    prof += [(0, z1)]
    return _revolve(prof, segs)


def _centered(m: Manifold) -> tuple[Manifold, np.ndarray]:
    lo, hi = np.array(m.bounding_box()[:3]), np.array(m.bounding_box()[3:])
    shift = -(lo + hi) / 2
    return m.translate(shift.tolist()), shift


def _placeholder_keypoints(mesh: trimesh.Trimesh) -> dict[str, list[float]]:
    lo, hi = mesh.bounds
    cx, cy = (lo[:2] + hi[:2]) / 2
    return {
        "head_top": [float(cx), float(cy), float(hi[2])],
        "bearing": [float(cx), float(cy), float(lo[2])],
        "tip": [float(cx), float(cy), float((lo[2] + hi[2]) / 2)],
    }


def _simple_result(
    body: Manifold, collision: list[trimesh.Trimesh], slot_ids=None, shift=None
) -> BuiltPart:
    mesh, slots = _to_trimesh(body, slot_ids)
    if shift is not None:
        collision = [c.copy().apply_translation(shift) for c in collision]
    return BuiltPart(
        mesh=mesh,
        slots=slots,
        keypoints=_placeholder_keypoints(mesh),
        labelled=False,
        collision=collision,
    )


def build_hex_nut(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    s, m, R = d["s"], d["m"], d["d"] / 2
    body = _prism(hexagon(s), 0, m) ^ _double_chamfer(s, 0, m, lod.round_segments)
    body = body - _tap(part, -0.5, m + 0.5, lod) - _bore_countersinks(R, 0, m, lod.round_segments)
    body, shift = _centered(body)
    proxy = _hull(np.vstack([np.c_[hexagon(s), np.zeros(6)], np.c_[hexagon(s), np.full(6, m)]]))
    return _simple_result(body, [proxy], shift=shift)


def build_nyloc_nut(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    s, m, h, R, p = d["s"], d["m"], d["h"], d["d"] / 2, part.pitch or 0.0
    segs = lod.round_segments
    collar_h = h - m
    r_col = 0.95 * s / 2
    r_rec = 0.78 * s / 2
    r_in = 0.95 * (R - INT_DEPTH * p)  # nylon bore, smaller than the thread minor diameter
    z_rec = m + 0.15 * collar_h
    rr = 0.2 * collar_h
    hexpart = _prism(hexagon(s), 0, m) ^ _double_chamfer(s, 0, m + 0.01, segs, top=False)
    collar = _revolve(
        [(0, m - 0.01), (r_col, m - 0.01), *arc((r_col - rr, h - rr), rr, 0, 90, 4), (0, h)], segs
    )
    metal = hexpart + collar
    metal = metal - Manifold.cylinder(h, r_rec, circular_segments=segs).translate([0, 0, z_rec])
    metal = metal - _tap(part, -0.5, z_rec + 0.01, lod) - _bore_countersinks(R, 0, h + 10, segs)
    insert_top = h - 0.1 * collar_h
    insert = _revolve(
        [(r_in, z_rec), (r_rec + 0.01, z_rec), (r_rec + 0.01, insert_top), (r_in, insert_top)], segs
    ).as_original()
    metal = metal.as_original()
    body = metal + insert
    body, shift = _centered(body)
    ring = regular_polygon(r_col, coll_n)
    proxy = _hull(
        np.vstack(
            [
                np.c_[hexagon(s), np.zeros(6)],
                np.c_[hexagon(s), np.full(6, m)],
                np.c_[ring, np.full(coll_n, h)],
            ]
        )
    )
    return _simple_result(
        body,
        [proxy],
        slot_ids={"body": metal.original_id(), "insert": insert.original_id()},
        shift=shift,
    )


def build_washer(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    d1, d2, h = d["d1"], d["d2"], d["h"]
    prof = rounded_rect_rz(d1 / 2, d2 / 2, -h / 2, h / 2, 0.2 * h, 3)
    body = _revolve(prof, lod.round_segments)
    proxy = _hull(_cyl_points(d2 / 2, -h / 2, h / 2, max(coll_n, 16)))
    return _simple_result(body, [proxy])


def _extrude_along_x(points_yz: list, length: float) -> Manifold:
    """Extrude a (y, z) cross-section along x (the slot direction), centred on x = 0."""
    return (
        CrossSection([ccw(points_yz)])
        .extrude(length)
        .rotate([90, 0, 90])
        .translate([-length / 2, 0, 0])
    )


def build_tnut_slide(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    L, W, H = d["length"], d["width"], d["height"]
    nw, nh = d["neck_width"], d["neck_height"]
    wing = H - nh
    prof = [
        (-W / 2, 0),
        (W / 2, 0),
        (W / 2, 0.6 * wing),
        (nw / 2, wing),
        (nw / 2, H),
        (-nw / 2, H),
        (-nw / 2, wing),
        (-W / 2, 0.6 * wing),
    ]
    body = _extrude_along_x(prof, L)
    body = (
        body
        - _tap(part, -0.5, H + 0.5, lod)
        - _bore_countersinks(d["d"] / 2, 0, H, lod.round_segments)
    )
    body, shift = _centered(body)
    mesh_pts = np.array(body.to_mesh().vert_properties)[:, :3]
    return _simple_result(body, [directional_hull(mesh_pts)])


def build_tnut_drop(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    d = part.dims
    L, W, H = d["length"], d["width"], d["height"]
    nw, nh = d["neck_width"], d["neck_height"]
    n = max(6, lod.round_segments // 8)
    # plan view: two opposite corners rounded so the nut can only turn one way in the slot
    rp = 0.9 * W
    plan = [(-L / 2, W / 2)]
    plan += arc((L / 2 - rp, W / 2 - rp), rp, 90, 0, n)[:-1] + [
        (L / 2, W / 2 - rp),
        (L / 2, -W / 2),
    ]
    plan += arc((-L / 2 + rp, -W / 2 + rp), rp, 270, 180, n)[:-1] + [(-L / 2, -W / 2 + rp)]
    block = _prism(np.array(plan), 0, H)
    # cross-section across the slot: rounded bottom, neck on top
    rb = min(0.45 * W, H - nh)
    yz = arc((-W / 2 + rb, rb), rb, 180, 270, n) + arc((W / 2 - rb, rb), rb, 270, 360, n)
    yz += [(W / 2, H - nh), (nw / 2, H - nh + 0.3 * nh), (nw / 2, H), (-nw / 2, H)]
    yz += [(-nw / 2, H - nh + 0.3 * nh), (-W / 2, H - nh)]
    body = block ^ _extrude_along_x(yz, L + 2)
    body = (
        body
        - _tap(part, -0.5, H + 0.5, lod)
        - _bore_countersinks(d["d"] / 2, 0, H, lod.round_segments)
    )
    body, shift = _centered(body)
    mesh_pts = np.array(body.to_mesh().vert_properties)[:, :3]
    return _simple_result(body, [directional_hull(mesh_pts)])


def _stadium(cx: float, r: float, half_len: float, n: int) -> np.ndarray:
    """Slot outline in the xy plane: two half circles of radius r, centres cx -/+ half_len."""
    right = arc((cx + half_len, 0.0), r, -90, 90, n)
    left = arc((cx - half_len, 0.0), r, 90, 270, n)
    return np.array(right + left)


def build_corner_bracket(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Die-cast extrusion corner bracket: two legs at 90 deg, triangular side walls, a slotted
    hole and an alignment key (sits in the profile slot) under each leg.

    Built with the corner at the origin, legs along +x (base) and +z (upright); the bracket is
    symmetric under swapping x and z, so the upright is the base mirrored in the plane x = z.
    """
    d = part.dims
    a, w, t, ts = d["a"], d["w"], d["t"], d["ts"]
    kw, kh = d["key_w"], d["key_h"]
    n = max(6, lod.round_segments // 4)

    def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Manifold:
        return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

    swap = [1.0, 0.0, -1.0]  # mirror plane normal: x <-> z
    leg = box(0, a, -w / 2, w / 2, 0, t) + box(t + 0.5, a - 0.5, -kw / 2, kw / 2, -kh, 0.01)
    xc = t + (a - t) / 2
    slot = _prism(_stadium(xc, d["dh"] / 2, d["slot"] / 2, n), -kh - 1, t + 1)
    tri = CrossSection([ccw([(0, 0), (a, 0), (0, a)])]).extrude(ts).rotate([90, 0, 0])
    sides = tri.translate([0, w / 2, 0]) + tri.translate([0, -w / 2 + ts, 0])
    body = leg + leg.mirror(swap) + sides - slot - slot.mirror(swap)
    body, shift = _centered(body)

    def box_hull(*b: float) -> trimesh.Trimesh:
        x0, x1, y0, y1, z0, z1 = b
        return _hull(np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]))

    base = box_hull(0, a, -w / 2, w / 2, -kh, t)
    upright = box_hull(-kh, t, -w / 2, w / 2, 0, a)
    walls = [
        _hull(np.array([[x, y, z] for x, z in ((0, 0), (a, 0), (0, a)) for y in (y0, y0 + ts)]))
        for y0 in (w / 2 - ts, -w / 2)
    ]
    return _simple_result(body, [base, upright, *walls], shift=shift)


BUILDERS: dict[str, Callable[[Part, Lod, int], BuiltPart]] = {
    "socket_cap": build_socket_cap,
    "hex": build_hex,
    "button": build_button,
    "countersunk": build_countersunk,
    "set_screw": build_set_screw,
    "threaded_rod": build_threaded_rod,
    "standoff_mf": build_standoff_mf,
    "standoff_ff": build_standoff_ff,
    "standoff_mm": build_standoff_mm,
    "corner_bracket": build_corner_bracket,
    "hex_nut": build_hex_nut,
    "nyloc_nut": build_nyloc_nut,
    "washer": build_washer,
    "tnut_slide": build_tnut_slide,
    "tnut_drop": build_tnut_drop,
}


def build(part: Part, lod: Lod, coll_n: int = 16, max_triangles: int = 200_000) -> BuiltPart:
    """Build the render mesh, slots, keypoints and collision pieces of a parametric part."""
    if part.source == "step":
        from .step import build_step

        return build_step(part, coll_n)
    if part.shape.startswith("unknown_"):
        from .unknown import UNKNOWN_BUILDERS  # imports this module, so only here

        if part.shape not in UNKNOWN_BUILDERS:
            raise BuildError(f"no builder for shape {part.shape!r}")
        return UNKNOWN_BUILDERS[part.shape](part, lod, coll_n)
    if part.shape not in BUILDERS:
        raise BuildError(f"no builder for shape {part.shape!r}")
    if part.kind in LENGTH_KINDS:
        d = part.dims
        threaded = (part.length or 0) + d.get("k", 0) + 2 * d.get("ls", 0) + 2 * d.get("lf", 0)
        lod = _thread_lod(part, lod, max_triangles, threaded)
    return BUILDERS[part.shape](part, lod, coll_n)
