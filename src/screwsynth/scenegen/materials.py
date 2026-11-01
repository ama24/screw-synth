"""Physically based finish presets: sample concrete shader parameters (pure Python).

Presets live in configs/config.yaml (scenegen.materials) so they can be tuned without code changes.
A value given as [lo, hi] is sampled uniformly; a scalar is used as is.
"""

from __future__ import annotations

import colorsys
import random
from collections.abc import Mapping
from typing import Any

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "black_oxide": {
        "base_color": [0.035, 0.035, 0.04],
        "value_jitter": 0.3,
        "metallic": 0.85,
        "roughness": [0.35, 0.6],
        "anisotropic": [0.0, 0.3],
        "coat": [0.0, 0.3],
    },
    "zinc_clear": {
        "base_color": [0.78, 0.80, 0.84],
        "hue_jitter": 0.02,
        "value_jitter": 0.06,
        "metallic": 1.0,
        "roughness": [0.15, 0.35],
        "anisotropic": [0.0, 0.4],
    },
    "zinc_yellow": {
        "base_color": [0.86, 0.72, 0.36],
        "hue_jitter": 0.04,
        "value_jitter": 0.08,
        "metallic": 1.0,
        "roughness": [0.15, 0.35],
        "anisotropic": [0.0, 0.3],
        "iridescence": [0.0, 0.3],
    },
    "stainless_a2": {
        "base_color": [0.70, 0.70, 0.71],
        "value_jitter": 0.05,
        "metallic": 1.0,
        "roughness": [0.12, 0.3],
        "anisotropic": [0.2, 0.7],
    },
    "stainless_a4": {
        "base_color": [0.68, 0.69, 0.70],
        "value_jitter": 0.05,
        "metallic": 1.0,
        "roughness": [0.1, 0.28],
        "anisotropic": [0.2, 0.7],
    },
    "plain_steel": {
        "base_color": [0.56, 0.56, 0.57],
        "value_jitter": 0.08,
        "metallic": 1.0,
        "roughness": [0.25, 0.5],
        "anisotropic": [0.1, 0.5],
    },
    "brass": {
        "base_color": [0.91, 0.74, 0.42],
        "hue_jitter": 0.02,
        "value_jitter": 0.06,
        "metallic": 1.0,
        "roughness": [0.15, 0.35],
        "anisotropic": [0.0, 0.4],
    },
    "nylon_white": {
        "base_color": [0.86, 0.85, 0.80],
        "value_jitter": 0.05,
        "metallic": 0.0,
        "roughness": [0.35, 0.6],
        "subsurface": [0.05, 0.2],
    },
    "nylon_black": {
        "base_color": [0.03, 0.03, 0.03],
        "value_jitter": 0.3,
        "metallic": 0.0,
        "roughness": [0.3, 0.55],
    },
    # die-cast / machined aluminium (corner brackets, standoffs): satin, slightly brushed
    "aluminium": {
        "base_color": [0.91, 0.92, 0.93],
        "value_jitter": 0.05,
        "metallic": 1.0,
        "roughness": [0.3, 0.55],
        "anisotropic": [0.0, 0.3],
    },
    # black anodised aluminium: a dark oxide layer over the metal, satin sheen
    "aluminium_black": {
        "base_color": [0.03, 0.03, 0.035],
        "value_jitter": 0.3,
        "metallic": 0.4,
        "roughness": [0.3, 0.55],
        "coat": [0.0, 0.2],
    },
    # nyloc inserts: blue is the most common in practice, then white (natural nylon)
    "insert_blue": {
        "base_color": [0.08, 0.22, 0.62],
        "hue_jitter": 0.02,
        "value_jitter": 0.12,
        "metallic": 0.0,
        "roughness": [0.4, 0.6],
        "subsurface": [0.05, 0.2],
    },
    "insert_white": {
        "base_color": [0.9, 0.88, 0.8],
        "value_jitter": 0.05,
        "metallic": 0.0,
        "roughness": [0.4, 0.6],
        "subsurface": [0.1, 0.3],
    },
}

DEFAULT_WEAR = {"scratches": 0.3, "grime": 0.2, "fingerprints": 0.1}
DEFAULT_INSERT_COLOURS = {"insert_blue": 0.7, "insert_white": 0.3}


def pick_insert_finish(weights: Mapping[str, float] | None, rng: random.Random) -> str:
    """Weighted choice of the nyloc insert preset (scenegen.insert_colours)."""
    table = {k: float(v) for k, v in (weights or DEFAULT_INSERT_COLOURS).items() if float(v) > 0}
    if not table:
        raise ValueError("scenegen.insert_colours needs at least one positive weight")
    names = sorted(table)
    return rng.choices(names, weights=[table[n] for n in names])[0]


def _sample(v: Any, rng: random.Random) -> Any:
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return rng.uniform(float(v[0]), float(v[1]))
    return v


def sample_material(
    finish: str,
    presets: Mapping[str, Mapping[str, Any]] | None,
    rng: random.Random,
    wear: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Concrete parameters for one instance: base colour (jittered in HSV), metallic, roughness, ..."""
    table = {**DEFAULT_PRESETS, **{k: dict(v) for k, v in (presets or {}).items()}}
    if finish not in table:
        raise KeyError(f"no material preset for finish {finish!r}")
    p = table[finish]
    r, g, b = p["base_color"]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + rng.uniform(-1, 1) * float(p.get("hue_jitter", 0.0))) % 1.0
    v = min(1.0, max(0.0, v * (1 + rng.uniform(-1, 1) * float(p.get("value_jitter", 0.0)))))
    out = {
        "finish": finish,
        "base_color": [round(c, 5) for c in colorsys.hsv_to_rgb(h, s, v)],
        "metallic": float(_sample(p.get("metallic", 1.0), rng)),
        "roughness": float(_sample(p.get("roughness", 0.3), rng)),
        "anisotropic": float(_sample(p.get("anisotropic", 0.0), rng)),
        "anisotropic_rotation": rng.random(),
        "coat": float(_sample(p.get("coat", 0.0), rng)),
        "subsurface": float(_sample(p.get("subsurface", 0.0), rng)),
        "iridescence": float(_sample(p.get("iridescence", 0.0), rng)),
    }
    w = {**DEFAULT_WEAR, **(wear or {})}
    out["scratches"] = rng.random() < w["scratches"]
    out["grime"] = rng.random() < w["grime"]
    out["fingerprints"] = rng.random() < w["fingerprints"]
    out["noise_seed"] = rng.randint(0, 10_000)
    return out
