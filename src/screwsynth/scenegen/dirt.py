"""Dirt: dark oil and dust particles, on the parts and on the plate (pure Python sampling).

Configured in scenegen.dirt (GUI: Dirt tab). Parts and plate are switched on and off separately:
- parts: each part is dirty with probability `share`; its oil coverage and dust density are drawn
  from the ranges once, so a part keeps its dirt in every photo of its layout;
- plate: each layout has a dirty plate with probability `share`, the same for all of its photos
  (lighting variants and picker removals), like a real plate between two cleanings.

Values: `oil` and `dust` are the covered share of the surface (0..1), `dust_size_mm` the size of
one dust speck. A value is one number or a range [min, max]. The shader side is in blender_scene.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

DEFAULTS: dict[str, dict[str, Any]] = {
    "parts": {
        "enabled": False,
        "share": 0.3,
        "oil": [0.05, 0.35],
        "dust": [0.05, 0.4],
        "dust_size_mm": [0.05, 0.2],
    },
    "plate": {
        "enabled": False,
        "share": 0.5,
        "oil": [0.02, 0.25],
        "dust": [0.05, 0.4],
        "dust_size_mm": [0.1, 0.4],
    },
}
CLEAN: dict[str, Any] = {"dirty": False, "oil": 0.0, "dust": 0.0, "dust_size_mm": 0.0, "seed": 0}


def _value(v: Any, rng: random.Random, lo: float = 0.0, hi: float = 1.0e9) -> float:
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"expected one number or a range [min, max], got {v!r}")
        a, b = sorted(float(x) for x in v)
        x = rng.uniform(a, b)
    else:
        x = float(v)
    return min(hi, max(lo, x))


def settings(section: Mapping[str, Any] | None, target: str) -> dict[str, Any]:
    """scenegen.dirt.<target> merged over the defaults (target: parts | plate)."""
    if target not in DEFAULTS:
        raise ValueError(f"dirt target must be one of {sorted(DEFAULTS)}, got {target!r}")
    given = dict((section or {}).get(target) or {})
    unknown = sorted(set(given) - set(DEFAULTS[target]))
    if unknown:
        raise ValueError(f"scenegen.dirt.{target}: unknown key(s) {unknown}")
    return {**DEFAULTS[target], **given}


def sample(section: Mapping[str, Any] | None, target: str, rng: random.Random) -> dict[str, Any]:
    """Dirt of one part or one plate: {dirty, oil, dust, dust_size_mm, seed}.

    The rng is always advanced by the same amount, so switching dirt on or off does not change
    any other random choice of the scene (same layouts, same parts, same lights).
    """
    st = settings(section, target)
    draws = [rng.random() for _ in range(5)]
    seed = rng.randint(0, 10_000)
    if not st["enabled"] or draws[0] >= float(st["share"]):
        return dict(CLEAN)
    sub = random.Random(int(draws[1] * 2**31) ^ seed)
    return {
        "dirty": True,
        "oil": round(_value(st["oil"], sub, 0.0, 1.0), 4),
        "dust": round(_value(st["dust"], sub, 0.0, 1.0), 4),
        "dust_size_mm": round(_value(st["dust_size_mm"], sub, 0.01, 5.0), 4),
        "seed": seed,
    }


def oil_threshold(coverage: float) -> float:
    """Noise level above which a point is oily, for a covered share of the surface.

    Blender's noise texture (Fac) clusters around 0.5 with a spread of roughly +-0.2: coverage 0
    needs a threshold above almost every value, coverage 1 one below almost every value.
    """
    c = min(1.0, max(0.0, coverage))
    return 0.75 - 0.5 * c
