"""Part-id and class-name builders and parsers.

Part ids (see CLAUDE.md):
  screws        <FAMILY>_<SIZE>x<LENGTH>        DIN912_M4x16, DIN913_M4x8, DIN975_M4x40
  standoffs     <FAMILY>_<SIZE>x<LENGTH>        STANDOFF_MF_M3x10
  nuts/washers  <FAMILY>_<SIZE>                 DIN934_M4, DIN125A_M4
  T-nuts        TNUT_<SLIDE|DROP>_S<slot>_<SIZE> TNUT_DROP_S6_M4
  profile parts <FAMILY>_<PROFILE>[_<STYLE>]    BRACKET_2020, BRACKET_2020_CAST (extrusion size,
                                                4 digits; styles are look-alike variants of one class)
  unknown       UNKNOWN_<TYPE>_<NN>             UNKNOWN_WOODSCREW_03
Family names are upper-case letters and digits, optionally in "_"-joined words (STANDOFF_MF).
Class name = part id without the length; every unknown part has the class UNKNOWN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEYPOINTS: tuple[str, str, str] = ("head_top", "bearing", "tip")
TNUT_FAMILY = "TNUT"
TNUT_KINDS: tuple[str, str] = ("SLIDE", "DROP")
UNKNOWN = "UNKNOWN"  # family and class of the non-catalogue parts the machine rejects

_NUM = r"\d+(?:\.\d+)?"
_SIZE = rf"M{_NUM}"
_FAMILY = r"[A-Z][A-Z0-9]*(?:_[A-Z][A-Z0-9]*)*"
_PROFILE = r"\d{4}"  # aluminium extrusion size: 2020, 3030, 4040
_SCREW_RE = re.compile(rf"^(?P<family>{_FAMILY})_(?P<size>{_SIZE})x(?P<length>{_NUM})$")
_TNUT_RE = re.compile(
    rf"^{TNUT_FAMILY}_(?P<kind>{'|'.join(TNUT_KINDS)})_S(?P<slot>\d+)_(?P<size>{_SIZE})$"
)
_PLAIN_RE = re.compile(rf"^(?P<family>{_FAMILY})_(?P<size>{_SIZE})$")
_PROFILE_RE = re.compile(
    rf"^(?P<family>{_FAMILY})_(?P<profile>{_PROFILE})(?:_(?P<style>[A-Z][A-Z0-9]*))?$"
)
_UNKNOWN_RE = re.compile(rf"^{UNKNOWN}_(?P<variant>[A-Z][A-Z0-9]*_\d+)$")


@dataclass(frozen=True)
class PartName:
    """Parsed part id."""

    family: str
    size: str
    length: float | None = None
    tnut_kind: str | None = None
    slot: int | None = None
    variant: str | None = None  # unknown parts: <TYPE>_<NN>
    profile: str | None = None  # profile parts: extrusion size, e.g. "2020" (also the size)
    style: str | None = None  # profile parts: variant of the same class, e.g. "CAST"

    @property
    def is_screw(self) -> bool:
        return self.length is not None

    @property
    def is_tnut(self) -> bool:
        return self.tnut_kind is not None

    @property
    def is_unknown(self) -> bool:
        return self.variant is not None

    @property
    def part_id(self) -> str:
        if self.variant is not None:
            return f"{UNKNOWN}_{self.variant}"
        if self.tnut_kind is not None:
            assert self.slot is not None
            return tnut_id(self.tnut_kind, self.slot, self.size)
        if self.profile is not None:
            return profile_id(self.family, self.profile, self.style)
        if self.length is not None:
            return screw_id(self.family, self.size, self.length)
        return plain_id(self.family, self.size)

    @property
    def class_name(self) -> str:
        if self.variant is not None:
            return UNKNOWN
        if self.profile is not None:
            return profile_id(self.family, self.profile)  # all styles share the class
        if self.tnut_kind is not None:
            return self.part_id
        return plain_id(self.family, self.size)


def _fmt_num(x: float) -> str:
    """Format a number without a trailing .0 (16 -> '16', 2.5 -> '2.5')."""
    if x <= 0:
        raise ValueError(f"expected a positive number, got {x}")
    return f"{float(x):g}"


def format_size(size: str | float) -> str:
    """Normalise a thread size: 4, '4', 'm4', 'M4' -> 'M4'; 2.5 -> 'M2.5'."""
    s = str(size).strip().upper()
    if s.startswith("M"):
        s = s[1:]
    try:
        value = float(s)
    except ValueError:
        raise ValueError(f"invalid thread size: {size!r}") from None
    return "M" + _fmt_num(value)


def size_value(size: str | float) -> float:
    """Nominal diameter in mm: 'M2.5' -> 2.5."""
    return float(format_size(size)[1:])


def _check_family(family: str) -> str:
    if not re.fullmatch(_FAMILY, family):
        raise ValueError(
            f"invalid family {family!r}: use upper-case letters and digits, e.g. DIN912"
        )
    if family == TNUT_FAMILY or family.startswith(f"{TNUT_FAMILY}_"):
        raise ValueError("use tnut_id() for T-nuts")
    if family == UNKNOWN or family.startswith(f"{UNKNOWN}_"):
        raise ValueError("use unknown_id() for unknown parts")
    return family


def screw_id(family: str, size: str | float, length: float) -> str:
    return f"{_check_family(family)}_{format_size(size)}x{_fmt_num(length)}"


def plain_id(family: str, size: str | float) -> str:
    """Id of a part without a length (nut, washer); also the class name of a screw."""
    return f"{_check_family(family)}_{format_size(size)}"


def profile_id(family: str, profile: str | int, style: str | None = None) -> str:
    """Id of a part made for an extrusion profile: BRACKET_2020, or with a style (a look-alike
    variant of the same class) BRACKET_2020_CAST. Without a style it is also the class name."""
    p = str(profile).strip()
    if not re.fullmatch(_PROFILE, p):
        raise ValueError(f"profile sizes have 4 digits (2020, 3030, 4040), got {profile!r}")
    pid = f"{_check_family(family)}_{p}"
    if style is None:
        return pid
    st = re.sub(r"[^A-Z0-9]", "", str(style).upper())
    if not st or not st[0].isalpha():
        raise ValueError(f"invalid style {style!r}: letters and digits, starting with a letter")
    return f"{pid}_{st}"


def tnut_id(kind: str, slot: int, size: str | float) -> str:
    kind = kind.upper()
    if kind not in TNUT_KINDS:
        raise ValueError(f"T-nut kind must be one of {TNUT_KINDS}, got {kind!r}")
    if int(slot) != slot or slot <= 0:
        raise ValueError(f"invalid slot width {slot!r}")
    return f"{TNUT_FAMILY}_{kind}_S{int(slot)}_{format_size(size)}"


def unknown_id(kind: str, n: int) -> str:
    """Id of an unknown (non-catalogue) part: unknown_id('wood_screw', 3) -> UNKNOWN_WOODSCREW_03."""
    t = re.sub(r"[^A-Z0-9]", "", kind.upper())
    if not t or not t[0].isalpha():
        raise ValueError(f"invalid unknown part type {kind!r}")
    return f"{UNKNOWN}_{t}_{int(n):02d}"


def parse_part_id(part_id: str) -> PartName:
    """Parse a part id into its components; ValueError if it breaks the naming convention."""
    if m := _UNKNOWN_RE.match(part_id):
        return PartName(family=UNKNOWN, size="-", variant=m["variant"])
    if m := _TNUT_RE.match(part_id):
        return PartName(
            family=TNUT_FAMILY,
            size=format_size(m["size"]),
            tnut_kind=m["kind"],
            slot=int(m["slot"]),
        )
    if m := _SCREW_RE.match(part_id):
        _check_family(m["family"])
        return PartName(family=m["family"], size=format_size(m["size"]), length=float(m["length"]))
    if m := _PLAIN_RE.match(part_id):
        _check_family(m["family"])
        return PartName(family=m["family"], size=format_size(m["size"]))
    if m := _PROFILE_RE.match(part_id):
        _check_family(m["family"])
        return PartName(
            family=m["family"], size=m["profile"], profile=m["profile"], style=m["style"]
        )
    raise ValueError(f"part id does not follow the naming convention: {part_id!r}")


def class_name(part_id: str) -> str:
    """Class name of a part: its id without the length (DIN912_M4x16 -> DIN912_M4)."""
    return parse_part_id(part_id).class_name
