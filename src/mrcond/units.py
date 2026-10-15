"""Unit parsing and normalisation for MR Conditional labeling quantities.

Canonical units per physical dimension:

==========  ===============  ==========================================
dimension   canonical unit   accepted spellings (examples)
==========  ===============  ==========================================
field       T                T, Tesla, -Tesla, mT, gauss (G)
gradient    T/m              T/m, mT/m, G/cm, gauss/cm (1 T/m = 100 G/cm)
slew        T/m/s            T/m/s, T/m/sec, mT/m/ms, T/(m*s), G/cm/s (100 G/cm/s = 1 T/m/s)
b1          uT               µT, μT, uT, microtesla
sar         W/kg             W/kg, W/Kg, watts per kilogram
time        min              min, minutes, s, sec, seconds, h, hours
length      mm               mm, millimeters, cm, centimeters
==========  ===============  ==========================================

All conversions are exact linear factors; values are rounded to 6 significant
digits to remove floating point noise (e.g. 3,000 G/cm -> 30 T/m).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "CANONICAL_UNITS",
    "Quantity",
    "UnitInfo",
    "UNITS",
    "canonical_unit",
    "convert",
    "find_quantities",
    "gauss_per_cm_to_tesla_per_m",
    "normalize_unit",
    "parse_number",
    "round_sig",
    "tesla_per_m_to_gauss_per_cm",
    "to_canonical",
]

CANONICAL_UNITS: dict[str, str] = {
    "field": "T",
    "gradient": "T/m",
    "slew": "T/m/s",
    "b1": "uT",
    "sar": "W/kg",
    "time": "min",
    "length": "mm",
    "dbdt": "T/s",
    "product": "T2/m",
}


@dataclass(frozen=True)
class UnitInfo:
    """A unit spelling resolved to its dimension and conversion factor."""

    unit: str  # normalised unit id, e.g. "G/cm"
    dimension: str  # e.g. "gradient"
    factor: float  # multiply a value in `unit` by this to get the canonical unit


UNITS: dict[str, UnitInfo] = {
    "T": UnitInfo("T", "field", 1.0),
    "mT": UnitInfo("mT", "field", 1e-3),
    "G": UnitInfo("G", "field", 1e-4),
    "T/m": UnitInfo("T/m", "gradient", 1.0),
    "mT/m": UnitInfo("mT/m", "gradient", 1e-3),
    "G/cm": UnitInfo("G/cm", "gradient", 1e-2),
    "T/m/s": UnitInfo("T/m/s", "slew", 1.0),
    "mT/m/ms": UnitInfo("mT/m/ms", "slew", 1.0),
    "G/cm/s": UnitInfo("G/cm/s", "slew", 1e-2),
    "uT": UnitInfo("uT", "b1", 1.0),
    "W/kg": UnitInfo("W/kg", "sar", 1.0),
    "min": UnitInfo("min", "time", 1.0),
    "s": UnitInfo("s", "time", 1.0 / 60.0),
    "h": UnitInfo("h", "time", 60.0),
    "mm": UnitInfo("mm", "length", 1.0),
    "cm": UnitInfo("cm", "length", 10.0),
    "T/s": UnitInfo("T/s", "dbdt", 1.0),  # rate of change dB/dt - not a field strength
    "T2/m": UnitInfo("T2/m", "product", 1.0),  # B0 x spatial gradient product
}

# Ordered (longest / most specific first). Each entry: (regex, unit id).
_UNIT_SPELLINGS: list[tuple[str, str]] = [
    (r"mT\s*/\s*m\s*/\s*ms\b", "mT/m/ms"),
    (r"T\s*/\s*m\s*/\s*s(?:ec(?:onds?)?)?\b", "T/m/s"),
    (r"T\s*/\s*\(\s*m\s*[·⋅*x]?\s*s\s*\)", "T/m/s"),
    (r"T\s*·?\s*m-1\s*·?\s*s-1\b", "T/m/s"),
    (r"(?:[Tt]esla|T)\s+per\s+met(?:er|re)\s+per\s+sec(?:ond)?\b", "T/m/s"),
    (r"T\s*[²2]\s*/\s*m\b", "T2/m"),
    (r"T\s*/\s*s(?:ec(?:onds?)?)?\b", "T/s"),
    (r"(?:mT|millitesla)\s*/\s*m(?:et(?:er|re))?\b(?!\s*/)", "mT/m"),
    (r"(?:[Tt]esla|T)\s*/\s*m(?:et(?:er|re))?\b(?!\s*/)", "T/m"),
    (r"(?:[Tt]esla|T)\s+per\s+met(?:er|re)\b(?!\s+per)", "T/m"),
    (r"(?:[Gg]auss|G)\s*/\s*cm\s*/\s*s(?:ec(?:onds?)?)?\b", "G/cm/s"),
    (r"(?:[Gg]auss|G)\s*/\s*cm\b(?!\s*/)", "G/cm"),
    (r"(?:[Gg]auss|G)\s+per\s+cent(?:imet(?:er|re))?\b", "G/cm"),
    (r"(?:µ|μ|u|mc)\s?T\b", "uT"),
    (r"micro-?\s?[Tt]esla\b", "uT"),
    (r"W\s*/\s*[Kk]g\b", "W/kg"),
    (r"[Ww]atts?\s*(?:/|per)\s*(?:kg|kilogram)\b", "W/kg"),
    (r"mT\b", "mT"),
    (r"millitesla\b", "mT"),
    (r"(?:[Tt]esla|TESLA)\b", "T"),
    (r"T(?![²2]\s*/)(?![a-zA-Z])", "T"),
    (r"(?:minutes?|mins?)\b", "min"),
    (r"min\.?", "min"),
    (r"(?:seconds?|secs?)\b", "s"),
    (r"s\b", "s"),
    (r"(?:hours?|hrs?)\b", "h"),
    (r"h\b", "h"),
    (r"(?:millimet(?:er|re)s?|mm)\b", "mm"),
    (r"(?:centimet(?:er|re)s?|cm)\b", "cm"),
    (r"[Gg]auss\b", "G"),
]

_UNIT_RE = re.compile("|".join(f"(?P<u{i}>{p})" for i, (p, _) in enumerate(_UNIT_SPELLINGS)))

# Numbers: thousands separators ("3,000"), decimal point or decimal comma ("1,5").
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
_NUMBER_RE = re.compile(rf"(?<![\w.]){_NUMBER}")

# A quantity: number, optional hyphen/space, unit. A list of numbers sharing a
# trailing unit ("1.5 and 3 T", "1.5 T or 3.0 T") is handled in find_quantities.
_QUANTITY_RE = re.compile(rf"(?<![\w.])(?P<num>{_NUMBER})\s*(?:-|‑|–)?\s*(?P<unit>{_UNIT_RE.pattern})")
_SHARED_UNIT_RE = re.compile(
    rf"(?<![\w.])(?P<nums>{_NUMBER}(?:\s*(?:,|/|and|or|&)\s*{_NUMBER})+)\s*(?:-|‑|–)?\s*"
    rf"(?P<unit>{_UNIT_RE.pattern})"
)


def parse_number(text: str) -> float:
    """Parse a number as printed in labeling ("3,000" -> 3000.0, "1,5" -> 1.5)."""
    s = text.strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        return float(s.replace(",", ""))
    if re.fullmatch(r"\d+,\d+", s):
        return float(s.replace(",", "."))
    return float(s)


def round_sig(x: float, sig: int = 6) -> float:
    """Round to `sig` significant digits (removes float noise from conversions)."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def normalize_unit(raw: str | None) -> UnitInfo | None:
    """Resolve a raw unit spelling (e.g. "gauss/cm", "μT", "minutes") to a UnitInfo."""
    if raw is None:
        return None
    s = raw.strip().rstrip(".")
    if not s:
        return None
    if s in UNITS:
        return UNITS[s]
    m = _UNIT_RE.fullmatch(s)
    if m is None:
        # tolerate surrounding words, e.g. "W/kg (Normal Operating Mode)"
        m = _UNIT_RE.match(s)
        if m is None:
            return None
    for i, (_, unit) in enumerate(_UNIT_SPELLINGS):
        if m.group(f"u{i}") is not None:
            return UNITS[unit]
    return None  # pragma: no cover


def canonical_unit(dimension: str) -> str:
    return CANONICAL_UNITS[dimension]


def to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Convert `value` in `unit` (any accepted spelling) to the dimension's canonical unit."""
    info = normalize_unit(unit)
    if info is None:
        raise ValueError(f"unknown unit: {unit!r}")
    return round_sig(value * info.factor), CANONICAL_UNITS[info.dimension]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between two units of the same dimension (e.g. T/m -> G/cm)."""
    a, b = normalize_unit(from_unit), normalize_unit(to_unit)
    if a is None or b is None:
        raise ValueError(f"unknown unit: {from_unit!r} or {to_unit!r}")
    if a.dimension != b.dimension:
        raise ValueError(f"incompatible units: {from_unit} ({a.dimension}) vs {to_unit} ({b.dimension})")
    return round_sig(value * a.factor / b.factor)


def tesla_per_m_to_gauss_per_cm(x: float) -> float:
    """1 T/m = 100 G/cm."""
    return round_sig(x * 100.0)


def gauss_per_cm_to_tesla_per_m(x: float) -> float:
    """100 G/cm = 1 T/m."""
    return round_sig(x / 100.0)


@dataclass(frozen=True)
class Quantity:
    """A number with a unit found in text, with character offsets into that text."""

    value: float
    raw_number: str
    raw_unit: str
    unit: str  # normalised unit id, e.g. "G/cm"
    dimension: str
    canonical_value: float
    canonical_unit: str
    start: int  # offset of the number
    end: int  # offset just past the unit

    def as_canonical(self) -> tuple[float, str]:
        return self.canonical_value, self.canonical_unit


def _unit_info_from_match(m: re.Match[str]) -> UnitInfo:
    for i, (_, unit) in enumerate(_UNIT_SPELLINGS):
        if m.group(f"u{i}") is not None:
            return UNITS[unit]
    raise AssertionError("unit group not found")  # pragma: no cover


def find_quantities(text: str, dimensions: set[str] | None = None) -> list[Quantity]:
    """Find all `<number> <unit>` quantities in text.

    Lists with a shared trailing unit ("1.5 and 3.0 T", "1.5/3 T") yield one
    Quantity per number. Results are sorted by start offset and de-duplicated.
    """
    found: dict[int, Quantity] = {}
    for m in _SHARED_UNIT_RE.finditer(text):
        info = _unit_info_from_match(m)
        if info.dimension != "field":
            # shared-unit lists are only trusted for field strengths ("1.5 and 3 T")
            continue
        unit_raw = m.group("unit")
        for nm in _NUMBER_RE.finditer(m.group("nums")):
            start = m.start("nums") + nm.start()
            val = parse_number(nm.group(0))
            cv = round_sig(val * info.factor)
            found.setdefault(
                start,
                Quantity(
                    val,
                    nm.group(0),
                    unit_raw,
                    info.unit,
                    info.dimension,
                    cv,
                    CANONICAL_UNITS[info.dimension],
                    start,
                    m.end(),
                ),
            )
    for m in _QUANTITY_RE.finditer(text):
        info = _unit_info_from_match(m)
        start = m.start("num")
        val = parse_number(m.group("num"))
        cv = round_sig(val * info.factor)
        q = Quantity(
            val,
            m.group("num"),
            m.group("unit"),
            info.unit,
            info.dimension,
            cv,
            CANONICAL_UNITS[info.dimension],
            start,
            m.end(),
        )
        found[start] = q  # explicit unit wins over shared-unit inference
    out = sorted(found.values(), key=lambda q: q.start)
    if dimensions is not None:
        out = [q for q in out if q.dimension in dimensions]
    return out


def find_numbers(text: str) -> list[tuple[float, int, int]]:
    """All bare numbers in text as (value, start, end)."""
    res = []
    for m in _NUMBER_RE.finditer(text):
        try:
            res.append((parse_number(m.group(0)), m.start(), m.end()))
        except ValueError:  # pragma: no cover
            continue
    return res


_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
    "sixty": 60,
    "ninety": 90,
}
_WORD_TIME_RE = re.compile(
    r"\b(?P<w>" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")[\s-]+(?P<u>hours?|minutes?)\b",
    re.IGNORECASE,
)


def find_word_durations(text: str) -> list[Quantity]:
    """Durations written in words ("one hour", "thirty minutes")."""
    out = []
    for m in _WORD_TIME_RE.finditer(text):
        val = float(_WORD_NUMBERS[m.group("w").lower()])
        info = UNITS["h"] if m.group("u").lower().startswith("hour") else UNITS["min"]
        out.append(
            Quantity(
                val,
                m.group("w"),
                m.group("u"),
                info.unit,
                "time",
                round_sig(val * info.factor),
                "min",
                m.start(),
                m.end(),
            )
        )
    return out
