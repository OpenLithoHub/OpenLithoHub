"""Gauge file parsing — measurement points used to score OPC fidelity.

A "gauge" is a point on the layout where the foundry / EDA tool measures
the printed CD (critical dimension) and compares it to the target. OPC
recipes are tuned to minimize the EPE (edge-placement error) at these
points. Calibre / Synopsys / commercial OPC pipelines all consume gauge
files; the formats are slightly different but the schema is essentially
the same:

* Calibre ``.gg`` text — whitespace-separated, often with ``# `` comments
  and an optional header line giving the column order.
* Generic CSV — same columns, comma-separated, header row required.

This module returns a uniform ``GaugeTable`` regardless of input format,
so downstream OPC scoring / training code does not have to care which
tool produced the gauges.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# The canonical column set we expose. Inputs may use different names; we
# normalize to these. Tangent is the gauge-line direction in degrees CCW
# from +x; measurement is taken perpendicular to it.
_CANONICAL = ("x", "y", "tangent", "target_cd", "measured_cd", "weight")

# Common synonyms seen in the wild. Lowercased keys → canonical name.
_ALIASES: dict[str, str] = {
    "x": "x",
    "y": "y",
    "x_nm": "x",
    "y_nm": "y",
    "x_um": "x",
    "y_um": "y",
    "tangent": "tangent",
    "tangent_deg": "tangent",
    "angle": "tangent",
    "theta": "tangent",
    "target": "target_cd",
    "target_cd": "target_cd",
    "target_nm": "target_cd",
    "cd_target": "target_cd",
    "measured": "measured_cd",
    "measured_cd": "measured_cd",
    "measured_nm": "measured_cd",
    "cd_measured": "measured_cd",
    "cd": "measured_cd",
    "weight": "weight",
    "w": "weight",
}


@dataclass(frozen=True)
class GaugePoint:
    """One gauge measurement point.

    Coordinates and CDs are in the file's native units (typically nm).
    Tangent is degrees CCW from +x. ``measured_cd`` may be ``None`` when
    the gauge file specifies targets only (pre-measurement).
    """

    x: float
    y: float
    tangent: float
    target_cd: float
    measured_cd: float | None
    weight: float


@dataclass(frozen=True)
class GaugeTable:
    """A parsed gauge file."""

    points: tuple[GaugePoint, ...]
    source: Path

    def __len__(self) -> int:
        return len(self.points)

    def epe(self) -> tuple[float, ...]:
        """Edge-placement error (measured - target) per point.

        Raises ValueError if any point has no measurement.
        """
        out: list[float] = []
        for p in self.points:
            if p.measured_cd is None:
                raise ValueError(f"Cannot compute EPE: gauge at ({p.x}, {p.y}) has no measured_cd.")
            out.append(p.measured_cd - p.target_cd)
        return tuple(out)

    def weighted_rms_epe(self) -> float:
        """sqrt( sum(w * epe^2) / sum(w) ) — the canonical OPC scoring metric."""
        epes = self.epe()
        wsum = sum(p.weight for p in self.points)
        if wsum == 0.0:
            raise ValueError("All gauge weights are zero; weighted RMS is undefined.")
        num = sum(p.weight * e * e for p, e in zip(self.points, epes, strict=True))
        return (num / wsum) ** 0.5


def parse_gauge(path: str | Path) -> GaugeTable:
    """Parse a gauge file (Calibre ``.gg`` or generic CSV).

    The dispatch is by extension:

    * ``.gg`` / ``.gauge`` / ``.txt`` → whitespace-separated, ``#`` comments.
      A header line beginning with ``#`` may give column names; otherwise
      we assume the canonical order ``x y tangent target_cd measured_cd weight``.
    * ``.csv`` → comma-separated, **header row required**. Column names
      are matched against a small synonym table (e.g. ``cd_target`` →
      ``target_cd``); unknown columns are ignored.

    Missing ``weight`` defaults to ``1.0``. Missing ``measured_cd`` (or the
    string "NA" / empty) is preserved as ``None`` so callers can tell
    "not yet measured" apart from "measured to be 0".
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gauge file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        rows, names = _read_csv(p)
    elif suffix in (".gg", ".gauge", ".txt"):
        rows, names = _read_calibre(p)
    else:
        raise ValueError(f"Unsupported gauge format: {suffix!r}. Use .gg, .gauge, .txt, or .csv.")

    canon = _canonicalize(names)
    points = tuple(_row_to_point(row, canon) for row in rows)
    return GaugeTable(points=points, source=p)


def _read_csv(path: Path) -> tuple[list[list[str]], list[str]]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Gauge CSV {path.name} is empty.") from None
        rows = [row for row in reader if row and not all(c.strip() == "" for c in row)]
    return rows, [h.strip() for h in header]


def _read_calibre(path: Path) -> tuple[list[list[str]], list[str]]:
    """Calibre .gg format: whitespace-separated, '#' comments.

    A comment line is treated as the header iff it contains tokens that
    cover all four required canonical columns (x, y, tangent, target_cd)
    after alias resolution. This avoids mistaking arbitrary commentary
    like '# Calibre OPCverify dump' for a header.
    """
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if header is None:
                    candidate = line.lstrip("#").strip().split()
                    if _is_header_line(candidate):
                        header = candidate
                continue
            rows.append(line.split())
    if header is None:
        header = list(_CANONICAL)
    return rows, header


def _is_header_line(tokens: list[str]) -> bool:
    """A header line must name all four required canonical columns."""
    if not tokens or any(not _looks_like_name(t) for t in tokens):
        return False
    resolved = {_ALIASES.get(t.lower()) for t in tokens}
    return {"x", "y", "tangent", "target_cd"}.issubset(resolved)


def _looks_like_name(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return True
    return False


def _canonicalize(names: list[str]) -> dict[str, int]:
    """Map canonical name → column index. Unknown columns are dropped."""
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        key = _ALIASES.get(name.lower())
        if key is not None and key not in out:
            out[key] = i
    missing = [c for c in ("x", "y", "tangent", "target_cd") if c not in out]
    if missing:
        raise ValueError(
            f"Gauge file is missing required column(s): {missing}. "
            f"Saw: {names}. Recognized aliases: {sorted(set(_ALIASES))}."
        )
    return out


def _row_to_point(row: list[str], canon: dict[str, int]) -> GaugePoint:
    def f(key: str) -> float:
        return float(row[canon[key]])

    measured: float | None
    if "measured_cd" in canon:
        raw = row[canon["measured_cd"]].strip()
        measured = None if raw == "" or raw.upper() == "NA" else float(raw)
    else:
        measured = None

    weight = f("weight") if "weight" in canon else 1.0

    return GaugePoint(
        x=f("x"),
        y=f("y"),
        tangent=f("tangent"),
        target_cd=f("target_cd"),
        measured_cd=measured,
        weight=weight,
    )
