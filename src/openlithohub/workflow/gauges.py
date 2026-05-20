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

# Common synonyms seen in the wild. Lowercased keys → (canonical name, scale-to-nm).
# Downstream consumers (eval_cmd, weighted_rms_epe) treat all length-bearing
# values as nm. ``_um`` aliases multiply by 1000 at parse time so an input file
# in microns produces nm internally — silently mismatching units used to be a
# 1000x error in the EPE numbers.
_ALIASES: dict[str, tuple[str, float]] = {
    "x": ("x", 1.0),
    "y": ("y", 1.0),
    "x_nm": ("x", 1.0),
    "y_nm": ("y", 1.0),
    "x_um": ("x", 1000.0),
    "y_um": ("y", 1000.0),
    "tangent": ("tangent", 1.0),
    "tangent_deg": ("tangent", 1.0),
    "angle": ("tangent", 1.0),
    "theta": ("tangent", 1.0),
    "target": ("target_cd", 1.0),
    "target_cd": ("target_cd", 1.0),
    "target_nm": ("target_cd", 1.0),
    "target_um": ("target_cd", 1000.0),
    "cd_target": ("target_cd", 1.0),
    "measured": ("measured_cd", 1.0),
    "measured_cd": ("measured_cd", 1.0),
    "measured_nm": ("measured_cd", 1.0),
    "measured_um": ("measured_cd", 1000.0),
    "cd_measured": ("measured_cd", 1.0),
    "cd": ("measured_cd", 1.0),
    "weight": ("weight", 1.0),
    "w": ("weight", 1.0),
}


@dataclass(frozen=True)
class GaugePoint:
    """One gauge measurement point.

    Coordinates and CDs are in nanometers. Inputs labeled with ``_um`` are
    converted to nm at parse time so all downstream consumers can assume
    nm units uniformly. Tangent is degrees CCW from +x. ``measured_cd``
    may be ``None`` when the gauge file specifies targets only
    (pre-measurement).
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
    # ICCAD'13 contest gauges carry a ±1 polarity flag per point (inside
    # vs outside feature) that has no analogue in Calibre / CSV gauges.
    # Populated only by :func:`parse_iccad13_gauge`; ``None`` otherwise.
    iccad13_polarities: tuple[int, ...] | None = None

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
        return float((num / wsum) ** 0.5)


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


def parse_iccad13_gauge(path: str | Path) -> GaugeTable:
    """Parse the ICCAD'13 contest gauge format.

    The contest distributes 5-column whitespace-separated text with no
    header: ``gauge_id  x_nm  y_nm  angle_deg  polarity``. ``polarity`` is
    ``+1`` for an inside-feature gauge and ``-1`` for outside (the
    convention used by the ICCAD'13 / SPIE contest scoring scripts).

    The format has no ``target_cd`` or ``measured_cd`` column — the
    contest scores EPE relative to the design intent encoded in the GDS,
    not against a measurement column. We map ``target_cd=0.0`` and
    ``measured_cd=None`` so downstream EPE code that reads
    ``GaugeTable`` keeps working; callers that need the polarity can
    inspect it via ``GaugeTable.iccad13_polarities`` after parsing.

    Lines beginning with ``#`` are treated as comments and skipped.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gauge file not found: {p}")

    points: list[GaugePoint] = []
    polarities: list[int] = []
    with p.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError(
                    f"{p.name}:{lineno}: ICCAD'13 gauge expects 5 columns "
                    f"(gauge_id x y angle polarity); got {len(tokens)}: {tokens}"
                )
            try:
                x = float(tokens[1])
                y = float(tokens[2])
                angle = float(tokens[3])
                pol = int(tokens[4])
            except ValueError as e:
                raise ValueError(f"{p.name}:{lineno}: failed to parse row {tokens}: {e}") from None
            if pol not in (-1, 1):
                raise ValueError(f"{p.name}:{lineno}: polarity must be +1 or -1, got {pol}")
            points.append(
                GaugePoint(
                    x=x,
                    y=y,
                    tangent=angle,
                    target_cd=0.0,
                    measured_cd=None,
                    weight=1.0,
                )
            )
            polarities.append(pol)

    table = GaugeTable(points=tuple(points), source=p, iccad13_polarities=tuple(polarities))
    return table


def write_iccad13_gauge(
    path: str | Path,
    points: list[GaugePoint] | tuple[GaugePoint, ...],
    polarities: list[int] | tuple[int, ...],
) -> None:
    """Write a gauge list to ICCAD'13 5-column text format.

    ``polarities`` must have the same length as ``points``; every entry
    must be ``+1`` or ``-1``. ``target_cd`` and ``measured_cd`` are not
    serialized because the contest format has no slot for them — write
    a CSV via :func:`parse_gauge` round-trip if you need those.
    """
    if len(points) != len(polarities):
        raise ValueError(
            f"points ({len(points)}) and polarities ({len(polarities)}) length mismatch"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for i, (pt, pol) in enumerate(zip(points, polarities, strict=True), start=1):
            if pol not in (-1, 1):
                raise ValueError(f"row {i}: polarity must be +1 or -1, got {pol}")
            fh.write(f"{i}\t{pt.x:.4f}\t{pt.y:.4f}\t{pt.tangent:.4f}\t{pol}\n")


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

    A header is REQUIRED. The previous behaviour fell back to the canonical
    column order when no header was found, which silently produced wrong
    EPE numbers on hand-written .gg variants that use a different column
    order. Refusing the file forces the caller to fix the input.
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
        raise ValueError(
            f"Gauge file {path.name} has no recognizable header. Calibre .gg "
            f"files must start with a '#'-prefixed line naming all four required "
            f"columns (x, y, tangent, target_cd)."
        )
    return rows, header


def _is_header_line(tokens: list[str]) -> bool:
    """A header line must name all four required canonical columns."""
    if not tokens or any(not _looks_like_name(t) for t in tokens):
        return False
    resolved = {_ALIASES.get(t.lower(), (None, 1.0))[0] for t in tokens}
    return {"x", "y", "tangent", "target_cd"}.issubset(resolved)


def _looks_like_name(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return True
    return False


def _canonicalize(names: list[str]) -> dict[str, tuple[int, float]]:
    """Map canonical name → (column index, scale-to-nm). Unknown columns are dropped."""
    out: dict[str, tuple[int, float]] = {}
    for i, name in enumerate(names):
        resolved = _ALIASES.get(name.lower())
        if resolved is not None and resolved[0] not in out:
            out[resolved[0]] = (i, resolved[1])
    missing = [c for c in ("x", "y", "tangent", "target_cd") if c not in out]
    if missing:
        raise ValueError(
            f"Gauge file is missing required column(s): {missing}. "
            f"Saw: {names}. Recognized aliases: {sorted(set(_ALIASES))}."
        )
    return out


def _row_to_point(row: list[str], canon: dict[str, tuple[int, float]]) -> GaugePoint:
    def f(key: str) -> float:
        idx, scale = canon[key]
        return float(row[idx]) * scale

    measured: float | None
    if "measured_cd" in canon:
        idx, scale = canon["measured_cd"]
        raw = row[idx].strip()
        measured = None if raw == "" or raw.upper() == "NA" else float(raw) * scale
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
