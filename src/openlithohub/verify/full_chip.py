"""Streaming-friendly aggregation of proof-carrying tile certificates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .types import CertificateStatus


@dataclass(frozen=True)
class TileProofStatus:
    tile_id: str
    status: CertificateStatus
    load_bearing: bool = True
    artifact_sha256: str | None = None
    note: str = ""


@dataclass(frozen=True)
class FullChipAggregate:
    status: CertificateStatus
    load_bearing_tiles: int
    passed_tiles: int
    failed_tiles: int
    inconclusive_tiles: int


def aggregate_full_chip_status(records: Iterable[TileProofStatus]) -> FullChipAggregate:
    """Aggregate without storing dense full-chip raster output.

    Acceptance rule:
      any certified load-bearing FAIL -> global FAIL;
      else any load-bearing INCONCLUSIVE -> global INCONCLUSIVE;
      else if all load-bearing tiles PASS -> global PASS.
    """
    load = passed = failed = inconclusive = 0
    for rec in records:
        if not rec.load_bearing:
            continue
        load += 1
        if rec.status is CertificateStatus.PASS:
            passed += 1
        elif rec.status is CertificateStatus.FAIL:
            failed += 1
        else:
            inconclusive += 1

    if load == 0:
        status = CertificateStatus.INCONCLUSIVE
    elif failed:
        status = CertificateStatus.FAIL
    elif inconclusive:
        status = CertificateStatus.INCONCLUSIVE
    else:
        status = CertificateStatus.PASS

    return FullChipAggregate(
        status=status,
        load_bearing_tiles=load,
        passed_tiles=passed,
        failed_tiles=failed,
        inconclusive_tiles=inconclusive,
    )
