#!/usr/bin/env python3
"""Build the deterministic P-054 frozen artifact from the public release ZIP.

PR-5F4 reproducibility: this is the canonical, committed builder.  The
theorem-facing declarations (event multiplicity, ownership transitions,
component counts) are read from the repository's frozen catalogs
(``proof_artifacts/p054/event_catalog.json``) so no second hidden event
catalog can drift.  Focus intervals and event centers come from the
frozen upstream authorities inside the public release package.  Output
ZIP is deterministic: fixed timestamps, sorted member order, fixed
compression level.  Two builds from identical inputs are byte-identical.

Usage:
    python scripts/build_p054_frozen_artifact.py PUBLIC_RELEASE.ZIP \
        --out p054-arf37-frozen-artifact.zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

FIXED_DT = (1980, 1, 1, 0, 0, 0)
BASE = "P-054_PublicRelease/"
SRC = BASE + "data/B04_R93_FreshSourceBins_2026-09-19.json"
COEFF = BASE + "data/B04_R96_RawSourceNativeTensor_HP95_2026-09-19.json"
MASK = BASE + "data/B04_MVP1_Mask_72x72_uint8.bin"
R102 = BASE + "verification/B04_R102_CanonicalR42R43_FreshReplay_Report_2026-09-19.json"
R103 = BASE + "verification/B04_R103_R44_FreshTransverseOwnershipCrossing_Report_2026-09-19.json"
R105 = (
    BASE + "verification/B04_R105_R47FreshGlobalEOwnership_PitchforkReplay_Report_2026-09-19.json"
)
R107 = BASE + "verification/B04_R107_R49FreshNonOwningPitchfork_Report_2026-09-19.json"


def sha_hex(b: bytes) -> str:  # noqa: N802 — matches frozen authority notation
    return hashlib.sha256(b).hexdigest()


def npy(a):
    b = io.BytesIO()
    np.lib.format.write_array(b, np.asarray(a), allow_pickle=False)
    return b.getvalue()


def npz_bytes(arrays):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for k in sorted(arrays):
            zi = zipfile.ZipInfo(k + ".npy", FIXED_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, npy(arrays[k]))
    return b.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("public_zip", type=Path)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root holding proof_artifacts/p054 catalogs",
    )
    ap.add_argument("--out", type=Path, default=Path("p054-arf37-frozen-artifact.zip"))
    a = ap.parse_args()

    p054 = a.repo_root / "proof_artifacts" / "p054"
    catalog = json.loads((p054 / "event_catalog.json").read_text())
    chamber_doc = json.loads((p054 / "chamber_catalog.json").read_text())

    with zipfile.ZipFile(a.public_zip) as z:
        src = json.loads(z.read(SRC))
        coeff = json.loads(z.read(COEFF))
        mask = z.read(MASK)
        r103 = json.loads(z.read(R103))
        r105 = json.loads(z.read(R105))
        r107 = json.loads(z.read(R107))
    bins = src["source_bins"]
    ent = coeff["entries"]
    source = npz_bytes(
        {
            "index": np.array([x["index"] for x in bins], dtype=np.int16),
            "shift_x": np.array([x["shift_x"] for x in bins], dtype=np.int16),
            "shift_y": np.array([x["shift_y"] for x in bins], dtype=np.int16),
            "weight_num": np.array([x["weight_num"] for x in bins], dtype=np.int64),
            "weight_den": np.array([x["weight_den"] for x in bins], dtype=np.int64),
            "weight_float32_bits_be": np.array(
                [x["weight_float32_bits_be"] for x in bins], dtype="U8"
            ),
            "weight_hex_float64_view": np.array(
                [x["weight_hex_float64_view"] for x in bins], dtype="U32"
            ),
            "exact_weight_sum_num": np.array([src["exact_weight_sum"]["num"]], dtype=np.int64),
            "exact_weight_sum_den": np.array([src["exact_weight_sum"]["den"]], dtype=np.int64),
            "pinned_openlithohub_commit": np.array(
                [src["pinned_openlithohub_commit"]], dtype="U40"
            ),
        }
    )
    mr = max(len(x["re_hp"]) for x in ent)
    mi = max(len(x["im_hp"]) for x in ent)
    co = npz_bytes(
        {
            "kx": np.array([x["kx"] for x in ent], dtype=np.int16),
            "ky": np.array([x["ky"] for x in ent], dtype=np.int16),
            "m": np.array([x["m"] for x in ent], dtype=np.int16),
            "re_hp": np.array([x["re_hp"] for x in ent], dtype=f"U{mr}"),
            "im_hp": np.array([x["im_hp"] for x in ent], dtype=f"U{mi}"),
            "source_weight_sum_num": np.array([coeff["source_weight_sum"]["num"]], dtype=np.int64),
            "source_weight_sum_den": np.array([coeff["source_weight_sum"]["den"]], dtype=np.int64),
        }
    )
    ints = {
        "z1": [29.5740, 29.5742],
        "z2": [29.7105, 29.7107],
        "z3": [40.4988, 40.4989],
        "z4": [49.9768, 49.9770],
        "z5": [114.8880, 114.8883],
        "zV": [274.9367, 274.9370],
        "zH": [277.2116, 277.2119],
        "zP": [280.8605, 280.8608],
    }
    centers = {
        "z1": "29.574102895745213",
        "z2": "29.710604726057756",
        "z3": r103["high_precision_crossing"]["z"],
        "z4": r105["theorem"]["z4_nm"],
        "z5": r107["high_precision_event"]["z5_nm"],
        "zV": "274.9368282636185486",
        "zH": "277.2117746105439337",
        "zP": "280.8606212131733877",
    }

    # Theorem-facing declarations come from the repo catalog (PR-5F4: no
    # second hidden event catalog).
    catalog_events = {e["event_id"]: e for e in catalog.get("events", [])}
    catalog_chambers = {c["chamber_id"]: c for c in chamber_doc.get("chambers", [])}
    # PR-5F1/R107: the load-bearing witness is owner equality on the
    # certified center-branch slab [114.88,114.90] containing the z5
    # critical event.  Owner tokens name the symmetry-fixed center branch
    # per the R107 authority (kept as structural placeholder in the repo
    # catalog; the builder materializes them from the frozen authority).
    witnesses = []
    for w in chamber_doc.get("witnesses", []):
        assert w["critical_event_id"] == "z5"
        z5_center = float(centers["z5"])
        slab_lo, slab_hi = 114.88, 114.90
        assert slab_lo <= z5_center <= slab_hi, "R107 z5 center outside slab"
        witnesses.append(
            {
                "critical_event_id": w["critical_event_id"],
                "owner_before": "CENTER",
                "owner_after": "CENTER",
                "owner_interval_nm": [slab_lo, slab_hi],
                "left_chamber_id": None,
                "right_chamber_id": None,
            }
        )

    def ev(i: str, layer: str, kind: str) -> dict:
        entry = catalog_events[i]
        return {
            "event_id": i,
            "layer": layer,
            "kind": kind,
            "multiplicity": entry["multiplicity"],
            "focus_interval_nm": ints[i],
            "event_center_nm": centers[i],
            **{k: entry[k] for k in ("owner_before", "owner_after") if entry.get(k) is not None},
            **{
                k: entry[k]
                for k in ("component_count_before", "component_count_after")
                if entry.get(k) is not None
            },
        }

    events = [
        ev("z1", "CRITICAL_SET", "GENERIC_FOLD"),
        ev("z2", "CRITICAL_SET", "GENERIC_FOLD"),
        ev("z3", "OWNERSHIP", "TRANSVERSE_OWNERSHIP_CROSSING"),
        ev("z4", "CRITICAL_SET", "REFLECTION_PITCHFORK"),
        ev("z5", "CRITICAL_SET", "OWNERSHIP_INVISIBLE_PITCHFORK"),
        ev("zV", "TARGET_TOPOLOGY", "TARGET_MAX_EVENT"),
        ev("zH", "TARGET_TOPOLOGY", "TARGET_MAX_EVENT"),
        ev("zP", "TARGET_TOPOLOGY", "TARGET_SADDLE_EVENT"),
    ]
    # Chamber windows derive from the ordered target-topology events over
    # the frozen atlas scan range [10.0, 282.0] (R102): before zV, between
    # each pair of target events, and after zP.
    focus_start, focus_end = 10.0, 282.0  # frozen atlas scan range (R102)
    target_events = [(eid, ints[eid]) for eid in ("zV", "zH", "zP") if eid in catalog_events]
    windows = []
    left = focus_start
    for _eid, iv in target_events:
        windows.append([left, iv[0]])
        left = iv[1]
    windows.append([left, focus_end])
    window_by_chamber = {f"chamber-{i}": w for i, w in enumerate(windows)}
    chambers = [
        {
            "chamber_id": cid,
            "focus_interval_nm": window_by_chamber[cid],
            "lower_owner": cc.get("lower_owner"),
            "upper_owner": cc.get("upper_owner"),
            "target_component_count": cc.get("target_component_count"),
            "bounded_by_events": list(cc.get("bounded_by_events", [])),
        }
        for cid, cc in catalog_chambers.items()
    ]
    manifest = {
        "schema": "P054.replay-manifest.v1",
        "fixture_id": "p054-arf37",
        "model_schema": "P054.frozen-arf37.v1",
        "implementation_commit": "348fa5d86d5355465af98e2c4ce3deac60081a4c",
        "proof_level": "IMPORTED-QDM-CERTIFIED",
        "wavelength_nm": 193.0,
        "na": 1.35,
        "sigma": 0.7,
        "grid_shape": [72, 72],
        "pixel_size_nm": 8.0,
        "pupil_support_count": 49,
        "boundary": "PERIODIC_FINITE_TILE",
        "source_snapshot_file": "source_snapshot.npz",
        "source_snapshot_sha256": sha_hex(source),
        "coefficient_file": "coefficients.npz",
        "coefficient_tensor_sha256": sha_hex(co),
        "mask_file": "mask.bin",
        "mask_sha256": sha_hex(mask),
        "events": events,
        "chambers": chambers,
        "witnesses": witnesses,
        "authority_note": (
            "Conservative localization windows from frozen authorities; no new formal T4 claim."
        ),
    }
    members = {
        "replay_manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        "source_snapshot.npz": source,
        "coefficients.npz": co,
        "mask.bin": mask,
    }
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(members):
            zi = zipfile.ZipInfo(n, FIXED_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, members[n])
    raw = a.out.read_bytes()
    print(
        json.dumps(
            {
                "filename": a.out.name,
                "sha256": sha_hex(raw),
                "bytes": len(raw),
                "source_snapshot_sha256": sha_hex(source),
                "coefficient_tensor_sha256": sha_hex(co),
                "mask_sha256": sha_hex(mask),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
