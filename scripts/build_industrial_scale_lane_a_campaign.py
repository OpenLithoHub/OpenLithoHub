"""Build the Lane-A window-scoped campaign manifest (GPU Authority Repair §19).

Each frozen Lane-A window runs as its OWN formal invocation with its own
canonical family + verifier PASS; this builder binds exactly the four
frozen windows (4096/8192/16384/32768) into one campaign manifest under
a single source / fixture / environment / protocol identity.  Every
family is re-verified at the formal tier before acceptance — an
unverified family is never stitched in, and a restart only re-runs the
missing window.

Usage::

    python scripts/build_industrial_scale_lane_a_campaign.py \\
        --window 4096=benchmarks/results/industrial-scale/runs/<id4096>/family \\
        --window 8192=… --window 16384=… --window 32768=… \\
        --out industrial-scale-lane-a-campaign.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale import FROZEN_LANE_A_WINDOWS  # noqa: E402
from openlithohub.benchmark.industrial_scale_campaign import (  # noqa: E402
    build_lane_a_campaign,
    write_campaign_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        metavar="WINDOW=FAMILY_DIR",
        help="frozen window and its verifier-closed family dir (repeat 4×)",
    )
    parser.add_argument("--out", required=True, help="campaign manifest output path")
    args = parser.parse_args()

    families: dict[int, Path] = {}
    for entry in args.window:
        window_text, _, dir_text = entry.partition("=")
        if not dir_text:
            parser.error(f"--window expects WINDOW=FAMILY_DIR, got {entry!r}")
        try:
            window = int(window_text)
        except ValueError:
            parser.error(f"--window expects an integer window, got {window_text!r}")
        if window in families:
            parser.error(f"duplicate window {window}")
        families[window] = Path(dir_text)

    try:
        manifest = build_lane_a_campaign(families)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"LANE-A CAMPAIGN: FAIL — {exc}", file=sys.stderr)
        return 1
    sha = write_campaign_manifest(args.out, manifest)
    print(f"LANE-A CAMPAIGN: BUILT — windows {list(FROZEN_LANE_A_WINDOWS)}")
    print(f"campaign manifest: {args.out}")
    print(f"campaign manifest sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
