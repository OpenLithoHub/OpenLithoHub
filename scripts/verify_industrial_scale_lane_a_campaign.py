"""Verify a Lane-A window-scoped campaign manifest (GPU Authority Repair §19).

Re-checks everything the campaign binds: exactly the four frozen windows
(no missing window, no duplicate), each window's family re-verified at
the formal tier, family/row SHA re-checked against the family bytes, and
ONE measurement source SHA, ONE fixture SHA, ONE environment-lock SHA
and ONE frozen protocol across all four windows.  Any mixture, drift or
non-PASS family is a hard FAIL.

Usage::

    python scripts/verify_industrial_scale_lane_a_campaign.py \\
        --campaign industrial-scale-lane-a-campaign.json \\
        --window 4096=<family-dir> --window 8192=<family-dir> \\
        --window 16384=<family-dir> --window 32768=<family-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale_campaign import (  # noqa: E402
    verify_lane_a_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="campaign manifest JSON path")
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        metavar="WINDOW=FAMILY_DIR",
        help="frozen window and its family dir (repeat 4×)",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.campaign).read_text())
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

    blockers = verify_lane_a_campaign(manifest, families)
    if blockers:
        print("LANE-A CAMPAIGN VERIFIER: FAIL", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 1
    print("LANE-A CAMPAIGN VERIFIER: PASS — four frozen windows, one authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
