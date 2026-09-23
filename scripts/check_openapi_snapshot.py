"""PR-D §10: deterministic OpenAPI v1 snapshot check.

The checked-in ``docs/api/openapi-v1.json`` is the API authority: a PR
that changes the schema (field added/removed/renamed, error envelope,
response models) must regenerate it in the same PR so the diff is
reviewable. CI fails on any drift.

Usage::

    python scripts/check_openapi_snapshot.py            # verify
    python scripts/check_openapi_snapshot.py --update   # regenerate

Determinism contract: the generated document is normalized (sorted keys,
stable indentation) and contains no timestamps or environment-specific
values — ``create_app()`` is side-effect free and the schema version comes
from the single ``API_SCHEMA_VERSION`` authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SNAPSHOT = Path("docs/api/openapi-v1.json")


def generate_openapi() -> dict:
    """Build the app and render its normalized OpenAPI document."""
    from openlithohub.server.app import create_app
    from openlithohub.server.config import ServerConfig

    app = create_app(ServerConfig())
    return app.openapi()


def normalize(document: dict) -> str:
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="regenerate the snapshot")
    args = parser.parse_args()

    rendered = normalize(generate_openapi())

    if args.update:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
        print(f"openapi snapshot updated: {SNAPSHOT}")
        return 0

    if not SNAPSHOT.exists():
        print(f"FAIL: snapshot missing: {SNAPSHOT}", file=sys.stderr)
        return 1

    checked_in = SNAPSHOT.read_text()
    if rendered == checked_in:
        print(f"openapi snapshot OK ({SNAPSHOT})")
        return 0

    print(
        f"FAIL: OpenAPI drift detected.\n"
        f"The generated schema differs from {SNAPSHOT}.\n"
        f"If this change is intentional, run "
        f"`python scripts/check_openapi_snapshot.py --update` and commit "
        f"the snapshot in the same PR (docs/server-api-contract.md).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
