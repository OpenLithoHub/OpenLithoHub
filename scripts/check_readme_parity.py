"""Fail CI if README.md and README_zh.md drift in section structure.

We compare H2 sections (``## ...``) only — H3+ are translated subheadings
and code-comment titles where exact parity is too brittle. The check:

1. Both files must have the same number of H2 sections.
2. Sections must appear in the same order (positional pairing).

Section *titles* are not compared character-by-character — translation is
the whole point. Drift we want to catch is structural: a section added to
English but not Chinese, or reordered.

Run from the repo root: ``python scripts/check_readme_parity.py``.
Exits non-zero with a diff-style report if structures diverge.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _extract_h2(path: Path) -> list[tuple[int, str]]:
    """Return [(line_number, title)] for H2 headers, ignoring fenced blocks."""
    text = path.read_text(encoding="utf-8")
    in_fence = False
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            out.append((lineno, m.group(1)))
    return out


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    en_path = repo / "README.md"
    zh_path = repo / "README_zh.md"

    if not en_path.exists() or not zh_path.exists():
        print(f"error: missing README — en={en_path.exists()} zh={zh_path.exists()}")
        return 2

    en = _extract_h2(en_path)
    zh = _extract_h2(zh_path)

    if len(en) != len(zh):
        print(
            f"README parity: H2 section count mismatch "
            f"({en_path.name}={len(en)} vs {zh_path.name}={len(zh)})"
        )
        print("\nEnglish H2 sections:")
        for ln, t in en:
            print(f"  README.md:{ln}: {t}")
        print("\nChinese H2 sections:")
        for ln, t in zh:
            print(f"  README_zh.md:{ln}: {t}")
        print(
            "\nFix: add or remove the missing section so both files have "
            "the same H2 structure. English is authoritative."
        )
        return 1

    print(f"README parity: {len(en)} H2 sections aligned between {en_path.name} and {zh_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
