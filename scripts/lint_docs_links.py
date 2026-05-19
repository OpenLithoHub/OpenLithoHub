"""Fail CI if a Markdown link in docs/ resolves to a path outside docs/.

Internal cross-page links inside docs/ (e.g. ``../leaderboard-submission.md``
from ``docs/announcements/2026-05-launch.md``) are fine — mkdocs resolves
them. The class of bug we want to catch is the README_zh.md ``../LICENSE``
case: a link that escapes the docs root, which mkdocs-strict only flags
once the page is added to nav.

Run from the repo root: ``python scripts/lint_docs_links.py``.
Exits non-zero with a list of offending file:line:link if any are found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Match Markdown inline links: [text](target) and image links ![alt](target).
# We deliberately ignore reference-style links (rare in this repo) and
# fenced code blocks aren't stripped — the lint is intentionally simple
# and the false-positive surface is tiny because we only care about
# escaping ``docs/``.
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _link_target_escapes_docs(source: Path, target: str, docs_root: Path) -> bool:
    # Skip absolute URLs, anchors, and mailto: — we only resolve
    # relative paths that could climb above docs/.
    if target.startswith(("http://", "https://", "mailto:", "#", "/")):
        return False
    # Strip in-page anchor and query string before resolution.
    clean = target.split("#", 1)[0].split("?", 1)[0]
    if not clean:
        return False
    resolved = (source.parent / clean).resolve()
    try:
        resolved.relative_to(docs_root.resolve())
    except ValueError:
        return True
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    docs_root = repo_root / "docs"
    if not docs_root.is_dir():
        print(f"docs/ not found at {docs_root}", file=sys.stderr)
        return 1

    offenders: list[str] = []
    for md in docs_root.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _LINK_RE.finditer(line):
                target = match.group(1)
                if _link_target_escapes_docs(md, target, docs_root):
                    rel = md.relative_to(repo_root)
                    offenders.append(f"{rel}:{lineno}: link target {target!r} escapes docs/")

    if offenders:
        print("docs link lint: links escaping docs/ are forbidden", file=sys.stderr)
        print("(use absolute https://github.com/... URLs instead)", file=sys.stderr)
        for o in offenders:
            print(f"  {o}", file=sys.stderr)
        return 1
    n = sum(1 for _ in docs_root.rglob("*.md"))
    print(f"docs link lint: no escapes found across {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
