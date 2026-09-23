#!/usr/bin/env python3
"""Docs command smoke (repair-plan §16 / PR-B).

Verifies that the commands and extras documented in README/README_zh/docs
cannot silently drift from reality:

1. every ``openlithohub[...]`` extra referenced in documentation exists in
   ``[project.optional-dependencies]``;
2. every console script declared in ``[project.scripts]`` resolves to an
   installed executable;
3. documented serve flags (--host/--port/--workers/--log-level/--reload)
   and optimize flags (--model/--input/--output) actually exist.

Exit code 0 means docs and packaging agree; anything else fails the
``docs-command-smoke`` CI job.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # noqa: S404 - fixed argv probes
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]
DOC_GLOBS = ["README.md", "README_zh.md", "docs/**/*.md"]

# Documented flags per CLI surface (subset that docs actually lean on).
EXPECTED_FLAGS = {
    ("serve",): {"--host", "--port", "--workers", "--log-level", "--reload"},
    ("optimize", "run"): {"--model", "--input", "--output", "--node", "--writer"},
}

EXTRA_RE = re.compile(r"openlithohub\[([a-z0-9,\-_]+)\]")


def doc_files() -> list[Path]:
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        files.extend(sorted(REPO.glob(pattern)))
    return files


def main() -> int:
    failures: list[str] = []
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))

    extras = pyproject["project"]["optional-dependencies"]
    scripts = pyproject["project"].get("scripts", {})

    # 1. documented extras must exist
    referenced: set[str] = set()
    for path in doc_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in EXTRA_RE.finditer(text):
            for name in match.group(1).split(","):
                name = name.strip()
                if name:
                    referenced.add(name)
    for name in sorted(referenced):
        if name not in extras:
            failures.append(f"doc references openlithohub[{name}] but the extra does not exist")

    # 2. declared console scripts must resolve (PATH, or the interpreter's
    # bin dir when run as <venv>/bin/python without PATH setup). Deliberately
    # NOT resolve()d: venv python is a symlink into the base interpreter.
    def resolve_script(name: str) -> str | None:
        found = shutil.which(name)
        if found is None:
            candidate = Path(sys.executable).parent / name
            if candidate.is_file():
                found = str(candidate)
        return found

    for name in scripts:
        if resolve_script(name) is None:
            failures.append(f"console script {name!r} is not installed/resolvable on PATH")

    # 3. documented flags must exist on the real CLI
    for argv, expected in EXPECTED_FLAGS.items():
        cmd = [name for name, value in scripts.items() if value.endswith(".cli.app:app")]
        entry = cmd[0] if cmd else "openlithohub"
        entry_path = resolve_script(entry)
        if entry_path is None:
            failures.append(f"cannot probe CLI: {entry!r} not resolvable")
            continue
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [entry_path, *argv, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        help_text = proc.stdout + proc.stderr
        for flag in sorted(expected):
            if flag not in help_text:
                failures.append(f"documented flag {flag} missing from `{' '.join(argv)} --help`")

    if failures:
        print("docs-command-smoke FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"docs-command-smoke OK: {len(referenced)} documented extras exist, "
        f"{len(scripts)} console scripts resolve, "
        f"{sum(len(v) for v in EXPECTED_FLAGS.values())} documented flags verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
