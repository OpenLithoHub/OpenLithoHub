"""IPython magic commands for OpenLithoHub."""

from __future__ import annotations

from typing import Any

try:
    from IPython.core.magic import Magics, line_magic, magics_class
except ImportError:
    # Provide stubs when IPython is not available
    class Magics:  # type: ignore[no-redef]
        def __init__(self, shell: Any = None) -> None:
            pass

    def magics_class(cls: Any) -> Any:  # type: ignore[no-redef]
        return cls

    def line_magic(func: Any) -> Any:  # type: ignore[no-redef]
        return func


@magics_class
class OpenLithoHubMagics(Magics):
    """IPython magic commands for quick lithography evaluation."""

    @line_magic
    def openlithohub(self, line: str) -> None:
        """Run openlithohub CLI commands inline.

        Usage:
            %openlithohub eval --model dummy-identity --dataset lithobench --data-root ./data
        """
        import shlex
        import subprocess
        import sys

        args = shlex.split(line)
        cmd = [sys.executable, "-m", "openlithohub.cli.app"] + args
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"\n[Exit code: {result.returncode}]")

    @line_magic
    def litho_eval(self, line: str) -> None:
        """Shorthand for %openlithohub eval ..."""
        self.openlithohub(f"eval {line}")
