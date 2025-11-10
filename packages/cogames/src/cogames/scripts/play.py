"""Backward-compatible entry point for ``python -m cogames.scripts.play``.

This module simply forwards to ``cogames play`` so that older workflows and
docs keep working while the main Typer CLI remains the single source of truth.
"""

from __future__ import annotations

import sys
from typing import Iterable

from typer.main import get_command

from cogames.main import app as cogames_app


def main(argv: Iterable[str] | None = None) -> None:
    """Invoke the Typer CLI's ``play`` sub-command.

    Args:
        argv: Optional custom argv list. When ``None`` we forward ``sys.argv``.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    cli = get_command(cogames_app)
    play_command = cli.commands.get("play")
    if play_command is None:  # pragma: no cover - defensive guard
        raise RuntimeError("'play' command missing from cogames CLI")
    play_command.main(
        args=args,
        prog_name="python -m cogames.scripts.play",
        standalone_mode=True,
    )


if __name__ == "__main__":  # pragma: no cover - CLI shim
    main()
