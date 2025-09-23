"""Helpers for invoking the run_tool CLI within tests."""

from __future__ import annotations

import sys
from typing import Mapping, NamedTuple

from metta.common.tool import run_tool as run_tool_module


class RunToolResult(NamedTuple):
    """Captured result from invoking the tool runner in-process."""

    returncode: int
    stdout: str
    stderr: str


def run_tool_in_process(
    *cli_args: str,
    monkeypatch,
    capsys,
    env_overrides: Mapping[str, str] | None = None,
    argv0: str = "tools/run.py",
) -> RunToolResult:
    """Invoke `metta.common.tool.run_tool.main()` without spawning a subprocess.

    Parameters
    ----------
    cli_args: str
        Arguments to pass to the CLI (after the entry point).
    monkeypatch: pytest.MonkeyPatch
        Pytest monkeypatch fixture, used to adjust sys modules and env vars.
    capsys: pytest.CaptureFixture[str]
        Capture fixture for stdout/stderr.
    env_overrides: Mapping[str, str] | None
        Optional environment variables to inject for the duration of the call.
    argv0: str
        Value to use for argv[0]; defaults to "tools/run.py" for consistency with scripts.
    """

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    if env_overrides:
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr(sys, "argv", [argv0, *cli_args], raising=False)

    returncode = run_tool_module.main()
    captured = capsys.readouterr()
    return RunToolResult(returncode, captured.out, captured.err)
