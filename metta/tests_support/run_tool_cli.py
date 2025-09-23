"""Utilities for invoking the run_tool CLI inside pytest."""

import sys
from typing import Mapping, NamedTuple

import pytest

from metta.common.tool import run_tool as run_tool_module


class RunToolResult(NamedTuple):
    """Result bundle returned by the run_tool_cli fixture."""

    returncode: int
    stdout: str
    stderr: str


@pytest.fixture
def run_tool_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Invoke metta.common.tool.run_tool.main() in-process and capture output."""

    def _run_tool_cli(
        *cli_args: str,
        env_overrides: Mapping[str, str] | None = None,
        argv0: str = "tools/run.py",
    ) -> RunToolResult:
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
        if env_overrides:
            for key, value in env_overrides.items():
                monkeypatch.setenv(key, value)
        monkeypatch.setattr(sys, "argv", [argv0, *cli_args], raising=False)
        returncode = run_tool_module.main()
        captured = capsys.readouterr()
        return RunToolResult(returncode, captured.out, captured.err)

    return _run_tool_cli
