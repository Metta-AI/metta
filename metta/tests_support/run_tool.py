"""Helpers for invoking the run_tool CLI within tests."""

from __future__ import annotations

import logging
import os
import signal
import sys
import warnings
from typing import Mapping, NamedTuple

from metta.common.tool import run_tool as run_tool_module
from metta.common.util import log_config as log_config_module


class _LoggerState(NamedTuple):
    """Snapshot of a logger's mutable state for restoration."""

    handlers: tuple[logging.Handler, ...]
    filters: tuple[logging.Filter, ...]
    level: int
    propagate: bool
    disabled: bool


def _capture_logger_state(logger: logging.Logger) -> _LoggerState:
    """Capture handlers, filters, and flags from a logger."""

    return _LoggerState(
        handlers=tuple(logger.handlers),
        filters=tuple(logger.filters),
        level=logger.level,
        propagate=logger.propagate,
        disabled=logger.disabled,
    )


def _restore_logger_state(logger: logging.Logger, state: _LoggerState) -> None:
    """Restore logger handlers, filters, and flags."""

    current_handlers = list(logger.handlers)
    for handler in current_handlers:
        if handler not in state.handlers:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    for handler in state.handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)

    logger.setLevel(state.level)
    logger.propagate = state.propagate
    logger.disabled = state.disabled
    logger.filters.clear()
    logger.filters.extend(state.filters)


def _restore_environment(snapshot: Mapping[str, str]) -> None:
    """Restore environment variables to a prior snapshot."""

    current_keys = set(os.environ.keys())
    snapshot_keys = set(snapshot.keys())

    for key in current_keys - snapshot_keys:
        os.environ.pop(key, None)

    for key in snapshot_keys:
        value = snapshot[key]
        if os.environ.get(key) != value:
            os.environ[key] = value


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
    log_config_module._init_console_logging.cache_clear()
    log_config_module._add_file_logging.cache_clear()

    env_snapshot = dict(os.environ)
    root_snapshot = _capture_logger_state(logging.getLogger())
    httpx_snapshot = _capture_logger_state(logging.getLogger("httpx"))
    original_logger_class = logging.getLoggerClass()
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    returncode: int | None = None
    raised: BaseException | None = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            try:
                returncode = run_tool_module.main()
            except SystemExit as exc:  # pragma: no cover - defensive
                code = exc.code
                if isinstance(code, int) or code is None:
                    returncode = int(code or 0)
                else:
                    try:
                        returncode = int(code)
                    except (TypeError, ValueError):
                        returncode = 1
            except BaseException as exc:  # pragma: no cover - propagate after cleanup
                raised = exc
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        logging.setLoggerClass(original_logger_class)
        _restore_logger_state(logging.getLogger(), root_snapshot)
        _restore_logger_state(logging.getLogger("httpx"), httpx_snapshot)
        _restore_environment(env_snapshot)

    captured = capsys.readouterr()

    if raised is not None:
        raise raised

    if returncode is None:  # pragma: no cover - sanity check
        raise RuntimeError("run_tool_module.main() did not produce a return code")

    return RunToolResult(returncode, captured.out, captured.err)
