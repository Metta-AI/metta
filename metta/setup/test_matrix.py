from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Suite:
    name: str
    ci_cwd: str
    ci_args: tuple[str, ...]
    default_target: str | None
    env_gate: str | None = None
    allow_missing: bool = False

    @property
    def cwd_path(self) -> Path:
        return REPO_ROOT / self.ci_cwd

    @property
    def default_target_path(self) -> Path | None:
        if self.default_target is None:
            return None
        return REPO_ROOT / self.default_target


SUITES: tuple[Suite, ...] = (
    Suite(name="tests", ci_cwd=".", ci_args=("tests",), default_target="tests"),
    Suite(name="mettascope", ci_cwd=".", ci_args=("mettascope/tests",), default_target="mettascope/tests"),
    Suite(name="agent", ci_cwd="agent", ci_args=("tests",), default_target="agent/tests"),
    Suite(
        name="app_backend",
        ci_cwd="app_backend",
        ci_args=("tests",),
        default_target="app_backend/tests",
        env_gate="RUN_APP_BACKEND_TESTS",
    ),
    Suite(name="common", ci_cwd="common", ci_args=("tests",), default_target="common/tests"),
    Suite(name="codebot", ci_cwd="packages/codebot", ci_args=("tests",), default_target="packages/codebot/tests"),
    Suite(name="cogames", ci_cwd="packages/cogames", ci_args=("tests",), default_target="packages/cogames/tests"),
    Suite(
        name="gitta",
        ci_cwd="packages/gitta",
        ci_args=("tests",),
        default_target="packages/gitta/tests",
        allow_missing=True,
    ),
    Suite(
        name="mettagrid-python",
        ci_cwd="packages/mettagrid",
        ci_args=("tests",),
        default_target="packages/mettagrid/tests",
    ),
)


def _env_str_to_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    return lowered not in {"0", "false", "no", "off"} if default else lowered in {"1", "true", "yes", "on"}


def _should_skip_suite(suite: Suite, *, explicit_skip_app_backend: bool | None) -> bool:
    if suite.name == "app_backend":
        if explicit_skip_app_backend is True:
            return True
        if explicit_skip_app_backend is False:
            return False
        return not _env_str_to_bool(os.environ.get(suite.env_gate or ""), default=True)
    if suite.env_gate:
        return not _env_str_to_bool(os.environ.get(suite.env_gate), default=True)
    return False


def _run_command(args: Sequence[str], *, cwd: Path) -> int:
    print(f"→ {' '.join(args)} (cwd={cwd})")
    completed = subprocess.run(args, cwd=cwd, check=False)
    return completed.returncode


def _run_default(extra_pytest_args: Sequence[str]) -> int:
    workers = os.environ.get("METTA_TEST_WORKERS", "auto")
    targets: list[str] = []
    for suite in SUITES:
        if suite.default_target is None:
            continue
        path = suite.default_target_path
        if path is not None and not path.exists():
            if suite.allow_missing:
                print(f"↷ Skipping {suite.name} default target (missing {path})")
                continue
            print(f"✖ Default test target missing: {path}", file=sys.stderr)
            return 1
        targets.append(suite.default_target)

    if not targets:
        print("✖ No default test targets found", file=sys.stderr)
        return 1

    cmd = [
        "uv",
        "run",
        "pytest",
        *targets,
        "--benchmark-disable",
        "-n",
        workers,
        *extra_pytest_args,
    ]
    return _run_command(cmd, cwd=REPO_ROOT)


def _run_ci(
    *,
    extra_pytest_args: Sequence[str],
    explicit_skip_app_backend: bool | None,
) -> int:
    base_args = [
        "-n",
        os.environ.get("METTA_TEST_WORKERS", "4"),
        "--timeout=100",
        "--timeout-method=thread",
        "--benchmark-skip",
        "--maxfail=1",
        "--disable-warnings",
        "--durations=10",
        "-v",
    ]

    for suite in SUITES:
        if _should_skip_suite(suite, explicit_skip_app_backend=explicit_skip_app_backend):
            print(f"↷ Skipping {suite.name} suite")
            continue

        if not suite.cwd_path.exists():
            if suite.allow_missing:
                print(f"↷ Skipping {suite.name} suite (directory missing)")
                continue
            print(f"✖ Suite directory missing: {suite.cwd_path}", file=sys.stderr)
            return 1

        cmd = [
            "uv",
            "run",
            "pytest",
            *base_args,
            *suite.ci_args,
            *extra_pytest_args,
        ]

        exit_code = _run_command(cmd, cwd=suite.cwd_path)
        if exit_code != 0:
            return exit_code

    return 0


def run_preset(
    preset: str,
    *,
    extra_pytest_args: Sequence[str],
    skip_app_backend: bool | None = None,
) -> int:
    normalized = preset.lower()
    if normalized == "default":
        return _run_default(extra_pytest_args)
    if normalized == "ci":
        return _run_ci(
            extra_pytest_args=extra_pytest_args,
            explicit_skip_app_backend=skip_app_backend,
        )
    raise ValueError(f"Unknown preset '{preset}'")


def available_presets() -> tuple[str, ...]:
    return ("default", "ci")


app = typer.Typer(help="Unified entry point for Metta test presets.")


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(
    ctx: typer.Context,
    preset: str = typer.Option("default", "--preset", "-p", help="Test preset to run."),
    skip_app_backend: bool = typer.Option(
        False,
        "--skip-app-backend",
        help="Skip the app_backend test suite even if enabled via environment (ci preset).",
    ),
) -> None:
    extra_args = list(ctx.args or [])
    try:
        exit_code = run_preset(
            preset,
            extra_pytest_args=extra_args,
            skip_app_backend=True if skip_app_backend else None,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    raise typer.Exit(exit_code)


@app.command()
def list_presets() -> None:
    for preset in available_presets():
        typer.echo(preset)


if __name__ == "__main__":
    app()
