import subprocess
import sys
from pathlib import Path
from typing import Sequence

import typer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from metta.common.util.fs import get_repo_root


class _Config(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    METTA_TEST_WORKERS: int | str | None = Field(default=None, description="Number of test workers to use.")
    RUN_APP_BACKEND_TESTS: bool = Field(default=True, description="Whether to run the app_backend tests.")


test_config = _Config()


class Suite(BaseModel):
    name: str
    ci_cwd: str
    ci_args: tuple[str, ...]
    default_target: str | None
    allow_missing: bool = False

    @property
    def cwd_path(self) -> Path:
        return get_repo_root() / self.ci_cwd

    @property
    def default_target_path(self) -> Path | None:
        if self.default_target is None:
            return None
        return get_repo_root() / self.default_target


SUITES: tuple[Suite, ...] = (
    Suite(name="tests", ci_cwd=".", ci_args=("tests",), default_target="tests"),
    Suite(name="mettascope", ci_cwd=".", ci_args=("mettascope/tests",), default_target="mettascope/tests"),
    Suite(name="agent", ci_cwd="agent", ci_args=("tests",), default_target="agent/tests"),
    Suite(name="app_backend", ci_cwd="app_backend", ci_args=("tests",), default_target="app_backend/tests"),
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


def _run_command(args: Sequence[str], *, cwd: Path) -> int:
    print(f"→ {' '.join(args)} (cwd={cwd})")
    completed = subprocess.run(args, cwd=cwd, check=False)
    return completed.returncode


def _run_default(*args: Sequence[str]) -> int:
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
        test_config.METTA_TEST_WORKERS or "auto",
        *args,
    ]
    return _run_command(cmd, cwd=get_repo_root())


def _run_ci(*args: Sequence[str]) -> int:
    base_args = [
        "-n",
        test_config.METTA_TEST_WORKERS or "4",
        "--timeout=100",
        "--timeout-method=thread",
        "--benchmark-skip",
        "--maxfail=1",
        "--disable-warnings",
        "--durations=10",
        "-v",
    ]

    for suite in SUITES:
        if test_config.RUN_APP_BACKEND_TESTS and suite.name == "app_backend":
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
            *args,
        ]

        exit_code = _run_command(cmd, cwd=suite.cwd_path)
        if exit_code != 0:
            return exit_code

    return 0


app = typer.Typer(help="Unified entry point for Metta test presets.")


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(
    ctx: typer.Context,
    preset: str = typer.Option("default", "--preset", "-p", help="Test preset to run."),
) -> None:
    extra_args = list(ctx.args or [])
    normalized = preset.lower()
    if normalized == "default":
        _run_default(*extra_args)
    elif normalized == "ci":
        _run_ci(*extra_args)
    else:
        print(f"Unknown preset '{preset}'", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
