import subprocess
from pathlib import Path
from typing import Sequence

import typer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, warning


class _Config(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    METTA_TEST_WORKERS: int | str | None = Field(default=None, description="Number of test workers to use.")
    RUN_APP_BACKEND_TESTS: bool = Field(default=True, description="Whether to run the app_backend tests.")


test_config = _Config()


class Suite(BaseModel):
    name: str
    target: str

    @property
    def target_path(self) -> Path:
        return get_repo_root() / self.target

    @property
    def target_enclosing_dir(self) -> Path:
        return self.target_path.parent


SUITES: tuple[Suite, ...] = (
    Suite(name="tests", target="tests"),
    Suite(name="mettascope", target="mettascope/tests"),
    Suite(name="agent", target="agent/tests"),
    Suite(name="app_backend", target="app_backend/tests"),
    Suite(name="common", target="common/tests"),
    Suite(name="codebot", target="packages/codebot/tests"),
    Suite(name="cogames", target="packages/cogames/tests"),
    Suite(name="gitta", target="packages/gitta/tests"),
    Suite(name="mettagrid", target="packages/mettagrid/tests"),
)


def _run_command(args: Sequence[str], *, cwd: Path) -> int:
    info(f"→ {' '.join(args)} (cwd={cwd})")
    completed = subprocess.run(args, cwd=cwd, check=False)
    return completed.returncode


def _run_default(*args: Sequence[str]) -> int:
    for suite in SUITES:
        if not suite.target_path.exists():
            error(f"Test suite directory missing: {suite.target_path}")
            return 1

    cmd = [
        "uv",
        "run",
        "pytest",
        *[str(suite.target) for suite in SUITES],
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
        if not test_config.RUN_APP_BACKEND_TESTS and suite.name == "app_backend":
            info(f"Skipping {suite.name} suite")
            continue

        if not suite.target_path.exists():
            warning(f"Suite directory missing: {suite.target_path}")
            return 1

        cmd = [
            "uv",
            "run",
            "pytest",
            *base_args,
            str(suite.target_path),
            *args,
        ]

        exit_code = _run_command(cmd, cwd=suite.target_enclosing_dir)
        if exit_code != 0:
            return exit_code

    return 0


app = typer.Typer(help="Python test runner")


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
        warning(f"Unknown preset '{preset}'")
        raise typer.Exit(1)
