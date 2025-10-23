from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Iterable, Sequence

import typer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info


class _Config(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    METTA_TEST_WORKERS: str | None = Field(default=None, description="Number of test workers to use.")


test_config = _Config()


class Package(BaseModel):
    name: str
    target: Path

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def target_path(self) -> Path:
        root = get_repo_root()
        if self.target.is_absolute():
            return self.target
        return root / self.target


PACKAGES: tuple[Package, ...] = (
    Package(name="tests", target=Path("tests")),
    Package(name="mettascope", target=Path("mettascope/tests")),
    Package(name="agent", target=Path("agent/tests")),
    Package(name="app_backend", target=Path("app_backend/tests")),
    Package(name="common", target=Path("common/tests")),
    Package(name="codebot", target=Path("packages/codebot/tests")),
    Package(name="cogames", target=Path("packages/cogames/tests")),
    Package(name="gitta", target=Path("packages/gitta/tests")),
    Package(name="mettagrid", target=Path("packages/mettagrid/tests")),
    Package(name="cortex", target=Path("packages/cortex/tests")),
)


DEFAULT_FLAGS: tuple[str, ...] = ("--benchmark-disable",)
CI_FLAGS: tuple[str, ...] = (
    "--timeout=100",
    "--timeout-method=thread",
    "--benchmark-skip",
    "--maxfail=1",
    "--disable-warnings",
    "--durations=10",
    "-v",
)
TESTMON_FLAGS: tuple[str, ...] = ("--testmon", "--testmon-label=staged")


def _run_command(args: Sequence[str]) -> int:
    info(f"→ {' '.join(args)}")
    completed = subprocess.run(args, cwd=get_repo_root(), check=False)
    return completed.returncode


def _resolve_package_targets(
    include: Iterable[str],
    exclude: Iterable[str],
) -> list[Package]:
    package_map = {package.key: package for package in PACKAGES}
    include_keys = [name.lower() for name in include]
    exclude_keys = {name.lower() for name in exclude}

    selected: list[Package]
    if include_keys:
        missing = [name for name in include_keys if name not in package_map]
        if missing:
            raise ValueError(f"Unknown suite(s): {', '.join(missing)}")
        selected = [package_map[name] for name in include_keys]
    else:
        selected = list(PACKAGES)

    if exclude_keys:
        selected = [package for package in selected if package.key not in exclude_keys]

    return selected


def _collect_package_targets(packages: Iterable[Package]) -> list[str]:
    targets: list[Path] = []
    for package in packages:
        path = package.target_path
        if not path.exists():
            raise ValueError(f"Test package directory missing: {path}")
        targets.append(path)
    if not targets:
        raise ValueError("No package targets resolved.")
    return [str(t.relative_to(get_repo_root())) for t in targets]


app = typer.Typer(
    help="Python test runner",
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)


@app.callback()
def run(
    ctx: typer.Context,
    targets: Annotated[list[str] | None, typer.Argument(help="Explicit pytest targets.")] = None,
    ci: bool = typer.Option(False, "--ci", help="Use CI-style settings and parallel suite execution."),
    packages: Annotated[
        list[str] | None, typer.Option("--package", "-p", help="Limit to specific named package(s).")
    ] = None,
    skip_packages: Annotated[
        list[str] | None, typer.Option("--skip-package", help="Exclude package(s) by name.")
    ] = None,
    changed: bool = typer.Option(
        False,
        "--changed",
        help="Run only tests impacted by staged/changed files using pytest-testmon.",
    ),
) -> None:
    extra_args = list(ctx.args or [])
    target_args = targets or []
    package_args = packages or []
    skip_package_args = skip_packages or []

    cmd = ["uv", "run", "pytest"]
    if target_args:
        if package_args or skip_package_args or changed or ci:
            error("Explicit targets cannot be combined with suite filters, --changed, or --ci.")
            raise typer.Exit(1)
        exit_code = _run_command([*cmd, *target_args, *extra_args])
        raise typer.Exit(exit_code)

    try:
        selected = _resolve_package_targets(package_args, skip_package_args)
        resolved_targets = _collect_package_targets(selected)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc

    if ci:
        cmd.extend(["-n", test_config.METTA_TEST_WORKERS or "4", *CI_FLAGS])
        cmd.extend(extra_args)

        exit_code = 0
        with ThreadPoolExecutor(max_workers=len(resolved_targets)) as pool:
            for code in pool.map(lambda path: _run_command([*cmd, path]), resolved_targets):
                exit_code = max(exit_code, code)
        raise typer.Exit(exit_code)
    else:
        cmd.extend(["-n", test_config.METTA_TEST_WORKERS or "auto", *DEFAULT_FLAGS])
        if changed:
            cmd.extend(TESTMON_FLAGS)
        cmd.extend(extra_args)
        cmd.extend(resolved_targets)
        raise typer.Exit(_run_command(cmd))
