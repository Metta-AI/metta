import os
import subprocess
from typing import Annotated, Literal

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

app = typer.Typer(
    help="MettaGrid C++ test runner",
    invoke_without_command=True,
)


def _run_target(target: str, *, verbose: bool) -> None:
    mettagrid_dir = get_repo_root() / "packages" / "mettagrid"
    env = os.environ.copy()
    if verbose:
        env["VERBOSE"] = "1"
    cmd = ["make", target]
    info(f"Running MettaGrid {target}...")
    exit_code = subprocess.run(cmd, cwd=mettagrid_dir, env=env, check=False).returncode
    if exit_code != 0:
        error(f"C++ {target} failed!")
        raise typer.Exit(exit_code)
    success(f"C++ {target} completed successfully!")


@app.callback()
def command(
    target: Annotated[
        Literal["test", "coverage", "benchmark"],
        typer.Argument(help="Target to run."),
    ] = "test",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    _run_target(target, verbose=verbose)


@app.command(help="Run MettaGrid C++ unit tests")
def test(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _run_target("test", verbose=verbose)


@app.command(help="Run MettaGrid C++ coverage target")
def coverage(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _run_target("coverage", verbose=verbose)


@app.command(help="Run MettaGrid C++ benchmark target")
def benchmark(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _run_target("benchmark", verbose=verbose)
