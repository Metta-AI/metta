import os
import subprocess
from typing import Annotated, Literal

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

app = typer.Typer(help="MettaGrid C++ test runner")


@app.command(help="Run MettaGrid C++ unit tests")
def command(
    target: Annotated[Literal["test", "coverage", "benchmark"], typer.Argument(help="Target to run.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
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
