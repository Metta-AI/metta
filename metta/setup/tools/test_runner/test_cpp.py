import os
import subprocess
from typing import Annotated

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
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help="Run benchmark tests.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Run unit tests.",
    ),
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    targets = {k: v for k, v in [("benchmark", benchmark), ("test", test)] if v}
    for target in targets.keys() or ["test"]:
        _run_target(target, verbose=verbose)
