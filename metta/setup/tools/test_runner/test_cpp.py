import os
import subprocess
import typing

import typer

import metta.common.util.fs
import metta.setup.utils

app = typer.Typer(
    help="MettaGrid C++ test runner",
    invoke_without_command=True,
)


def _run_target(target: str, *, verbose: bool) -> None:
    mettagrid_dir = metta.common.util.fs.get_repo_root() / "packages" / "mettagrid"
    env = os.environ.copy()
    if verbose:
        env["VERBOSE"] = "1"
    cmd = ["make", target]
    metta.setup.utils.info(f"Running MettaGrid {target}...")
    exit_code = subprocess.run(cmd, cwd=mettagrid_dir, env=env, check=False).returncode
    if exit_code != 0:
        metta.setup.utils.error(f"C++ {target} failed!")
        raise typer.Exit(exit_code)
    metta.setup.utils.success(f"C++ {target} completed successfully!")


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
    verbose: typing.Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    targets = {k: v for k, v in [("benchmark", benchmark), ("test", test)] if v}
    for target in targets.keys() or ["test"]:
        _run_target(target, verbose=verbose)
