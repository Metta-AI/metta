import subprocess
from typing import Optional

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

app = typer.Typer(
    help="MettaGrid Nim test runner",
    invoke_without_command=True,
)


@app.callback()
def command(
    filter: Optional[str] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter test files by substring match (e.g. 'protocol' matches test_protocol_*.nim)",
    ),
) -> None:
    tests_dir = get_repo_root() / "packages" / "mettagrid" / "nim" / "mettascope" / "tests"
    test_files = list(tests_dir.glob("test_*.nim"))
    if filter:
        test_files = [f for f in test_files if filter in f.name]
    if not test_files:
        error(f"No test files found matching filter: {filter}")
        raise typer.Exit(1)
    for test_file in sorted(test_files):
        info(f"Running {test_file}...")
        exit_code = subprocess.run(["nim", "r", str(test_file)], check=False).returncode
        if exit_code != 0:
            error(f"Nim {test_file} failed!")
            raise typer.Exit(exit_code)
        success(f"Nim {test_file} completed successfully!")
