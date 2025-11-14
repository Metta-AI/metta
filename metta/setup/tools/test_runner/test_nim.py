import subprocess

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

app = typer.Typer(
    help="MettaGrid Nim test runner",
    invoke_without_command=True,
)


@app.callback()
def command() -> None:
    test_files = (get_repo_root() / "packages" / "mettagrid" / "nim" / "mettascope" / "tests").glob("test_*.nim")
    for test_file in test_files:
        info(f"Running {test_file}...")
        exit_code = subprocess.run(["nim", "r", str(test_file)], check=False).returncode
        if exit_code != 0:
            error(f"Nim {test_file} failed!")
            raise typer.Exit(exit_code)
        success(f"Nim {test_file} completed successfully!")
