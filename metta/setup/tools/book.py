#!/usr/bin/env -S uv run
import pathlib
import subprocess
import typing

import rich.console
import typer

import metta.common.util.fs
import metta.setup.utils

console = rich.console.Console()
app = typer.Typer(
    help="Interactive marimo notebook commands",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

repo_root = metta.common.util.fs.get_repo_root()
marimo_dir = repo_root / "notebooks" / "marimo"


def _get_notebooks() -> list[pathlib.Path]:
    if not marimo_dir.exists():
        marimo_dir.mkdir(parents=True, exist_ok=True)
    return sorted([f for f in marimo_dir.glob("*.py") if not f.name.startswith("__")])


def _select_notebook() -> pathlib.Path | None:
    notebooks = _get_notebooks()

    if not notebooks:
        metta.setup.utils.info("No notebooks found in notebooks/marimo/")
        return None

    choices = [(nb, nb.name) for nb in notebooks]
    selected = metta.setup.utils.prompt_choice("Select a notebook:", choices)

    if selected:
        metta.setup.utils.info(f"\n→ Selected: {selected.name}\n")

    return selected


def _run_marimo(cmd: list[str]):
    try:
        subprocess.run(cmd, cwd=marimo_dir, check=True)
    except subprocess.CalledProcessError as e:
        metta.setup.utils.error(f"Marimo command failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        metta.setup.utils.info("\nMarimo shutdown")
        raise typer.Exit(0) from None


@app.command(name="home")
def cmd_home():
    """Open marimo home page."""
    metta.setup.utils.info("Launching marimo home page...")
    _run_marimo(["marimo", "edit"])


@app.command(name="open")
def cmd_open(filename: typing.Annotated[str | None, typer.Argument(help="Notebook filename")] = None):
    """Open an existing notebook."""
    if filename:
        notebook_path = marimo_dir / filename
        if not notebook_path.exists():
            metta.setup.utils.error(f"Notebook not found: {filename}")
            raise typer.Exit(1)
        metta.setup.utils.info(f"Opening {filename}...")
        _run_marimo(["marimo", "edit", filename])
    else:
        notebook = _select_notebook()
        if notebook:
            metta.setup.utils.info(f"Launching notebook: {notebook.name}...")
            _run_marimo(["marimo", "edit", notebook.name])


@app.command(name="run")
def cmd_run(filename: typing.Annotated[str | None, typer.Argument(help="Notebook filename")] = None):
    """Run a notebook in read-only mode."""
    if filename:
        notebook_path = marimo_dir / filename
        if not notebook_path.exists():
            metta.setup.utils.error(f"Notebook not found: {filename}")
            raise typer.Exit(1)
        metta.setup.utils.info(f"Running {filename}...")
        _run_marimo(["marimo", "run", filename])
    else:
        notebook = _select_notebook()
        if notebook:
            metta.setup.utils.info(f"Running notebook in read-only mode: {notebook.name}...")
            _run_marimo(["marimo", "run", notebook.name])


@app.command(name="new")
def cmd_new(name: typing.Annotated[str | None, typer.Argument(help="Notebook name")] = None):
    """Create a new notebook."""
    if not name:
        name = typer.prompt("Enter notebook name (without .py extension)")

    if not name:
        metta.setup.utils.error("Notebook name is required")
        raise typer.Exit(1)

    if not name.endswith(".py"):
        name = f"{name}.py"

    notebook_path = marimo_dir / name
    if notebook_path.exists():
        metta.setup.utils.error(f"Notebook already exists: {name}")
        raise typer.Exit(1)

    metta.setup.utils.info(f"Creating new notebook: {name}")
    _run_marimo(["marimo", "new", name])
    metta.setup.utils.success(f"Created {name}")


@app.command(name="list")
def cmd_list():
    """List all available notebooks."""
    notebooks = _get_notebooks()
    if not notebooks:
        metta.setup.utils.info("No notebooks found in notebooks/marimo/")
    else:
        console.print("\n[bold]Available notebooks:[/bold]")
        for nb in notebooks:
            console.print(f"  • {nb.name}")
        console.print()


def main():
    app()


if __name__ == "__main__":
    main()
