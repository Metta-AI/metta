#!/usr/bin/env -S uv run
import subprocess
import sys
from pathlib import Path

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, prompt_choice, success


class BookCommands:
    def __init__(self):
        self.repo_root = get_repo_root()
        self.marimo_dir = self.repo_root / "experiments" / "marimo"

    def _get_notebooks(self) -> list[Path]:
        """Get all marimo notebooks (.py files) in experiments/marimo."""
        if not self.marimo_dir.exists():
            self.marimo_dir.mkdir(parents=True, exist_ok=True)
        return sorted([f for f in self.marimo_dir.glob("*.py") if not f.name.startswith("__")])

    def _select_notebook(self) -> Path | None:
        """Interactive notebook selection."""
        notebooks = self._get_notebooks()

        if not notebooks:
            info("No notebooks found in experiments/marimo/")
            return None

        choices = [(nb, nb.name) for nb in notebooks]
        selected = prompt_choice("Select a notebook:", choices)

        if selected:
            info(f"\n→ Selected: {selected.name}\n")

        return selected

    def _run_marimo(self, cmd: list[str]) -> None:
        """Run marimo command from experiments/marimo directory."""
        try:
            subprocess.run(cmd, cwd=self.marimo_dir, check=True)
        except subprocess.CalledProcessError as e:
            error(f"Marimo command failed: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            info("\nMarimo shutdown")
            sys.exit(0)

    def cmd_home(self) -> None:
        """Run marimo edit to open the home page."""
        info("Launching marimo home page...")
        self._run_marimo(["marimo", "edit"])

    def cmd_open(self, filename: str | None = None) -> None:
        """Open a specific notebook or show interactive menu."""
        if filename:
            notebook_path = self.marimo_dir / filename
            if not notebook_path.exists():
                error(f"Notebook not found: {filename}")
                sys.exit(1)
            info(f"Opening {filename}...")
            self._run_marimo(["marimo", "edit", filename])
        else:
            # Interactive selection
            notebook = self._select_notebook()
            if notebook:
                info(f"Launching notebook: {notebook.name}...")
                self._run_marimo(["marimo", "edit", notebook.name])

    def cmd_run(self, filename: str | None = None) -> None:
        """Run a notebook in read-only mode."""
        if filename:
            notebook_path = self.marimo_dir / filename
            if not notebook_path.exists():
                error(f"Notebook not found: {filename}")
                sys.exit(1)
            info(f"Running {filename}...")
            self._run_marimo(["marimo", "run", filename])
        else:
            # Interactive selection
            notebook = self._select_notebook()
            if notebook:
                info(f"Running notebook in read-only mode: {notebook.name}...")
                self._run_marimo(["marimo", "run", notebook.name])

    def cmd_new(self, name: str | None = None) -> None:
        """Create a new notebook in experiments/marimo."""
        if not name:
            name = input("Enter notebook name (without .py extension): ").strip()

        if not name:
            error("Notebook name is required")
            sys.exit(1)

        # Ensure .py extension
        if not name.endswith(".py"):
            name = f"{name}.py"

        notebook_path = self.marimo_dir / name
        if notebook_path.exists():
            error(f"Notebook already exists: {name}")
            sys.exit(1)

        info(f"Creating new notebook: {name}")
        self._run_marimo(["marimo", "new", name])
        success(f"Created {name}")

    def main(self, argv: list[str] | None = None) -> None:
        """Main entry point for book commands."""
        if not argv:
            # Show interactive menu
            actions = [
                ("home", "Open marimo home"),
                ("open", "Open an existing notebook"),
                ("run", "Run a notebook (read-only)"),
                ("new", "Create a new notebook"),
            ]

            action = prompt_choice("Select an action:", actions)

            # Show selected action
            action_map = dict(actions)
            info(f"\n→ Selected: {action_map[action]}\n")

            if action == "home":
                self.cmd_home()
            elif action == "open":
                self.cmd_open()
            elif action == "run":
                self.cmd_run()
            elif action == "new":
                self.cmd_new()
        else:
            # Parse command line
            cmd = argv[0] if argv else None
            args = argv[1:] if len(argv) > 1 else []

            if cmd == "home":
                self.cmd_home()
            elif cmd == "open":
                self.cmd_open(args[0] if args else None)
            elif cmd == "run":
                self.cmd_run(args[0] if args else None)
            elif cmd == "new":
                self.cmd_new(args[0] if args else None)
            else:
                error(f"Unknown command: {cmd}")
                info("Available commands: home, open, run, new")
                sys.exit(1)
