#!/usr/bin/env -S uv run
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success, warning

console = Console()
app = typer.Typer(
    help="Symlink setup for global metta command",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

repo_root = get_repo_root()
wrapper_script = repo_root / "metta" / "setup" / "installer" / "bin" / "metta"
local_bin = Path.home() / ".local" / "bin"
target_symlink = local_bin / "metta"


def _local_bin_is_in_path() -> bool:
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    return str(local_bin) in path_dirs


def _check_existing_metta() -> Optional[str]:
    if not target_symlink.exists():
        return None

    if target_symlink.is_symlink():
        target = target_symlink.resolve()
        if target == wrapper_script.resolve():
            return "ours"

    return "other"


def setup_path(force: bool = False) -> None:
    wrapper_script.chmod(0o755)

    try:
        local_bin.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error(f"Failed to create {local_bin}: {e}")
        info("Please create this directory manually and re-run the installer.")
        return

    existing = _check_existing_metta()

    if existing == "ours":
        success("metta is already installed and linked correctly.")
        return
    elif existing == "other":
        if force:
            info(f"Replacing existing metta command at {target_symlink}")
            try:
                target_symlink.unlink()
            except Exception as e:
                error(f"Failed to remove existing file: {e}")
                return
        else:
            warning(f"A 'metta' command already exists at {target_symlink}")
            info("Not overwriting existing command.")
            info("Run with --force if you want to replace it.")
            return

    try:
        target_symlink.symlink_to(wrapper_script)
        success(f"Created symlink: {target_symlink} → {wrapper_script}")
    except Exception as e:
        error(f"Failed to create symlink: {e}")
        info("You can create it manually with:")
        info(f"  ln -s {wrapper_script} {target_symlink}")
        return

    if _local_bin_is_in_path():
        success("metta is now available globally!")
        info("You can run: metta --help")
    else:
        info("")
        warning(f"{local_bin} is not in your PATH")
        info("To use metta globally, add this to your shell profile:")
        info('  export PATH="$HOME/.local/bin:$PATH"')
        info("")
        info("For example:")
        info("  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc")
        info("  source ~/.bashrc")


@app.command(name="setup")
def cmd_setup(force: Annotated[bool, typer.Option("--force", help="Replace existing metta command")] = False):
    """Create symlink to make metta command globally available."""
    setup_path(force=force)


@app.command(name="check")
def cmd_check():
    status = _check_existing_metta()
    if status == "ours":
        success("metta command is properly installed")
    elif status == "other":
        warning("metta command is installed from a different checkout")
        info("Run 'metta symlink-setup setup --force' to reinstall from this checkout")
    else:
        warning("metta command is not installed")
        info("Run 'metta symlink-setup setup' to install")


def main():
    app()


if __name__ == "__main__":
    main()
