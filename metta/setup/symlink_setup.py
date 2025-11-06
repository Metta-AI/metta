#!/usr/bin/env -S uv run
import os
import pathlib
import typing

import rich.console
import typer

import metta.common.util.fs
import metta.setup.utils

console = rich.console.Console()
app = typer.Typer(
    help="Symlink setup for global metta command",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

repo_root = metta.common.util.fs.get_repo_root()
wrapper_script = repo_root / "metta" / "setup" / "installer" / "bin" / "metta"
local_bin = pathlib.Path.home() / ".local" / "bin"
target_symlink = local_bin / "metta"


def _local_bin_is_in_path() -> bool:
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    return str(local_bin) in path_dirs


def _check_existing_metta() -> typing.Optional[str]:
    if not target_symlink.exists():
        return None

    if target_symlink.is_symlink():
        target = target_symlink.resolve()
        resolved_target = wrapper_script.resolve()
        if target == resolved_target:
            return "ours"
        elif target.read_bytes() == resolved_target.read_bytes():
            return "other-but-same-content"

    return "other"


def setup_path(force: bool = False, quiet: bool = False) -> None:
    wrapper_script.chmod(0o755)

    try:
        local_bin.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        metta.setup.utils.error(f"""
        Failed to create {local_bin}: {e}
        Please create this directory manually and re-run the installer.
        """)
        return

    existing = _check_existing_metta()

    if existing in ("ours", "other-but-same-content"):
        if not quiet:
            metta.setup.utils.success("metta is already symlinked correctly.")
        return
    elif existing == "other":
        if force:
            metta.setup.utils.info(f"Replacing existing metta command at {target_symlink}")
            try:
                target_symlink.unlink()
            except Exception as e:
                metta.setup.utils.error(f"Failed to remove existing file: {e}")
                return
        else:
            metta.setup.utils.debug(f"""
            A 'metta' command already exists at {target_symlink}
            Run with --force if you want to replace it.
            """)
            return

    try:
        target_symlink.symlink_to(wrapper_script)
        metta.setup.utils.success(f"Created symlink: {target_symlink} → {wrapper_script}")
    except Exception as e:
        metta.setup.utils.error(f"""
        Failed to create symlink: {e}
        You can create it manually with:
          ln -s {wrapper_script} {target_symlink}
        """)
        return

    if _local_bin_is_in_path():
        metta.setup.utils.success("metta command is now available")
    else:
        metta.setup.utils.warning(f"{local_bin} is not in your PATH")
        metta.setup.utils.info("""
        To use metta globally, add this to your shell profile:
          export PATH="$HOME/.local/bin:$PATH"

        For example:
          echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc
          source ~/.bashrc
        """)


@app.command(name="setup")
def cmd_setup(
    force: typing.Annotated[
        bool, typer.Option("--force", help="Create or replace symlink to make metta command globally available.")
    ] = False,
    quiet: typing.Annotated[bool, typer.Option("--quiet", help="Do not print success messages.")] = False,
):
    setup_path(force=force, quiet=quiet)


@app.command(name="check")
def cmd_check():
    status = _check_existing_metta()
    if status in ("ours", "other-but-same-content"):
        metta.setup.utils.success("metta command is properly installed")
    elif status == "other":
        metta.setup.utils.warning("""
        metta command is installed from a different checkout
        Run 'metta symlink-setup setup --force' to reinstall from this checkout
        """)
    else:
        metta.setup.utils.warning("""
        metta command is not installed
        Run 'metta symlink-setup setup' to install
        """)


def main():
    app()


if __name__ == "__main__":
    main()
