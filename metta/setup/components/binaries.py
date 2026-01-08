import os
from pathlib import Path
from typing import Literal, Optional

from metta.common.util.fs import get_repo_root
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import debug, error, info, success, warning

checked_in_bin = get_repo_root() / "metta" / "setup" / "installer" / "bin"
local_bin = Path.home() / ".local" / "bin"


@register_module
class BinarySymlinksSetup(SetupModule):
    always_required = True

    @property
    def name(self) -> str:
        return "binary-symlinks"

    @property
    def description(self) -> str:
        return "Binaries for metta and cogames CLIs"

    def check_installed(self) -> bool:
        needs_install = False
        for target_symlink, wrapper_script in desired_symlinks:
            status = _check_existing(target_symlink, wrapper_script)
            name = target_symlink.name
            if status in ("ours", "other-but-same-content"):
                pass
            elif status == "other":
                info(f"""
              {name} command is installed from a different checkout
              Run 'metta install binary-symlinks --force' to reinstall from this checkout
              """)
            else:
                debug(f"""{name} command is not installed""")
                needs_install = True
        return not needs_install

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        for target_symlink, wrapper_script in desired_symlinks:
            setup_path(force=force, quiet=non_interactive, target_symlink=target_symlink, wrapper_script=wrapper_script)


def _check_existing(
    target_symlink: Path, wrapper_script: Path
) -> Optional[Literal["ours", "other-but-same-content", "other"]]:
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


def setup_path(force: bool, quiet: bool, target_symlink: Path, wrapper_script: Path) -> None:
    wrapper_script.chmod(0o755)

    try:
        local_bin.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error(f"""
        Failed to create {local_bin}: {e}
        Please create this directory manually and re-run the installer.
        """)
        return

    existing = _check_existing(target_symlink, wrapper_script)

    name = target_symlink.name

    if existing in ("ours", "other-but-same-content"):
        if not quiet:
            success(f"{name} symlinked correctly.")
        return
    elif existing == "other":
        if force:
            info(f"Replacing existing {target_symlink.name} command at {target_symlink}")
            try:
                target_symlink.unlink()
            except Exception as e:
                error(f"Failed to remove existing file: {e}")
                return
        else:
            debug(f"""
            A '{target_symlink.name}' command already exists at {target_symlink}
            Run with --force if you want to replace it.
            """)
            return

    try:
        target_symlink.symlink_to(wrapper_script)
        success(f"Created symlink: {target_symlink} → {wrapper_script}")
    except Exception as e:
        error(f"""
        Failed to create symlink: {e}
        You can create it manually with:
          ln -s {wrapper_script} {target_symlink}
        """)
        return

    if str(local_bin) in os.environ.get("PATH", "").split(os.pathsep):
        success(f"{name} command is now available")
    else:
        warning(f"{local_bin} is not in your PATH")
        info(f"""
        To use {name} globally, add this to your shell profile:
          export PATH="$HOME/.local/bin:$PATH"

        For example:
          echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc
          source ~/.bashrc
        """)


desired_symlinks = [
    (local_bin / "metta", checked_in_bin / "metta"),
    (local_bin / "cogames", checked_in_bin / "cogames"),
]
