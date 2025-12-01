import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success


def cmd_clean(
    force: Annotated[bool, typer.Option("--force", help="Force clean")] = False,
):
    repo_root = get_repo_root()
    removed_dirs = []

    def _remove_matching_dirs(base: Path, patterns: list[str], *, include_globs: bool = False) -> None:
        for pattern in patterns:
            candidates = base.glob(pattern) if include_globs else (base / pattern,)
            for path in candidates:
                if not path.exists() or not path.is_dir():
                    continue
                info(f"  Removing {path.relative_to(repo_root)}...")
                removed_dirs.append(path)
                subprocess.run(["chmod", "-R", "u+w", str(path)], cwd=repo_root, check=False)
                subprocess.run(["rm", "-rf", str(path)], cwd=repo_root, check=False)

    def _remove_exact_dir(path: Path) -> None:
        if path.exists():
            info(f"  Removing {path.relative_to(repo_root)}...")
            removed_dirs.append(path)
            shutil.rmtree(path)

    _remove_exact_dir(repo_root / "build")

    mettagrid_dir = repo_root / "packages" / "mettagrid"
    _remove_exact_dir(mettagrid_dir / "build-debug")
    _remove_exact_dir(mettagrid_dir / "build-release")

    _remove_matching_dirs(repo_root, ["bazel-*"], include_globs=True)
    _remove_matching_dirs(repo_root, [".bazel_output"])
    if mettagrid_dir.exists():
        _remove_matching_dirs(mettagrid_dir, ["bazel-*"], include_globs=True)
        _remove_matching_dirs(mettagrid_dir, [".bazel_output"])

    if force:
        _remove_exact_dir(mettagrid_dir / "nim" / "mettascope" / "bindings" / "generated")

    # Walk through directory tree bottom-up
    for dirpath, _, _ in os.walk(repo_root, topdown=False):
        dir_path = Path(dirpath)

        # Skip .git directory and its subdirectories
        if ".git" in dir_path.parts:
            continue

        # Skip build cache directories
        if any(part in [".turbo", ".vite-temp", "node_modules"] for part in dir_path.parts):
            continue

        if is_dir_empty_or_pycache_only(dir_path):
            success(f"Removing: {dir_path.relative_to(repo_root)}")
            shutil.rmtree(dir_path)
            removed_dirs.append(dir_path)
    info(f"Removed {len(removed_dirs)} directories")


def is_dir_empty_or_pycache_only(dir_path: Path) -> bool:
    try:
        contents = list(os.listdir(dir_path))

        # Empty directory
        if not contents:
            return True

        # Only contains __pycache__
        if len(contents) == 1 and contents[0] == "__pycache__":
            return True

        return False
    except PermissionError:
        error(f"Permission denied: {dir_path}")
        return False
