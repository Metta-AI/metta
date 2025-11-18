"""Utility helpers for CI/stable recipe jobs."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Callable


def copy_latest_checkpoint(src_dir: str, dest_dir: str) -> Path:
    """Copy the latest .mpt checkpoint from src_dir into dest_dir.

    If no local checkpoints are found, attempts to pull the latest via CheckpointManager
    (supports remote prefixes) based on the run and data_dir inferred from src_dir.

    Returns the destination path of the copied file.
    Raises FileNotFoundError if no checkpoints are found locally or remotely.
    """
    src = Path(src_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(src.glob("*.mpt"), key=lambda x: x.stat().st_mtime)

    if checkpoints:
        latest = checkpoints[-1]
        dest_path = dest / latest.name
        shutil.copy2(latest, dest_path)
        return dest_path

    # No local checkpoints: try remote via CheckpointManager
    try:
        from metta.common.util.file import local_copy
        from metta.rl.checkpoint_manager import CheckpointManager
        from metta.rl.system_config import SystemConfig

        run_dir = src.parent
        data_dir = run_dir.parent
        cm = CheckpointManager(run=run_dir.name, system_cfg=SystemConfig(data_dir=data_dir))
        latest_uri = cm.get_latest_checkpoint()
        if not latest_uri:
            raise FileNotFoundError
        with local_copy(latest_uri) as local_path:
            dest_path = dest / Path(local_path).name
            shutil.copy2(local_path, dest_path)
            return dest_path
    except Exception:
        raise FileNotFoundError(f"No checkpoints found in {src_dir} (local or remote)") from None


def copy_and_eval(src_dir: str, dest_dir: str, eval_module: str, eval_fn: str, eval_arg_name: str) -> None:
    """Copy latest checkpoint then invoke the given evaluate function.

    eval_module: dotted module path (e.g., "recipes.prod.arena_basic_easy_shaped")
    eval_fn: function name in that module (e.g., "evaluate_latest_in_dir")
    eval_arg_name: keyword argument name for the directory (e.g., "dir_path")
    """
    copy_latest_checkpoint(src_dir, dest_dir)
    mod = importlib.import_module(eval_module)
    fn: Callable = getattr(mod, eval_fn)
    fn(**{eval_arg_name: Path(dest_dir)})
