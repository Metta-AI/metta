from __future__ import annotations

import json
import logging
import os
import time
import weakref
from typing import Any, Dict

import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Use WeakKeyDictionary to associate state with each wandb.Run without mutating the object
_ABORT_STATE: weakref.WeakKeyDictionary[wandb.sdk.wandb_run.Run, Dict[str, Any]] = weakref.WeakKeyDictionary()


def abort_requested(wandb_run: wandb.sdk.wandb_run.Run | None, min_interval_sec: int = 60) -> bool:
    """Return True if the WandB run has an "abort" tag.

    The API call is throttled to *min_interval_sec* seconds.
    """
    if wandb_run is None:
        return False

    state = _ABORT_STATE.setdefault(wandb_run, {"last_check": 0.0, "cached_result": False})
    now = time.time()

    # Return cached result if within throttle interval
    if now - state["last_check"] < min_interval_sec:
        return state["cached_result"]

    # Time to check again
    state["last_check"] = now
    try:
        run_obj = wandb.Api().run(wandb_run.path)
        state["cached_result"] = "abort" in run_obj.tags
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        state["cached_result"] = False

    return state["cached_result"]


def upload_env_configs(curriculum: Any, wandb_run: wandb.sdk.wandb_run.Run | None) -> None:
    """Serialize resolved env configs for each bucket and upload to run files."""
    if wandb_run is None:
        return
    try:
        if not hasattr(curriculum, "get_env_cfg_by_bucket"):
            logger.debug("Curriculum does not implement get_env_cfg_by_bucket; skipping env config upload")
            return
        env_cfgs: Dict[str, Any] = curriculum.get_env_cfg_by_bucket()  # type: ignore[attr-defined]
        resolved = {k: OmegaConf.to_container(v, resolve=True) for k, v in env_cfgs.items()}
        payload = json.dumps(resolved, indent=2)
        file_path = os.path.join(wandb_run.dir, "env_configs.json")
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(payload)
        try:
            wandb_run.save(file_path, base_path=wandb_run.dir, policy="now")
        except Exception:
            pass  # offline mode
    except Exception as e:
        logger.warning(f"Failed to upload env configs: {e}")
