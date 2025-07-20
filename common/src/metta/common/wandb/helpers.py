from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Policy artifact upload
# -------------------------------------------------------------------------------------------------


def add_policy_artifact(wandb_run: wandb.sdk.wandb_run.Run | None, policy_store: Any, policy_record: Any) -> None:
    """Upload *policy_record* as an artifact to *wandb_run* using policy_store helper.

    Safe-no-op if wandb disabled or record is None.
    """
    if wandb_run is None or policy_record is None:
        return
    try:
        policy_store.add_to_wandb_run(wandb_run.id, policy_record)
    except Exception as e:
        logger.warning(f"Failed to upload policy to WandB: {e}")


# -------------------------------------------------------------------------------------------------
# Abort tag check (throttled)
# -------------------------------------------------------------------------------------------------

_LAST_ABORT_CHECK: float = 0.0
_ABORT_CACHE: bool = False


def abort_requested(wandb_run: wandb.sdk.wandb_run.Run | None, min_interval_sec: int = 60) -> bool:
    """Return True if the WandB run has an "abort" tag.

    The API call is throttled to *min_interval_sec* seconds.
    """
    global _LAST_ABORT_CHECK, _ABORT_CACHE

    if wandb_run is None:
        return False

    now = time.time()
    if now - _LAST_ABORT_CHECK < min_interval_sec:
        return _ABORT_CACHE

    _LAST_ABORT_CHECK = now
    try:
        run_obj = wandb.Api().run(wandb_run.path)
        _ABORT_CACHE = "abort" in run_obj.tags
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        _ABORT_CACHE = False
    return _ABORT_CACHE


# -------------------------------------------------------------------------------------------------
# Environment-config upload helper
# -------------------------------------------------------------------------------------------------


def upload_env_configs(curriculum: Any, wandb_run: wandb.sdk.wandb_run.Run | None) -> None:
    """Serialize resolved env configs for each bucket and upload to run files."""
    if wandb_run is None:
        return
    try:
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
