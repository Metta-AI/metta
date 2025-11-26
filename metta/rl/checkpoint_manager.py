import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.tools.utils.auto_config import auto_policy_storage_decision
from mettagrid.policy.mpt_artifact import save_mpt
from mettagrid.util.url_schemes import checkpoint_filename
from mettagrid.util.url_schemes import get_latest_checkpoint as _get_latest_checkpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages run directories and trainer state checkpointing."""

    def __init__(self, run: str, system_cfg: SystemConfig):
        if not run or not run.strip():
            raise ValueError("Run name cannot be empty")
        if any(char in run for char in [" ", "/", "*", "\\", ":", "<", ">", "|", "?", '"']):
            raise ValueError(f"Run name contains invalid characters: {run}")
        if "__" in run:
            raise ValueError(f"Run name cannot contain '__': {run}")

        self.run_name = run
        self.run_dir = system_cfg.data_dir / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"

        os.makedirs(system_cfg.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._remote_prefix: str | None = None
        if not system_cfg.local_only:
            self._setup_remote_prefix()

    def _setup_remote_prefix(self) -> None:
        storage_decision = auto_policy_storage_decision(self.run_name)
        if storage_decision.remote_prefix:
            self._remote_prefix = storage_decision.remote_prefix
            if storage_decision.reason == "env_override":
                logger.info("Using POLICY_REMOTE_PREFIX: %s", storage_decision.remote_prefix)
            else:
                logger.info("Policies will sync to %s", storage_decision.remote_prefix)
        elif storage_decision.reason == "not_connected":
            logger.info("AWS SSO not detected; policies will remain local.")
        elif storage_decision.reason == "aws_not_enabled":
            logger.info("AWS disabled; policies will remain local.")
        elif storage_decision.reason == "no_base_prefix":
            logger.info("Remote prefix unset; policies will remain local.")

    @property
    def remote_checkpoints_enabled(self) -> bool:
        return self._remote_prefix is not None

    def get_latest_checkpoint(self) -> str | None:
        local_max = _get_latest_checkpoint(f"file://{self.checkpoint_dir}")
        remote_max = _get_latest_checkpoint(self._remote_prefix) if self._remote_prefix else None
        candidates = [c for c in [local_max, remote_max] if c]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["epoch"])["uri"]

    def save_policy_checkpoint(self, state_dict: dict, architecture, epoch: int) -> str:
        filename = checkpoint_filename(self.run_name, epoch)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        local_uri = save_mpt(self.checkpoint_dir / filename, architecture=architecture, state_dict=state_dict)

        if self._remote_prefix:
            remote_uri = save_mpt(f"{self._remote_prefix}/{filename}", architecture=architecture, state_dict=state_dict)
            logger.debug("Policy checkpoint saved remotely to %s", remote_uri)
            return remote_uri

        logger.debug("Policy checkpoint saved locally to %s", local_uri)
        return local_uri

    def load_trainer_state(self) -> Optional[Dict[str, Any]]:
        trainer_file = self.checkpoint_dir / "trainer_state.pt"
        if not trainer_file.exists():
            return None
        state = torch.load(trainer_file, map_location="cpu", weights_only=False)
        result = {
            "optimizer_state": state.get("optimizer", state.get("optimizer_state")),
            "epoch": state.get("epoch", 0),
            "agent_step": state.get("agent_step", 0),
        }
        if "stopwatch_state" in state:
            result["stopwatch_state"] = state["stopwatch_state"]
        if "curriculum_state" in state:
            result["curriculum_state"] = state["curriculum_state"]
        if "loss_states" in state:
            result["loss_states"] = state["loss_states"]
        return result

    def save_trainer_state(
        self,
        optimizer,
        epoch: int,
        agent_step: int,
        stopwatch_state: Optional[Dict[str, Any]] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        loss_states: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_file = self.checkpoint_dir / "trainer_state.pt"

        is_schedulefree = is_schedulefree_optimizer(optimizer)
        if is_schedulefree:
            optimizer.eval()

        state: dict[str, Any] = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        if curriculum_state:
            state["curriculum_state"] = curriculum_state
        if loss_states is not None:
            state["loss_states"] = loss_states

        with tempfile.NamedTemporaryFile(
            dir=self.checkpoint_dir,
            prefix=".trainer_state.pt.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

            try:
                torch.save(state, tmp_path)
                tmp_path.replace(trainer_file)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

        if is_schedulefree:
            optimizer.train()
