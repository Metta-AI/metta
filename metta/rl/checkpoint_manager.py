import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.tools.utils.auto_config import auto_policy_storage_decision
from mettagrid.policy.mpt_artifact import load_mpt, save_mpt
from mettagrid.policy.mpt_policy import MptPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename, parse_uri, resolve_uri

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages run directories and trainer state checkpointing."""

    def __init__(self, run: str, system_cfg: SystemConfig, require_remote_enabled: bool = False):
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
        if require_remote_enabled and self._remote_prefix is None:
            raise ValueError("Remote checkpoints are required but remote prefix is not set")

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

    def get_latest_checkpoint(self) -> str | None:
        def try_resolve(uri: str) -> tuple[str, int] | None:
            try:
                resolved = resolve_uri(uri)
                info = parse_uri(resolved).checkpoint_info
                if info:
                    return (resolved, info[1])
            except (ValueError, FileNotFoundError):
                pass
            return None

        local = try_resolve(f"file://{self.checkpoint_dir}")
        remote = try_resolve(self._remote_prefix) if self._remote_prefix else None
        candidates = [c for c in [local, remote] if c]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def save_policy_checkpoint(
        self,
        state_dict: dict,
        architecture,
        epoch: int,
        policy_env_info: PolicyEnvInterface | None = None,
    ) -> str:
        filename = checkpoint_filename(self.run_name, epoch)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        local_uri = save_mpt(self.checkpoint_dir / filename, architecture=architecture, state_dict=state_dict)
        if policy_env_info is not None:
            self._persist_env_metadata(policy_env_info)

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

    @staticmethod
    def load_from_uri(
        uri: str,
        policy_env_info: PolicyEnvInterface,
        device: torch.device | str = "cpu",
        *,
        strict: bool = True,
        validate_env: bool = True,
    ) -> MptPolicy:
        """Load a policy checkpoint and optionally validate env metadata."""

        resolved_uri = resolve_uri(uri)
        parsed = parse_uri(resolved_uri)
        local_path = Path(parsed.local_path) if parsed.local_path else None
        local_meta = CheckpointManager._load_env_metadata(local_path) if local_path else None
        if validate_env and local_meta:
            CheckpointManager._validate_env_metadata(local_meta, policy_env_info)

        artifact = load_mpt(resolved_uri)
        policy = artifact.instantiate(policy_env_info, device=device, strict=strict)
        return policy

    def _persist_env_metadata(self, policy_env_info: PolicyEnvInterface) -> None:
        """Persist minimal env metadata alongside checkpoints for resume/eval validation."""

        meta = {
            "num_agents": policy_env_info.num_agents,
            "obs_width": policy_env_info.obs_width,
            "obs_height": policy_env_info.obs_height,
            "actions": [a.name for a in policy_env_info.actions.actions()],
            "tags": policy_env_info.tags,
        }
        target = self.checkpoint_dir / "policy_env_info.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(target)

    @staticmethod
    def _load_env_metadata(resolved_uri: str | Path | None) -> dict[str, Any] | None:
        """Attempt to read env metadata colocated with the checkpoint file."""

        try:
            if resolved_uri is None:
                return None
            path = Path(resolved_uri)
            meta_path = path.parent / "policy_env_info.json"
            if meta_path.exists():
                return json.loads(meta_path.read_text())
        except Exception:
            logger.debug("Failed to read env metadata near %s", resolved_uri, exc_info=True)
        return None

    @staticmethod
    def _validate_env_metadata(meta: dict[str, Any], policy_env_info: PolicyEnvInterface) -> None:
        """Warn if saved env metadata differs from the current environment."""

        try:
            mismatches = []
            if meta.get("num_agents") != policy_env_info.num_agents:
                mismatches.append(f"num_agents saved={meta.get('num_agents')} current={policy_env_info.num_agents}")
            if meta.get("obs_width") != policy_env_info.obs_width or meta.get("obs_height") != policy_env_info.obs_height:
                mismatches.append(
                    f"obs_shape saved=({meta.get('obs_width')},{meta.get('obs_height')}) "
                    f"current=({policy_env_info.obs_width},{policy_env_info.obs_height})"
                )
            saved_actions = meta.get("actions", [])
            current_actions = [a.name for a in policy_env_info.actions.actions()]
            if saved_actions != current_actions:
                mismatches.append(f"actions saved={saved_actions} current={current_actions}")
            if mismatches:
                logger.warning("Environment metadata mismatch when loading policy: %s", "; ".join(mismatches))
        except Exception:
            logger.debug("Env metadata validation failed", exc_info=True)


# Here temporarily for backwards-compatibility but we will move it
CheckpointPolicy = MptPolicy
