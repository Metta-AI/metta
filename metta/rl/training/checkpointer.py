"""Policy checkpoint management component."""

import logging
from typing import Optional

import torch
from pydantic import Field

from metta.agent.policy import Policy, PolicyArchitecture
from metta.common.util.file import write_file
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, TrainerComponent
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

logger = logging.getLogger(__name__)


class CheckpointerConfig(Config):
    """Configuration for policy checkpointing."""

    epoch_interval: int = Field(default=30, ge=0)  # How often to save policy checkpoints (in epochs)


class Checkpointer(TrainerComponent):
    """Manages policy checkpointing with distributed awareness and URI support."""

    def __init__(
        self,
        *,
        config: CheckpointerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        policy_architecture: PolicyArchitecture,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_architecture: PolicyArchitecture = policy_architecture
        self._latest_policy_uri: Optional[str] = None

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        context.latest_policy_uri_fn = self.get_latest_policy_uri
        context.latest_policy_uri_value = self.get_latest_policy_uri()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def load_or_create_policy(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        policy_uri: Optional[str] = None,
    ) -> Policy:
        """Load the latest policy checkpoint or create a new policy."""

        candidate_uri: Optional[str] = policy_uri or self._checkpoint_manager.get_latest_checkpoint()
        load_device = torch.device(self._distributed.config.device)

        # Distributed: master loads once, broadcasts payload; workers rebuild locally.
        if self._distributed.is_distributed:
            normalized_uri = (
                CheckpointManager.normalize_uri(candidate_uri)
                if self._distributed.is_master() and candidate_uri
                else None
            )
            normalized_uri = self._distributed.broadcast_from_master(normalized_uri)

            if normalized_uri:
                spec = CheckpointManager.policy_spec_from_uri(
                    normalized_uri,
                    device=load_device,
                    policy_architecture=self._policy_architecture,
                )
                payload = None
                if self._distributed.is_master():
                    loaded_policy = initialize_or_load_policy(policy_env_info, spec)
                    policy_obj = self._require_policy_instance(loaded_policy)
                    state_dict = {k: v.cpu() for k, v in policy_obj.state_dict().items()}
                    arch = getattr(policy_obj, "_policy_architecture", self._policy_architecture)
                    action_count = len(policy_env_info.actions.actions())
                    payload = (state_dict, arch, action_count, normalized_uri)
                state_dict, arch, action_count, normalized_uri = self._distributed.broadcast_from_master(payload)

                local_action_count = len(policy_env_info.actions.actions())
                if local_action_count != action_count:
                    msg = f"Action space mismatch on resume: master={action_count}, rank={local_action_count}"
                    raise ValueError(msg)

                policy = arch.make_policy(policy_env_info).to(load_device)
                if hasattr(policy, "initialize_to_environment"):
                    policy.initialize_to_environment(policy_env_info, load_device)
                missing, unexpected = policy.load_state_dict(state_dict, strict=True)
                if missing or unexpected:
                    raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")
                if self._distributed.is_master():
                    self._latest_policy_uri = normalized_uri
                    logger.info("Loaded policy from %s", normalized_uri)
                return policy

        # Non-distributed or fallthrough: load locally (fail hard on errors)
        if candidate_uri:
            normalized_uri = CheckpointManager.normalize_uri(candidate_uri)
            spec = CheckpointManager.policy_spec_from_uri(
                normalized_uri,
                device=load_device,
                policy_architecture=self._policy_architecture,
            )
            policy = initialize_or_load_policy(policy_env_info, spec)
            policy = self._require_policy_instance(policy)
            self._latest_policy_uri = normalized_uri
            logger.info("Loaded policy from %s", normalized_uri)
            return policy

        logger.info("Creating new policy for training run")
        fresh_policy = self._policy_architecture.make_policy(policy_env_info)
        return self._require_policy_instance(fresh_policy)

    def get_latest_policy_uri(self) -> Optional[str]:
        """Return the most recent checkpoint URI."""
        return self._checkpoint_manager.get_latest_checkpoint() or self._latest_policy_uri

    # ------------------------------------------------------------------
    # Callback entry-points
    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self._distributed.should_checkpoint():
            return

        if epoch % self._config.epoch_interval != 0:
            return

        self._save_policy(epoch)

    def on_training_complete(self) -> None:  # type: ignore[override]
        if not self._distributed.should_checkpoint():
            return

        self._save_policy(self.context.epoch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _policy_to_save(self) -> Policy:
        policy: Policy = self.context.policy
        if hasattr(policy, "module"):
            return policy.module  # type: ignore[return-value]
        return policy

    def _require_policy_instance(self, candidate: object) -> Policy:
        if isinstance(candidate, Policy):
            return candidate

        # Unwrap CheckpointPolicy to get the actual Policy instance
        if hasattr(candidate, "_policy") and isinstance(candidate._policy, Policy):  # type: ignore[attr-defined]
            return candidate._policy  # type: ignore[attr-defined]

        msg = f"Checkpointer expected Policy, got {type(candidate).__name__}"
        raise TypeError(msg)

    def _save_policy(self, epoch: int) -> None:
        policy = self._policy_to_save()

        filename = f"{self._checkpoint_manager.run_name}:v{epoch}.mpt"
        checkpoint_dir = self._checkpoint_manager.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        local_path = checkpoint_dir / filename

        local_uri = policy.save_policy(local_path, policy_architecture=self._policy_architecture)

        uri = local_uri
        if getattr(self._checkpoint_manager, "_remote_prefix", None):
            remote_uri = f"{self._checkpoint_manager._remote_prefix}/{filename}"
            write_file(remote_uri, str(local_path))
            uri = remote_uri

        self._latest_policy_uri = uri
        self.context.latest_policy_uri_value = uri
        self.context.latest_saved_policy_epoch = epoch
        logger.debug("Policy checkpoint saved to %s", uri)
