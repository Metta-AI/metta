"""Policy checkpoint management component."""

import logging
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field

from metta.agent.policy import Policy, PolicyArchitecture
from metta.common.util.file import write_file
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.policy_artifact import save_policy_artifact_safetensors
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

        # Broadcast just the URI; each rank loads locally to avoid pickling DDP-wrapped modules
        if self._distributed.is_master():
            normalized_uri = CheckpointManager.normalize_uri(candidate_uri) if candidate_uri else None
        else:
            normalized_uri = None
        normalized_uri = self._distributed.broadcast_from_master(normalized_uri)

        if normalized_uri:
            try:
                load_device = torch.device(self._distributed.config.device)
                spec = CheckpointManager.policy_spec_from_uri(normalized_uri, device=load_device)
                policy = initialize_or_load_policy(policy_env_info, spec)
                policy = self._ensure_save_capable(policy)
                # Guard against silently loading a policy with no trainable params (e.g., failed state_dict load)
                trainable = sum(p.numel() for p in policy.parameters() if getattr(p, "requires_grad", False))
                if trainable == 0:
                    raise ValueError("Loaded policy has zero trainable parameters")
                if self._distributed.is_master():
                    self._latest_policy_uri = normalized_uri
                    logger.info("Loaded policy from %s", normalized_uri)
                return policy
            except FileNotFoundError:
                if self._distributed.is_master():
                    logger.warning("Policy checkpoint %s not found; training will start fresh", normalized_uri)
            except Exception as exc:  # pragma: no cover - defensive guard
                if self._distributed.is_master():
                    logger.warning("Failed to load policy from %s: %s", normalized_uri, exc)

        logger.info("Creating new policy for training run")
        fresh_policy = self._policy_architecture.make_policy(policy_env_info)
        return self._ensure_save_capable(fresh_policy)

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

    def _ensure_save_capable(self, policy: Policy) -> Policy:
        """Attach a save_policy method if missing so saving is uniform."""
        if hasattr(policy, "save_policy"):
            return policy

        def save_policy(self, destination: str | Path, *, policy_architecture: PolicyArchitecture) -> str:
            path = Path(destination).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_policy_artifact_safetensors(
                path,
                policy_architecture=policy_architecture,
                state_dict=self.state_dict(),
            )
            return f"file://{path.resolve()}"

        policy.save_policy = save_policy.__get__(policy, policy.__class__)  # type: ignore[attr-defined]
        return policy

    def _save_policy(self, epoch: int) -> None:
        policy = self._ensure_save_capable(self._policy_to_save())

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
        try:
            self.context.latest_saved_policy_epoch = epoch
        except AttributeError:
            logger.debug("Component context missing latest_saved_policy_epoch attribute")
        logger.debug("Policy checkpoint saved to %s", uri)
