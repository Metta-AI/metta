"""Policy checkpoint management component."""

import logging
from collections.abc import Callable
from typing import Optional

import torch
from pydantic import Field

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, TrainerComponent
from mettagrid.base_config import Config
from mettagrid.policy.mpt_artifact import MptArtifact, load_mpt
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import resolve_uri

logger = logging.getLogger(__name__)


class CheckpointerConfig(Config):
    epoch_interval: int = Field(default=30, ge=0)


class Checkpointer(TrainerComponent):
    """Manages policy checkpointing with distributed awareness."""

    def __init__(
        self,
        *,
        config: CheckpointerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        policy_architecture: PolicyArchitecture | None,
        policy_getter: Callable[[], Policy] | None = None,
        policy_name: str | None = None,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_architecture: PolicyArchitecture | None = policy_architecture
        self._latest_policy_uri: Optional[str] = None
        self._policy_getter = policy_getter
        self._policy_name = policy_name

    def register(self, context) -> None:
        super().register(context)
        if not self._policy_name:
            raise ValueError("Checkpointer requires policy_name when using multi-policy assets")
        latest_uris = getattr(context, "latest_policy_uris", None)
        if latest_uris is None:
            latest_uris = {}
            context.latest_policy_uris = latest_uris
        latest_uris[self._policy_name] = self.get_latest_policy_uri()

    def load_or_create_policy(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        policy_uri: Optional[str] = None,
    ) -> Policy:
        """Load the latest policy checkpoint or create a new policy."""
        candidate_uri = policy_uri or self._checkpoint_manager.get_latest_checkpoint()
        load_device = torch.device(self._distributed.config.device)

        if self._distributed.is_distributed:
            normalized_uri = None
            if self._distributed.is_master() and candidate_uri:
                normalized_uri = resolve_uri(candidate_uri).canonical
            normalized_uri = self._distributed.broadcast_from_master(normalized_uri)

            if normalized_uri:
                artifact: MptArtifact | None = None
                if self._distributed.is_master():
                    artifact = load_mpt(normalized_uri)

                state_dict = self._distributed.broadcast_from_master(
                    {k: v.cpu() for k, v in artifact.state_dict.items()} if artifact else None
                )
                arch = self._distributed.broadcast_from_master(artifact.architecture if artifact else None)
                action_count = self._distributed.broadcast_from_master(
                    len(policy_env_info.actions.actions()) if self._distributed.is_master() else None
                )

                local_action_count = len(policy_env_info.actions.actions())
                if local_action_count != action_count:
                    raise ValueError(f"Action space mismatch: master={action_count}, rank={local_action_count}")

                if arch is None:
                    raise RuntimeError(f"Loaded artifact from {normalized_uri} is missing architecture")
                self._policy_architecture = arch
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

        if candidate_uri:
            artifact = load_mpt(candidate_uri)
            policy = artifact.instantiate(policy_env_info, self._distributed.config.device)
            self._policy_architecture = artifact.architecture
            self._latest_policy_uri = resolve_uri(candidate_uri).canonical
            logger.info("Loaded policy from %s", candidate_uri)
            return policy

        logger.info("Creating new policy for training run")
        if self._policy_architecture is None:
            raise ValueError("Cannot create a new policy without a policy_architecture")
        return self._policy_architecture.make_policy(policy_env_info)

    def get_latest_policy_uri(self) -> Optional[str]:
        return self._checkpoint_manager.get_latest_checkpoint() or self._latest_policy_uri

    def on_epoch_end(self, epoch: int) -> None:
        if not self._distributed.should_checkpoint():
            return
        if epoch % self._config.epoch_interval != 0:
            return
        self._save_policy(epoch)

    def on_training_complete(self) -> None:
        if not self._distributed.should_checkpoint():
            return
        self._save_policy(self.context.epoch)

    def _policy_to_save(self) -> Policy:
        policy: Policy = self._policy_getter() if self._policy_getter is not None else self.context.policy
        if hasattr(policy, "module"):
            return policy.module
        return policy

    def _save_policy(self, epoch: int) -> None:
        policy = self._policy_to_save()
        if self._policy_architecture is None:
            raise RuntimeError("Cannot checkpoint a policy without a known architecture")
        uri = self._checkpoint_manager.save_policy_checkpoint(
            state_dict=policy.state_dict(),
            architecture=self._policy_architecture,
            epoch=epoch,
        )

        self._latest_policy_uri = uri
        latest_uris = getattr(self.context, "latest_policy_uris", None)
        if latest_uris is None:
            latest_uris = {}
            self.context.latest_policy_uris = latest_uris
        if self._policy_name:
            latest_uris[self._policy_name] = uri
        try:
            # Track the most recent epoch at which *any* policy checkpoint was saved.
            self.context.latest_saved_policy_epoch = epoch
        except AttributeError:
            logger.debug("Component context missing latest_saved_policy_epoch attribute")

        # Log latest checkpoint URI to wandb if available
        stats_reporter = getattr(self.context, "stats_reporter", None)
        wandb_run = getattr(stats_reporter, "wandb_run", None) if stats_reporter is not None else None
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"checkpoint/{self._policy_name}/latest_uri": uri,
                    f"checkpoint/{self._policy_name}/latest_epoch": float(epoch),
                },
                step=self.context.agent_step,
            )
            logger.info(f"Logged checkpoint URI to wandb: {uri}")
