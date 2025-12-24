"""Policy checkpoint management component."""

import logging
from typing import Optional

import torch
from pydantic import Field

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_bundle import load_checkpoint_state
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, TrainerComponent
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri, resolve_uri

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
        policy_architecture: PolicyArchitecture,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_architecture: PolicyArchitecture = policy_architecture
        self._latest_policy_uri: Optional[str] = None

    def register(self, context) -> None:
        super().register(context)
        context.latest_policy_uri_fn = self.get_latest_policy_uri
        context.latest_policy_uri_value = self.get_latest_policy_uri()

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
                payload: dict[str, object] | None = None
                if self._distributed.is_master():
                    policy_spec = policy_spec_from_uri(normalized_uri)
                    architecture_spec, state_dict = load_checkpoint_state(policy_spec)
                    payload = {
                        "architecture_spec": architecture_spec,
                        "state_dict": {k: v.cpu() for k, v in state_dict.items()},
                        "action_count": len(policy_env_info.actions.actions()),
                    }
                payload = self._distributed.broadcast_from_master(payload)
                architecture_spec = payload["architecture_spec"]
                state_dict = payload["state_dict"]
                local_action_count = len(policy_env_info.actions.actions())
                action_count = payload["action_count"]
                if local_action_count != action_count:
                    raise ValueError(f"Action space mismatch: master={action_count}, rank={local_action_count}")

                policy = (
                    load_symbol(architecture_spec.split("(", 1)[0].strip())
                    .from_spec(architecture_spec)
                    .make_policy(policy_env_info)
                    .to(load_device)
                )
                missing, unexpected = policy.load_state_dict(state_dict, strict=True)
                if missing or unexpected:
                    raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")
                policy.initialize_to_environment(policy_env_info, load_device)

                if self._distributed.is_master():
                    self._latest_policy_uri = normalized_uri
                    logger.info("Loaded policy from %s", normalized_uri)
                return policy

        if candidate_uri:
            policy = initialize_or_load_policy(
                policy_env_info,
                policy_spec_from_uri(candidate_uri),
                device_override=str(load_device),
            )
            self._latest_policy_uri = resolve_uri(candidate_uri).canonical
            logger.info("Loaded policy from %s", candidate_uri)
            return policy

        logger.info("Creating new policy for training run")
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
        policy: Policy = self.context.policy
        if hasattr(policy, "module"):
            return policy.module
        return policy

    def _save_policy(self, epoch: int) -> None:
        policy = self._policy_to_save()
        uri = self._checkpoint_manager.save_policy_checkpoint(
            state_dict=policy.state_dict(),
            architecture=self._policy_architecture,
            epoch=epoch,
        )

        self._latest_policy_uri = uri
        self.context.latest_policy_uri_value = uri
        try:
            self.context.latest_saved_policy_epoch = epoch
        except AttributeError:
            logger.debug("Component context missing latest_saved_policy_epoch attribute")

        # Log latest checkpoint URI to wandb if available
        stats_reporter = getattr(self.context, "stats_reporter", None)
        wandb_run = getattr(stats_reporter, "wandb_run", None) if stats_reporter is not None else None
        if wandb_run is not None:
            wandb_run.log(
                {
                    "checkpoint/latest_uri": uri,
                    "checkpoint/latest_epoch": float(epoch),
                },
                step=self.context.agent_step,
            )
            logger.info(f"Logged checkpoint URI to wandb: {uri}")
