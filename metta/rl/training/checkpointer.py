"""Policy checkpoint management component."""

import logging
import typing

import pydantic
import torch

import metta.agent.policy
import metta.rl.checkpoint_manager
import metta.rl.training
import mettagrid.base_config
import mettagrid.policy.policy_env_interface

logger = logging.getLogger(__name__)


class CheckpointerConfig(mettagrid.base_config.Config):
    """Configuration for policy checkpointing."""

    epoch_interval: int = pydantic.Field(default=30, ge=0)  # How often to save policy checkpoints (in epochs)


class Checkpointer(metta.rl.training.TrainerComponent):
    """Manages policy checkpointing with distributed awareness and URI support."""

    def __init__(
        self,
        *,
        config: CheckpointerConfig,
        checkpoint_manager: metta.rl.checkpoint_manager.CheckpointManager,
        distributed_helper: metta.rl.training.DistributedHelper,
        policy_architecture: metta.agent.policy.PolicyArchitecture,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_architecture: metta.agent.policy.PolicyArchitecture = policy_architecture
        self._latest_policy_uri: typing.Optional[str] = None

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
        game_rules: mettagrid.policy.policy_env_interface.PolicyEnvInterface,
        *,
        policy_uri: typing.Optional[str] = None,
    ) -> metta.agent.policy.Policy:
        """Load the latest policy checkpoint or create a new policy."""

        policy: typing.Optional[metta.agent.policy.Policy] = None
        candidate_uri: typing.Optional[str] = policy_uri

        if candidate_uri is None:
            candidate_uri = self._checkpoint_manager.get_latest_checkpoint()

        if self._distributed.is_master() and candidate_uri:
            normalized_uri = metta.rl.checkpoint_manager.CheckpointManager.normalize_uri(candidate_uri)
            try:
                load_device = torch.device(self._distributed.config.device)
                policy = self._checkpoint_manager.load_from_uri(normalized_uri, game_rules, load_device)
                self._latest_policy_uri = normalized_uri
                logger.info("Loaded policy from %s", normalized_uri)
            except FileNotFoundError:
                logger.warning("Policy checkpoint %s not found; training will start fresh", normalized_uri)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to load policy from %s: %s", normalized_uri, exc)

        policy = self._distributed.broadcast_from_master(policy)
        if policy is not None:
            return policy

        logger.info("Creating new policy for training run")
        return self._policy_architecture.make_policy(game_rules)

    def get_latest_policy_uri(self) -> typing.Optional[str]:
        """Return the most recent checkpoint URI tracked by this component."""
        if self._latest_policy_uri:
            return self._latest_policy_uri
        return self._checkpoint_manager.get_latest_checkpoint()

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
    def _policy_to_save(self) -> metta.agent.policy.Policy:
        policy: metta.agent.policy.Policy = self.context.policy
        if hasattr(policy, "module"):
            return policy.module  # type: ignore[return-value]
        return policy

    def _save_policy(self, epoch: int, *, force: bool = False) -> None:
        policy = self._policy_to_save()

        uri = self._checkpoint_manager.save_agent(
            policy,
            epoch,
            policy_architecture=self._policy_architecture,
        )
        self._latest_policy_uri = uri
        self.context.latest_policy_uri_value = uri
        try:
            self.context.latest_saved_policy_epoch = epoch
        except AttributeError:
            logger.debug("Component context missing latest_saved_policy_epoch attribute")
        logger.debug("Policy checkpoint saved to %s", uri)
