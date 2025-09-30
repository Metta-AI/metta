"""Policy checkpoint management component."""

import logging
from typing import Optional

from pydantic import Field

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, EnvironmentMetaData, TrainerComponent
from mettagrid.config import Config

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
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
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
        env_metadata: EnvironmentMetaData,
        policy_architecture: PolicyArchitecture,
        *,
        policy_uri: Optional[str] = None,
    ) -> Policy:
        """Load the latest policy checkpoint or create a new policy."""

        policy: Optional[Policy] = None
        candidate_uri: Optional[str] = policy_uri

        if candidate_uri is None:
            existing = self._checkpoint_manager.select_checkpoints("latest", count=1)
            candidate_uri = existing[0] if existing else None

        if self._distributed.is_master() and candidate_uri:
            normalized_uri = CheckpointManager.normalize_uri(candidate_uri)
            try:
                policy = self._checkpoint_manager.load_from_uri(normalized_uri)
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
        return policy_architecture.make_policy(env_metadata)

    def get_latest_policy_uri(self) -> Optional[str]:
        """Return the most recent checkpoint URI tracked by this component."""
        if self._latest_policy_uri:
            return self._latest_policy_uri
        latest = self._checkpoint_manager.select_checkpoints("latest", count=1)
        return latest[0] if latest else None

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

        self._save_policy(self.context.epoch, force=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _policy_to_save(self) -> Policy:
        policy: Policy = self.context.policy
        if hasattr(policy, "module"):
            return policy.module  # type: ignore[return-value]
        return policy

    def _collect_metadata(self, epoch: int, *, is_final: bool = False) -> dict:
        elapsed_breakdown = self.context.stopwatch.get_all_elapsed()
        metadata = {
            "epoch": epoch,
            "agent_step": self.context.agent_step,
            "total_time": self.context.stopwatch.get_elapsed(),
            "total_train_time": elapsed_breakdown.get("_rollout", 0) + elapsed_breakdown.get("_train", 0),
            "is_final": is_final,
        }
        eval_scores = self.context.latest_eval_scores
        if eval_scores and (eval_scores.category_scores or eval_scores.simulation_scores):
            metadata.update(
                {
                    "score": eval_scores.avg_simulation_score,
                    "avg_reward": eval_scores.avg_category_score,
                }
            )
        return metadata

    def _save_policy(self, epoch: int, *, force: bool = False) -> None:
        policy = self._policy_to_save()
        metadata = self._collect_metadata(epoch, is_final=force)

        uri = self._checkpoint_manager.save_agent(policy, epoch, metadata)
        self._latest_policy_uri = uri
        self.context.latest_policy_uri_value = uri
        try:
            self.context.latest_saved_policy_epoch = epoch
        except AttributeError:
            logger.debug("Component context missing latest_saved_policy_epoch attribute")
        logger.debug("Policy checkpoint saved to %s", uri)
