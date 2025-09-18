"""Policy checkpoint management component."""

from __future__ import annotations

import logging
from typing import Optional

from metta.agent.policy_architecture import PolicyArchitecture
from metta.agent.policy_base import Policy
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import TrainerComponent
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.training_environment import EnvironmentMetaData

logger = logging.getLogger(__name__)


class PolicyCheckpointerConfig(Config):
    """Configuration for policy checkpointing."""

    epoch_interval: int = 100
    """How often to save policy checkpoints (in epochs)."""


class PolicyCheckpointer(TrainerComponent):
    """Manages policy checkpointing with distributed awareness and URI support."""

    def __init__(
        self,
        *,
        config: PolicyCheckpointerConfig,
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
    def register(self, trainer) -> None:  # type: ignore[override]
        super().register(trainer)
        trainer.policy_checkpointer = self

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
        trainer = self._trainer
        if trainer is None or not self._distributed.should_checkpoint():
            return

        if epoch % self._config.epoch_interval != 0:
            return

        self._save_policy(trainer, epoch, force=False)

    def on_training_complete(self) -> None:  # type: ignore[override]
        trainer = self._trainer
        if trainer is None or not self._distributed.should_checkpoint():
            return

        self._save_policy(trainer, trainer.epoch, force=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _policy_to_save(self, trainer) -> Policy:
        policy: Policy = trainer.policy
        if hasattr(policy, "module"):
            return policy.module  # type: ignore[return-value]
        return policy

    def _collect_metadata(self, trainer, epoch: int, *, is_final: bool = False) -> dict:
        elapsed_breakdown = trainer.stopwatch.get_all_elapsed()
        metadata = {
            "epoch": epoch,
            "agent_step": trainer.agent_step,
            "total_time": trainer.stopwatch.get_elapsed(),
            "total_train_time": elapsed_breakdown.get("_rollout", 0) + elapsed_breakdown.get("_train", 0),
            "is_final": is_final,
        }
        evaluator = getattr(trainer, "evaluator", None)
        if evaluator is not None:
            try:
                eval_scores = evaluator.get_latest_scores()
            except AttributeError:
                eval_scores = None
            if eval_scores and (eval_scores.category_scores or eval_scores.simulation_scores):
                metadata.update(
                    {
                        "score": eval_scores.avg_simulation_score,
                        "avg_reward": eval_scores.avg_category_score,
                    }
                )
        return metadata

    def _save_policy(self, trainer, epoch: int, *, force: bool) -> None:
        policy = self._policy_to_save(trainer)
        metadata = self._collect_metadata(trainer, epoch, is_final=force)

        if not force and epoch % self._config.epoch_interval != 0:
            return

        uri = self._checkpoint_manager.save_agent(policy, epoch, metadata)
        self._latest_policy_uri = uri
        logger.debug("Policy checkpoint saved to %s", uri)
