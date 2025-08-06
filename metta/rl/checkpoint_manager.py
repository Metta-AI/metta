"""Checkpoint management for Metta training."""

import logging
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from metta.agent.metta_agent import DistributedMettaAgent, make_policy
from metta.common.util.fs import wait_for_file
from metta.common.util.heartbeat import record_heartbeat
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.policy_management import cleanup_old_policies
from metta.rl.trainer_checkpoint import TrainerCheckpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing for both trainer state and policies."""

    def __init__(
        self,
        checkpoint_dir: str,
        policy_store: Any,
        trainer_cfg: Any,
        device: torch.device,
        is_master: bool,
        rank: int,
        run_name: str,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            policy_store: PolicyStore instance for saving/loading policies
            trainer_cfg: Trainer configuration
            device: Training device
            is_master: Whether this is the master process
            rank: Process rank for distributed training
            run_name: Name of the current run
        """
        self.checkpoint_dir = checkpoint_dir
        self.policy_store = policy_store
        self.trainer_cfg = trainer_cfg
        self.device = device
        self.is_master = is_master
        self.rank = rank
        self.run_name = run_name

        # Ensure checkpoint directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(
        self,
        run_dir: str,
        metta_grid_env: Any,
        cfg: Any,
    ) -> Tuple[TrainerCheckpoint | None, Any, int, int]:
        """Load checkpoint and policy if they exist, or create new ones.

        Args:
            run_dir: Directory containing checkpoints
            metta_grid_env: MettaGridEnv instance for policy creation
            cfg: Full config for policy creation

        Returns:
            Tuple of (checkpoint, policy_record, agent_step, epoch)
        """
        # Try to load trainer checkpoint
        checkpoint = TrainerCheckpoint.load(run_dir)
        agent_step = 0
        epoch = 0

        if checkpoint:
            agent_step = checkpoint.agent_step
            epoch = checkpoint.epoch
            logger.info(f"Restored from checkpoint at {agent_step} steps")

        # Try to load policy from checkpoint
        if checkpoint and checkpoint.policy_path:
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            policy_record = self.policy_store.policy_record(checkpoint.policy_path)
            self._restore_feature_mapping(policy_record)
            return checkpoint, policy_record, agent_step, epoch

        # Try to load initial policy from config
        if self.trainer_cfg.initial_policy and self.trainer_cfg.initial_policy.uri:
            logger.info(f"Loading initial policy URI: {self.trainer_cfg.initial_policy.uri}")
            policy_record = self.policy_store.policy_record(self.trainer_cfg.initial_policy.uri)
            self._restore_feature_mapping(policy_record)
            return checkpoint, policy_record, agent_step, epoch

        # Check for existing policy at default path
        default_path = os.path.join(self.checkpoint_dir, self.policy_store.make_model_name(0))
        if os.path.exists(default_path):
            logger.info(f"Loading policy from default path: {default_path}")
            policy_record = self.policy_store.policy_record(default_path)
            self._restore_feature_mapping(policy_record)
            return checkpoint, policy_record, agent_step, epoch

        # Create new policy with distributed coordination
        if torch.distributed.is_initialized() and not self.is_master:
            # Non-master waits for master to create
            logger.info(f"Rank {self.rank}: Waiting for master to create policy at {default_path}")
            # NOTE: Barrier removed - synchronization handled at call site
            if not wait_for_file(default_path, timeout=300):
                raise RuntimeError(f"Rank {self.rank}: Timeout waiting for policy at {default_path}")

            policy_record = self.policy_store.policy_record(default_path)
            self._restore_feature_mapping(policy_record)
            return checkpoint, policy_record, agent_step, epoch
        else:
            # Master creates new policy
            name = self.policy_store.make_model_name(0)
            pr = self.policy_store.create_empty_policy_record(name)
            pr.policy = make_policy(metta_grid_env, cfg)
            saved_pr = self.policy_store.save(pr)
            logger.info(f"Created and saved new policy to {saved_pr.uri}")
            # NOTE: Barrier removed - synchronization handled at call site
            return checkpoint, saved_pr, agent_step, epoch

    def save_checkpoint(
        self,
        agent_step: int,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        policy_path: str,
        timer: Any,
        run_dir: str,
        kickstarter: Any | None = None,
        force: bool = False,
    ) -> bool:
        """Save trainer checkpoint if needed.

        Args:
            agent_step: Current agent step
            epoch: Current epoch
            optimizer: Optimizer to save state from
            policy_path: Path to the saved policy
            timer: Stopwatch timer instance
            run_dir: Directory to save checkpoint in
            kickstarter: Optional kickstarter object for teacher_uri
            force: Force save even if interval not reached

        Returns:
            True if checkpoint was saved, False otherwise
        """
        # Combined check: save only if (forced OR at interval) AND is master
        checkpoint_interval = self.trainer_cfg.checkpoint.checkpoint_interval
        should_save = force or (checkpoint_interval and epoch % checkpoint_interval == 0)
        if not should_save:
            return False

        # This method should only be called by master
        if not self.is_master:
            logger.warning(f"save_checkpoint called on non-master rank {self.rank}")
            return False

        logger.info(f"Saving checkpoint at epoch {epoch}")

        # Record heartbeat to prevent timeout during long save operations
        record_heartbeat()

        # Build extra args if kickstarter is provided
        extra_args = {}
        if kickstarter and hasattr(kickstarter, "enabled") and kickstarter.enabled:
            if hasattr(kickstarter, "teacher_uri") and kickstarter.teacher_uri is not None:
                extra_args["teacher_pr_uri"] = kickstarter.teacher_uri

        # Create checkpoint
        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            policy_path=policy_path,
            stopwatch_state=timer.get_state()
            if hasattr(timer, "get_state")
            else timer.save_state()
            if hasattr(timer, "save_state")
            else None,
            extra_args=extra_args,
        )

        # Save checkpoint
        checkpoint.save(run_dir)

        # Cleanup old policies
        cleanup_old_policies(self.checkpoint_dir)

        return True

    def save_policy(
        self,
        policy: Any,
        epoch: int,
        agent_step: int,
        evals: EvalRewardSummary,
        timer: Any,
        initial_policy_record: Any | None,
        force: bool = False,
    ) -> Any | None:
        """Save policy with metadata if needed.

        Args:
            policy: Policy to save
            epoch: Current epoch
            agent_step: Current agent step
            evals: Evaluation scores
            timer: Stopwatch timer
            initial_policy_record: Initial policy record for metadata
            force: Force save even if interval not reached

        Returns:
            Saved policy record or None
        """
        # Check if we should save based on interval (all ranks must agree)
        checkpoint_interval = self.trainer_cfg.checkpoint.checkpoint_interval
        if not force and checkpoint_interval and epoch % checkpoint_interval != 0:
            return None

        # Now all ranks that should save are here
        # This method should only be called by master
        if not self.is_master:
            logger.warning(f"save_policy called on non-master rank {self.rank}")
            return None

        logger.info(f"Saving policy at epoch {epoch}")

        # Record heartbeat to prevent timeout during long save operations
        record_heartbeat()

        # Extract the actual policy module from distributed wrapper if needed
        policy_to_save = policy
        if isinstance(policy, DistributedMettaAgent):
            policy_to_save = policy.module

        # Build metadata
        name = self.policy_store.make_model_name(epoch)

        # Extract average reward and scores from evals
        avg_reward = 0.0
        score = 0.0
        evals_dict = {}

        if evals:
            if hasattr(evals, "avg_category_score"):
                # EvalRewardSummary object
                avg_reward = getattr(evals, "avg_category_score", 0.0) or 0.0
                category_scores = list(evals.category_scores.values())
                score = float(np.mean(category_scores)) if category_scores else 0.0
                evals_dict = {
                    "category_scores": evals.category_scores,
                    "simulation_scores": {
                        f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()
                    },
                    "avg_category_score": evals.avg_category_score,
                    "avg_simulation_score": evals.avg_simulation_score,
                }
            else:
                # Dict format
                score_keys = [k for k in evals.keys() if k.endswith("/score")]
                if score_keys:
                    avg_reward = sum(evals[k] for k in score_keys) / len(score_keys)
                score = avg_reward
                evals_dict = evals

        metadata = {
            "epoch": epoch,
            "agent_step": agent_step,
            "total_time": timer.get_elapsed(),
            "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
            "run": self.run_name,
            "initial_pr": initial_policy_record.uri if initial_policy_record else None,
            "generation": initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
            "evals": evals_dict,
            "avg_reward": avg_reward,
            "score": score,  # Aggregated score for sweep evaluation
        }

        # Save original feature mapping
        if hasattr(policy_to_save, "get_original_feature_mapping"):
            original_feature_mapping = policy_to_save.get_original_feature_mapping()
            if original_feature_mapping is not None:
                metadata["original_feature_mapping"] = original_feature_mapping
                logger.info(
                    f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata"
                )

        # Create and save policy record
        policy_record = self.policy_store.create_empty_policy_record(name)
        policy_record.metadata = metadata
        policy_record.policy = policy_to_save

        saved_policy_record = self.policy_store.save(policy_record)
        logger.info(f"Successfully saved policy at epoch {epoch}")

        # Clean up old policies periodically
        if epoch % 10 == 0:
            cleanup_old_policies(self.checkpoint_dir)

        # NOTE: Barrier removed - synchronization handled at call site
        return saved_policy_record

    def should_checkpoint(self, epoch: int, force: bool = False) -> bool:
        """Check if we should checkpoint at this epoch.

        All ranks must return the same value to maintain synchronization.

        Args:
            epoch: Current epoch
            force: Force checkpoint regardless of interval

        Returns:
            True if we should checkpoint
        """
        if force:
            return True

        checkpoint_interval = self.trainer_cfg.checkpoint.checkpoint_interval
        return checkpoint_interval and epoch % checkpoint_interval == 0

    def _restore_feature_mapping(self, policy_record: Any) -> None:
        """Restore original feature mapping from policy metadata if available."""
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from policy metadata")
