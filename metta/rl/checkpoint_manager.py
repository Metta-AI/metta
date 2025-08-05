"""Checkpoint management for Metta training."""

import logging
from pathlib import Path
from typing import Any

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.collections import remove_none_values
from metta.common.util.heartbeat import record_heartbeat
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.kickstarter import Kickstarter
from metta.rl.policy_management import cleanup_old_policies
from metta.rl.puffer_policy import PytorchAgent
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import CheckpointConfig
from metta.rl.utils import should_run

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing for both trainer state and policies."""

    def __init__(
        self,
        policy_store: PolicyStore,
        checkpoint_config: CheckpointConfig,
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
        self.policy_store = policy_store
        self.checkpoint_cfg = checkpoint_config
        self.device = device
        self.is_master = is_master
        self.rank = rank
        self.run_name = run_name

        # Ensure checkpoint directory exists
        Path(self.checkpoint_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        agent_step: int,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        policy_path: str,
        timer: Stopwatch,
        run_dir: str,
        kickstarter: Kickstarter | None = None,
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
        should_save = should_run(epoch, self.checkpoint_cfg.checkpoint_interval, self.is_master, force)
        if not should_save:
            return False

        logger.info(f"Saving checkpoint at epoch {epoch}")

        # Record heartbeat to prevent timeout during long save operations
        record_heartbeat()

        # Create checkpoint
        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            policy_path=policy_path,
            stopwatch_state=timer.save_state(),
            extra_args=remove_none_values({"teacher_pr_uri": kickstarter and kickstarter.teacher_uri}),
        )

        # Save checkpoint
        checkpoint.save(run_dir)

        # Cleanup old policies
        cleanup_old_policies(self.checkpoint_cfg.checkpoint_dir)

        return True

    def save_policy(
        self,
        policy: PolicyAgent,
        epoch: int,
        agent_step: int,
        evals: EvalRewardSummary,
        timer: Stopwatch,
        initial_policy_record: PolicyRecord | None,
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
        # Allow non-master ranks through; they are handled below
        if not should_run(
            epoch, self.checkpoint_cfg.checkpoint_interval, is_master=True, force=force, non_master_ok=True
        ):
            return None

        # Now all ranks that should save are here
        # Only master saves policies, but all ranks must participate in barrier
        if not self.is_master:
            # Non-master ranks need to participate in the barrier below
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return None

        logger.info(f"Saving policy at epoch {epoch}")

        # Record heartbeat to prevent timeout during long save operations
        record_heartbeat()

        # Extract the actual policy module from distributed wrapper if needed
        policy_to_save: MettaAgent | PytorchAgent = (
            policy.module if isinstance(policy, DistributedMettaAgent) else policy
        )

        # Build metadata
        name = self.policy_store.make_model_name(epoch)

        # Extract average reward and scores from evals
        evals_dict = {
            "category_scores": evals.category_scores.copy(),
            "simulation_scores": {f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()},
            "avg_category_score": evals.avg_category_score,
            "avg_simulation_score": evals.avg_simulation_score,
        }

        # TODO: reformat this; there is redundancy
        metadata = {
            "epoch": epoch,
            "agent_step": agent_step,
            "total_time": timer.get_elapsed(),
            "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
            "run": self.run_name,
            "initial_pr": initial_policy_record.uri if initial_policy_record else None,
            "generation": initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
            "evals": evals_dict,
            "avg_reward": evals.avg_category_score,
            "score": evals.avg_simulation_score,  # Aggregated score for sweep evaluation
        }

        # Save original feature mapping
        if isinstance(policy_to_save, MettaAgent):
            original_feature_mapping = policy_to_save.get_original_feature_mapping()
            if original_feature_mapping is not None:
                metadata["original_feature_mapping"] = original_feature_mapping
                logger.info(
                    f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata"
                )

            ##########################################################below
            original_action_config = policy_to_save.get_original_action_config()
            if original_action_config is not None:
                metadata["original_action_config"] = original_action_config
                logger.info(
                    f"Saving original_action_config with {len(original_action_config['names'])} actions to metadata"
                )
            ##########################################################

        # Create and save policy record
        policy_record = self.policy_store.create_empty_policy_record(
            name=name, checkpoint_dir=self.checkpoint_cfg.checkpoint_dir
        )
        policy_record.metadata = metadata
        policy_record.policy = policy_to_save

        saved_policy_record = self.policy_store.save(policy_record)
        logger.info(f"Successfully saved policy at epoch {epoch}")

        # Clean up old policies periodically
        if should_run(epoch, 10, self.is_master):
            cleanup_old_policies(self.checkpoint_cfg.checkpoint_dir)

        # Synchronize all ranks to ensure the policy is fully saved before continuing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return saved_policy_record
