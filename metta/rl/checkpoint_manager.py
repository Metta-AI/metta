"""Checkpoint management for Metta training."""

import logging
from pathlib import Path

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.collections import remove_none_values
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.kickstarter import Kickstarter
from metta.rl.policy_initializer import PolicyInitializer
from metta.rl.policy_management import cleanup_old_policies
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import CheckpointConfig
from metta.rl.utils import should_run
from metta.rl.wandb import upload_policy_artifact

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
        initializer: PolicyInitializer,
    ):
        """Initialize checkpoint manager.

        Args:
            policy_store: PolicyStore instance for saving/loading policies
            checkpoint_config: Checkpoint configuration
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
        self.initializer = initializer

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
    ) -> bool:
        """Save trainer checkpoint if needed."""
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

        return True

    def model_suffix(self) -> str:
        return self.checkpoint_cfg.model_suffix()

    def save_policy(
        self,
        policy: PolicyAgent,
        epoch: int,
        agent_step: int,
        evals: EvalRewardSummary,
        timer: Stopwatch,
        initial_policy_uri: str | None,
    ) -> PolicyRecord:
        """Save policy with metadata if needed."""

        logger.info(f"Saving policy at epoch {epoch}")

        # Extract the actual policy module from distributed wrapper if needed
        policy_to_save: MettaAgent = policy.module if isinstance(policy, DistributedMettaAgent) else policy

        # Build metadata
        name = PolicyStore.make_model_name(epoch, self.model_suffix())

        # Extract average reward and scores from evals
        evals_dict = {
            "category_scores": evals.category_scores.copy(),
            "simulation_scores": {f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()},
            "avg_category_score": evals.avg_category_score,
            "avg_simulation_score": evals.avg_simulation_score,
        }

        metadata = {
            "epoch": epoch,
            "agent_step": agent_step,
            "total_time": timer.get_elapsed(),
            "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
            "run": self.run_name,
            "initial_pr": initial_policy_uri,
        }

        # Only include evaluation metadata if we have meaningful scores
        # (i.e., when local evaluation was performed on the current machine, not when remote evaluation was requested)
        has_meaningful_scores = bool(evals.category_scores or evals.simulation_scores)
        if has_meaningful_scores:
            # Extract average reward and scores from evals
            evals_dict = {
                "category_scores": evals.category_scores.copy(),
                "simulation_scores": {f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()},
                "avg_category_score": evals.avg_category_score,
                "avg_simulation_score": evals.avg_simulation_score,
            }

            metadata.update(
                {
                    "evals": evals_dict,
                    "avg_reward": evals.avg_category_score,
                    "score": evals.avg_simulation_score,  # Aggregated score for sweep evaluation
                }
            )
            logger.info(
                f"Including evaluation scores in policy metadata: "
                f"avg_reward={evals.avg_category_score:.4f}, score={evals.avg_simulation_score:.4f}"
            )
        else:
            logger.info(
                "No meaningful evaluation scores available - skipping eval metadata (likely using remote evaluation)"
            )

        # Save original feature mapping
        if isinstance(policy_to_save, MettaAgent):
            original_feature_mapping = policy_to_save.get_original_feature_mapping()
            if original_feature_mapping is not None:
                metadata["original_feature_mapping"] = original_feature_mapping
                logger.info(
                    f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata"
                )

        # Create and save policy record
        path = PolicyStore.make_model_path(self.checkpoint_cfg.checkpoint_dir, name)
        policy_record = self.initializer.create_policy_record(
            self.policy_store, path, policy_to_save, self.checkpoint_cfg.checkpoint_file_type, metadata
        )

        # Save the policy record
        policy_record = self.policy_store.save(policy_record, self.checkpoint_cfg.checkpoint_file_type)

        logger.info(f"Successfully saved policy at epoch {epoch}")

        return policy_record


def maybe_establish_checkpoint(
    checkpoint_manager: CheckpointManager,
    epoch: int,
    policy: PolicyAgent,
    agent_step: int,
    eval_scores: EvalRewardSummary,
    timer: Stopwatch,
    initial_policy_uri: str | None,
    optimizer: torch.optim.Optimizer,
    run_dir: str,
    kickstarter: Kickstarter | None,
    wandb_run: WandbRun | None,
    force: bool = False,
) -> tuple[PolicyRecord, str | None] | None:
    cfg = checkpoint_manager.checkpoint_cfg

    if not should_run(epoch, cfg.checkpoint_interval, force=force):
        return None

    record_heartbeat()

    logger.info(f"Saving checkpoint at epoch {epoch}")
    new_record = checkpoint_manager.save_policy(
        policy=policy,
        epoch=epoch,
        agent_step=agent_step,
        evals=eval_scores,
        timer=timer,
        initial_policy_uri=initial_policy_uri,
    )
    if not new_record.uri:
        # We shouldn't get here
        logger.warning(f"Saved policy record did not have a uri: {new_record}")
        return None

    logger.info(f"Creating a checkpoint at {new_record.uri}")
    record_heartbeat()
    checkpoint_manager.save_checkpoint(
        agent_step=agent_step,
        epoch=epoch,
        optimizer=optimizer,
        policy_path=new_record.uri,
        timer=timer,
        run_dir=run_dir,
        kickstarter=kickstarter,
    )

    wandb_policy_name: str | None = None
    # TODO: enforce that wandb_checkpoint_interval is a multiple of checkpoint_interval
    if should_run(epoch, cfg.wandb_checkpoint_interval, force=force):
        record_heartbeat()
        wandb_policy_name = upload_policy_artifact(wandb_run, checkpoint_manager.policy_store, new_record)

    # Clean up old policies every 10 times we write
    if should_run(epoch, cfg.checkpoint_interval * 10, force=force):
        cleanup_old_policies(checkpoint_manager.checkpoint_cfg.checkpoint_dir)

    return new_record, wandb_policy_name
