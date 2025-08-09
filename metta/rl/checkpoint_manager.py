"""Checkpoint management for Metta training."""

import logging
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent, make_policy
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.fs import wait_for_file
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.env_config import EnvConfig
from metta.rl.policy_management import cleanup_old_policies, validate_policy_environment_match
from metta.rl.puffer_policy import PytorchAgent
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import CheckpointConfig, TrainerConfig
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
    ) -> bool:
        """Save trainer checkpoint if needed."""
        # Create checkpoint
        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            policy_path=policy_path,
            stopwatch_state=timer.save_state(),
        )

        # Save checkpoint
        checkpoint.save(run_dir)

        return True

    def save_policy(
        self,
        policy: PolicyAgent,
        epoch: int,
        agent_step: int,
        evals: EvalRewardSummary,
        timer: Stopwatch,
        initial_policy_record: PolicyRecord,
    ) -> PolicyRecord:
        """Save policy with metadata if needed."""

        logger.info(f"Saving policy at epoch {epoch}")

        # Extract the actual policy module from distributed wrapper if needed
        policy_to_save: MettaAgent | PytorchAgent = (
            policy.module if isinstance(policy, DistributedMettaAgent) else policy
        )

        # Build metadata
        name = self.policy_store.make_model_name(epoch)

        # Base metadata without evaluation scores
        metadata = {
            "epoch": epoch,
            "agent_step": agent_step,
            "total_time": timer.get_elapsed(),
            "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
            "run": self.run_name,
            "initial_pr": initial_policy_record.uri if initial_policy_record else None,
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
        policy_record = self.policy_store.create_empty_policy_record(
            name=name, checkpoint_dir=self.checkpoint_cfg.checkpoint_dir
        )
        policy_record.metadata = metadata
        policy_record.policy = policy_to_save

        saved_policy_record = self.policy_store.save(policy_record)
        logger.info(f"Successfully saved policy at epoch {epoch}")

        return saved_policy_record

    def load_or_create_policy(
        self,
        agent_cfg: DictConfig,
        env_cfg: EnvConfig,
        trainer_cfg: TrainerConfig,
        checkpoint: TrainerCheckpoint | None,
        metta_grid_env: MettaGridEnv,
    ) -> PolicyRecord:
        """
        Load or initialize policy with distributed coordination.

        First, checks if there is an existing policy at any of:
            - checkpoint.policy_path
            - trainer_cfg.initial_policy.uri
            - default_path (checkpoint_dir/model_{epoch}.pt)
        If so, returns the policy record.

        If not, then distributed workers wait until the master creates the policy at default_path,
        and the master creates a new policy record and saves it to default_path.
        """

        # Check if policy already exists at default path - all ranks check this
        default_model_name = self.policy_store.make_model_name(0)
        default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, default_model_name)

        # First priority: checkpoint
        policy_record: PolicyRecord | None = None
        policy_path: str | None = (
            (checkpoint and checkpoint.policy_path)
            or (trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri)
            or (default_path if os.path.exists(default_path) else None)
        )
        if policy_path:
            logger.info(f"Loading policy from {policy_path}")
            policy_record = self.policy_store.policy_record(policy_path)
        elif self.is_master:
            logger.info("No existing policy found, creating new one")
            new_policy_record = self.policy_store.create_empty_policy_record(
                checkpoint_dir=trainer_cfg.checkpoint.checkpoint_dir, name=default_model_name
            )
            new_policy_record.policy = make_policy(metta_grid_env, env_cfg, agent_cfg)
            policy_record = self.policy_store.save(new_policy_record)
            logger.info(f"Created and saved new policy to {policy_record.uri}")

        elif torch.distributed.is_initialized():
            logger.info(
                f"No existing policy found. Rank {self.rank}: Waiting for master to create policy at {default_path}"
            )

            def log_progress(elapsed: float, status: str) -> None:
                if status == "waiting" and int(elapsed) % 10 == 0 and elapsed > 0:
                    logger.info(f"Rank {self.rank}: Still waiting for policy file... ({elapsed:.0f}s elapsed)")
                elif status == "found":
                    logger.info(f"Rank {self.rank}: Policy file found, waiting for write to complete...")
                elif status == "stable":
                    logger.info(f"Rank {self.rank}: Policy file stable after {elapsed:.1f}s")

            if not wait_for_file(default_path, timeout=300, progress_callback=log_progress):
                raise RuntimeError(f"Rank {self.rank}: Timeout waiting for policy at {default_path}")

            try:
                policy_record = self.policy_store.policy_record(default_path)
            except Exception as e:
                raise RuntimeError(f"Rank {self.rank}: Failed to load policy from {default_path}: {e}") from e
        else:
            raise RuntimeError(f"Non-master rank {self.rank} found without torch.distributed initialized")

        validate_policy_environment_match(policy_record.policy, metta_grid_env)
        return policy_record


def maybe_establish_checkpoint(
    checkpoint_manager: CheckpointManager,
    epoch: int,
    policy: PolicyAgent,
    agent_step: int,
    eval_scores: EvalRewardSummary,
    timer: Stopwatch,
    initial_policy_record: PolicyRecord,
    optimizer: torch.optim.Optimizer,
    run_dir: str,
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
        initial_policy_record=initial_policy_record,
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
