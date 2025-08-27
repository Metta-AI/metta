"""Checkpoint management for Metta training."""

import logging
import os
from pathlib import Path

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent
from metta.agent.policy_loader import AgentBuilder, PolicyLoader
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.util.distribution_utils import get_from_master
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.collections import remove_none_values
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid import MettaGridEnv
from metta.rl.kickstarter import Kickstarter
from metta.rl.policy_management import cleanup_old_policies, validate_policy_environment_match
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import CheckpointConfig, TrainerConfig
from metta.rl.utils import should_run
from metta.rl.wandb import upload_policy_artifact

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing for both trainer state and policies."""

    def __init__(
        self,
        policy_loader: PolicyLoader,
        checkpoint_config: CheckpointConfig,
        device: torch.device,
        is_master: bool,
        rank: int,
        run_name: str,
    ):
        """Initialize checkpoint manager.

        Args:
            policy_loader: PolicyLoader instance for saving/loading policies
            checkpoint_config: Checkpoint configuration
            device: Training device
            is_master: Whether this is the master process
            rank: Process rank for distributed training
            run_name: Name of the current run
        """
        self.policy_loader = policy_loader
        self.checkpoint_cfg = checkpoint_config
        self.device = device
        self.is_master = is_master
        self.rank = rank
        self.run_name = run_name

        self._ensure_checkpoint_directory()
        self._validate_checkpoint_intervals()

    def _ensure_checkpoint_directory(self) -> None:
        """Ensure checkpoint directory exists."""
        if self.checkpoint_cfg.checkpoint_dir is not None:
            Path(self.checkpoint_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _validate_checkpoint_intervals(self) -> None:
        """Validate that wandb_checkpoint_interval is a multiple of checkpoint_interval."""
        if (
            self.checkpoint_cfg.wandb_checkpoint_interval is not None
            and self.checkpoint_cfg.checkpoint_interval is not None
            and self.checkpoint_cfg.wandb_checkpoint_interval % self.checkpoint_cfg.checkpoint_interval != 0
        ):
            raise ValueError(
                f"wandb_checkpoint_interval ({self.checkpoint_cfg.wandb_checkpoint_interval}) "
                f"must be a multiple of checkpoint_interval ({self.checkpoint_cfg.checkpoint_interval})"
            )

    def make_model_name(self, epoch: int, model_suffix: str) -> str:
        """Create a model name for the given epoch."""
        return f"model_{epoch:04d}{model_suffix}"

    def _save_policy_to_file(self, policy_record: PolicyRecord) -> str:
        """Save policy record to file and return the path."""
        if self.checkpoint_cfg.checkpoint_file_type == "safetensors":
            return self.policy_loader.save_to_safetensors_file(policy_record, None)
        else:
            return self.policy_loader.save_to_pt_file(policy_record, None)

    def _build_base_metadata(
        self, epoch: int, agent_step: int, timer: Stopwatch, initial_policy_uri: str | None
    ) -> dict:
        """Build base metadata without evaluation scores."""
        return {
            "epoch": epoch,
            "agent_step": agent_step,
            "total_time": timer.get_elapsed(),
            "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
            "run": self.run_name,
            "initial_pr": initial_policy_uri,
        }

    def _build_evaluation_metadata(self, evals: EvalRewardSummary) -> dict:
        """Build evaluation metadata if meaningful scores are available."""
        has_meaningful_scores = bool(evals.category_scores or evals.simulation_scores)

        if not has_meaningful_scores:
            logger.info(
                "No meaningful evaluation scores available - skipping eval metadata (likely using remote evaluation)"
            )
            return {}

        evals_dict = {
            "category_scores": evals.category_scores.copy(),
            "simulation_scores": {f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()},
            "avg_category_score": evals.avg_category_score,
            "avg_simulation_score": evals.avg_simulation_score,
        }

        logger.info(
            f"Including evaluation scores in policy metadata: "
            f"avg_reward={evals.avg_category_score:.4f}, score={evals.avg_simulation_score:.4f}"
        )

        return {
            "evals": evals_dict,
            "avg_reward": evals.avg_category_score,
            "score": evals.avg_simulation_score,  # Aggregated score for sweep evaluation
        }

    def _add_feature_mapping_metadata(self, policy: MettaAgent, metadata: dict) -> None:
        """Add original feature mapping to metadata if available."""
        original_feature_mapping = policy.get_original_feature_mapping()
        if original_feature_mapping is not None:
            metadata["original_feature_mapping"] = original_feature_mapping
            logger.info(f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata")

    def _extract_policy_for_saving(self, policy: PolicyAgent) -> MettaAgent:
        """Extract the actual policy module from distributed wrapper if needed."""
        return policy.module if isinstance(policy, DistributedMettaAgent) else policy

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
        policy_to_save: MettaAgent = self._extract_policy_for_saving(policy)

        # Build metadata
        name = self.make_model_name(epoch, self.checkpoint_cfg.model_suffix())
        metadata = self._build_base_metadata(epoch, agent_step, timer, initial_policy_uri)
        metadata.update(self._build_evaluation_metadata(evals))

        # Add feature mapping metadata
        if isinstance(policy_to_save, MettaAgent):
            self._add_feature_mapping_metadata(policy_to_save, metadata)

        # Create and save policy record
        if self.checkpoint_cfg.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in checkpoint_config to save policies")

        policy_record_to_save = PolicyRecord(
            run_name=name,
            uri=f"file://{self.checkpoint_cfg.checkpoint_dir}/{name}",
            metadata=metadata,
            policy=policy_to_save,
        )

        # Save the policy record
        path = self._save_policy_to_file(policy_record_to_save)
        policy_record_to_save.uri = f"file://{path}"

        logger.info(f"Successfully saved policy at epoch {epoch}")
        return policy_record_to_save

    def get_or_create_policy_record(
        self,
        agent_builder: AgentBuilder,
        trainer_cfg: TrainerConfig,  # this arg is only needed to distinguish between pt and safetensors
        policy_path: str,
        should_create: bool,
        metta_grid_env: MettaGridEnv,
    ) -> PolicyRecord:
        """Load or initialize policy with distributed coordination.

        Args:
            agent_builder: Builder for creating new agents
            trainer_cfg: Trainer configuration
            policy_path: Path to the policy (always provided)
            should_create: If True, create new policy; if False, load existing policy
            metta_grid_env: Environment for validation
        """

        # Check for distributed training configuration error early
        if not self.is_master and not torch.distributed.is_initialized():
            raise RuntimeError(
                f"Non-master rank {self.rank} found without torch.distributed initialized. "
                "This likely indicates a configuration error in distributed training setup."
            )
        # from here, not(checkpoint_manager.is_master) == torch.distributed.is_initialized()

        if should_create:
            # Create new policy
            logger.info(f"Rank {self.rank}: Creating new policy")
            agent = agent_builder.initialize_agent(PolicyMetadata(), None)
            policy_record = PolicyRecord(
                run_name=os.path.basename(policy_path),
                uri=f"file://{policy_path}",
                metadata=PolicyMetadata(),
                policy=agent,
            )

            # Only master saves the new policy to disk
            if self.is_master:
                if trainer_cfg.checkpoint.checkpoint_file_type == "safetensors":
                    saved_path = self.policy_loader.save_to_safetensors_file(policy_record, None)
                else:
                    saved_path = self.policy_loader.save_to_pt_file(policy_record, None)
                policy_record.uri = f"file://{saved_path}"
                logger.info(f"Master saved new policy to {policy_record.uri}")
            else:
                logger.info(f"Rank {self.rank}: Created policy structure for DDP sync")
        else:
            # Load existing policy
            logger.info(f"Rank {self.rank}: Loading policy from {policy_path}")
            policy_record = self.policy_loader.load_from_file(policy_path)

        if policy_record is None:
            raise RuntimeError("Failed to initialize policy record")

        # Synchronize policy metadata from master using NCCL broadcast of objects.
        # This avoids file I/O on non-master ranks while ensuring consistent metadata.
        # ?? should this code move to AgentBuilder?
        if not self.is_master:
            try:
                synced_metadata = get_from_master(policy_record.metadata if self.is_master else None)
                if synced_metadata is not None:
                    policy_record.metadata = synced_metadata
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Failed to sync policy metadata from master: {e}")

        validate_policy_environment_match(policy_record.policy, metta_grid_env)
        return policy_record


def _should_cleanup_policies(epoch: int, checkpoint_interval: int, force: bool) -> bool:
    """Check if it's time to cleanup old policies."""
    CLEANUP_MULTIPLIER = 10
    return should_run(epoch, checkpoint_interval * CLEANUP_MULTIPLIER, force=force)


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
    """Establish a checkpoint if conditions are met."""
    cfg = checkpoint_manager.checkpoint_cfg

    if not should_run(epoch, cfg.checkpoint_interval, force=force):
        return None

    record_heartbeat()
    logger.info(f"Saving checkpoint at epoch {epoch}")

    # Save policy
    new_record = checkpoint_manager.save_policy(
        policy=policy,
        epoch=epoch,
        agent_step=agent_step,
        evals=eval_scores,
        timer=timer,
        initial_policy_uri=initial_policy_uri,
    )
    if not new_record.uri:
        logger.warning(f"Saved policy record did not have a uri: {new_record}")
        return None

    # Save trainer checkpoint
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

    # Upload to wandb if needed
    wandb_policy_name: str | None = None
    if should_run(epoch, cfg.wandb_checkpoint_interval, force=force):
        record_heartbeat()
        wandb_policy_name = upload_policy_artifact(wandb_run, checkpoint_manager.policy_loader, new_record)

    # Cleanup old policies
    if _should_cleanup_policies(epoch, cfg.checkpoint_interval, force) and cfg.checkpoint_dir is not None:
        cleanup_old_policies(cfg.checkpoint_dir)

    return new_record, wandb_policy_name
