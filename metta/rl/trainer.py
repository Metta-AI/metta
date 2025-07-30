import asyncio
import logging
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch
import torch.distributed
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from metta.agent.metta_agent import DistributedMettaAgent, make_policy
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.agent.util.distribution_utils import get_from_master
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch, with_instance_timer
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_config import PyPolicyGameConfig
from metta.mettagrid.mettagrid_env import MettaGridEnv, dtype_actions
from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import create_trainer_config
from metta.rl.util.advantage import compute_advantage
from metta.rl.util.batch_utils import (
    calculate_batch_sizes,
    calculate_prioritized_sampling_params,
)
from metta.rl.util.losses import process_minibatch_update
from metta.rl.util.optimization import (
    calculate_explained_variance,
    compute_gradient_stats,
)
from metta.rl.util.policy_management import (
    cleanup_old_policies,
    load_or_initialize_policy,
    validate_policy_environment_match,
)
from metta.rl.util.rollout import (
    get_lstm_config,
    get_observation,
    run_policy_inference,
)
from metta.rl.util.stats import (
    accumulate_rollout_stats,
    build_wandb_stats,
    compute_timing_stats,
    process_training_stats,
)
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.utils import get_or_create_policy_ids, wandb_policy_name_to_uri

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")

# Get rank for logger name
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{rank}-{local_rank}")


class MettaTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        wandb_run: WandbRun | None,
        policy_store: PolicyStore | None,
        sim_suite_config: SimulationSuiteConfig,
        stats_client: StatsClient | None,
        **kwargs: Any,
    ):
        logger.info(f"run_dir = {cfg.run_dir}")
        checkpoints_dir = Path(cfg.run_dir) / "checkpoints"
        if checkpoints_dir.exists():
            files = sorted(os.listdir(checkpoints_dir))
            recent_files = files[-3:] if len(files) >= 3 else files
            logger.info(f"Recent checkpoints: {', '.join(recent_files)}")

        self.cfg = cfg
        self.trainer_cfg = trainer_cfg = create_trainer_config(cfg)

        # self.hyperparameter_scheduler = HyperparameterScheduler(
        # trainer_cfg, self, trainer_cfg.total_timesteps, logging)

        if trainer_cfg.checkpoint.checkpoint_dir:
            os.makedirs(trainer_cfg.checkpoint.checkpoint_dir, exist_ok=True)

        self._sim_suite_config = sim_suite_config
        self._stats_client = stats_client

        if torch.distributed.is_initialized():
            self._master = torch.distributed.get_rank() == 0
            self._world_size = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()
            logger.info(
                f"Rank: {os.environ['RANK']}, Local rank: {os.environ['LOCAL_RANK']}, World size: {self._world_size}"
            )
        else:
            self._master = True
            self._world_size = 1
            self._rank = 0

        self.device: torch.device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        self._batch_size = trainer_cfg.batch_size
        self._minibatch_size = trainer_cfg.minibatch_size

        self.torch_profiler = TorchProfiler(self._master, trainer_cfg.profiler, wandb_run, cfg.run_dir)
        self.losses = Losses()
        self.stats = defaultdict(list)
        self.grad_stats = {}
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.evals = EvalRewardSummary()

        self.timer = Stopwatch(logger)
        self.timer.start()

        if self._master:
            self._memory_monitor = MemoryMonitor()
            self._system_monitor = SystemMonitor(
                sampling_interval_sec=1.0,  # Sample every second
                history_size=100,  # Keep last 100 samples
                logger=logger,
                auto_start=True,  # Start monitoring immediately
                external_timer=self.timer,  # Pass trainer's timer for persistent elapsed time
            )

        curriculum_config = trainer_cfg.curriculum_or_env
        env_overrides = DictConfig(trainer_cfg.env_overrides)
        self._curriculum = curriculum_from_config_path(curriculum_config, env_overrides)

        # Add training task to the suite
        self._sim_suite_config.simulations["eval/training_task"] = SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",  # won't be used, dynamic `env_cfg()` should override all of it
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        self._make_vecenv()

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
        )

        self.agent_step: int = 0
        self.epoch: int = 0

        checkpoint = TrainerCheckpoint.load(cfg.run_dir)
        if checkpoint:
            logger.info(f"Restoring from checkpoint at {checkpoint.agent_step} steps")
            self.agent_step = checkpoint.agent_step
            self.epoch = checkpoint.epoch
            if checkpoint.stopwatch_state is not None:
                logger.info("Restoring timer state from checkpoint")
                self.timer.load_state(checkpoint.stopwatch_state, resume_running=True)

        # At this point we want to load the policy on the master rank and just initialize a random policy
        # on the other ranks.
        self.policy, self.initial_policy_record, self.latest_saved_policy_record = None, None, None
        if self._master:
            # we only load all these things on master
            self.policy, self.initial_policy_record, self.latest_saved_policy_record = load_or_initialize_policy(
                self.cfg,
                self.policy_store,
                metta_grid_env,
                self.device,
            )

        # We have to make sure the initial and latest saved policy records are the same on all ranks.
        self.policy = get_from_master(self.policy)
        self.initial_policy_record = get_from_master(self.initial_policy_record)
        self.latest_saved_policy_record = get_from_master(self.latest_saved_policy_record)

        if self._master:
            self._log_master(f"MettaTrainer loaded: {self.policy}")

        if trainer_cfg.compile:
            self._log_master("Compiling policy")
            self.policy = torch.compile(self.policy, mode=trainer_cfg.compile_mode)

        # FIX BEFORE MERGE
        # if self._master and self.policy_store is not None:
        #     self.kickstarter = Kickstarter(
        #         trainer_cfg.kickstart,
        #         self.device,
        #         self.policy_store,
        #         metta_grid_env,
        #     )

        if torch.distributed.is_initialized():
            logger.debug(f"Initializing DistributedDataParallel on device {self.device}")
            self.policy = DistributedMettaAgent(self.policy, self.device)
            # Ensure all ranks have initialized DDP before proceeding
            torch.distributed.barrier()

        self._make_experience_buffer()

        self._stats_epoch_start = self.epoch
        self._stats_epoch_id: UUID | None = None
        self._stats_run_id: UUID | None = None

        # Optimizer
        optimizer_type = trainer_cfg.optimizer.type
        assert optimizer_type in ("adam", "muon"), f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}"
        opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=trainer_cfg.optimizer.weight_decay,
        )

        # Validate that policy matches environment
        validate_policy_environment_match(self.policy, metta_grid_env)

        if checkpoint and checkpoint.optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                self._log_master("Successfully loaded optimizer state from checkpoint")
            except ValueError:
                self._log_master("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

        if wandb_run and self._master:
            # Define metrics (wandb x-axis values)
            metrics = ["agent_step", "epoch", "total_time", "train_time"]
            for metric in metrics:
                wandb_run.define_metric(f"metric/{metric}")

            # set the default x-axis to be step count
            wandb_run.define_metric("*", step_metric="metric/agent_step")

            # set up plots that do not use steps as the x-axis
            metric_overrides = [
                ("overview/reward_vs_total_time", "metric/total_time"),
            ]

            for metric_name, step_metric in metric_overrides:
                wandb_run.define_metric(metric_name, step_metric=step_metric)

            # Log model parameters
            num_params = sum(p.numel() for p in self.policy.parameters())
            if wandb_run.summary:
                wandb_run.summary["model/total_parameters"] = num_params

        if self._master:
            self._memory_monitor.add(self, name="MettaTrainer", track_attributes=True)

        self._log_master(f"MettaTrainer initialization complete on device: {self.device}")

    def _log_master(self, message: str, level: str = "info") -> None:
        """Log a message only on the master node.

        Args:
            message: The message to log
            level: The log level ('debug', 'info', 'warning', 'error', 'critical')
        """
        if not self._master:
            return

        log_func = getattr(logger, level, logger.info)
        log_func(message)

    def train(self) -> None:
        self._log_master("Starting training")
        trainer_cfg = self.trainer_cfg

        if self._stats_client is not None:
            if self.wandb_run is not None:
                name = self.wandb_run.name if self.wandb_run.name is not None else "unknown"
                url = self.wandb_run.url
                tags: list[str] | None = list(self.wandb_run.tags) if self.wandb_run.tags is not None else None
                description = self.wandb_run.notes
            else:
                name = "unknown"
                url = None
                tags = None
                description = None

            try:
                self._stats_run_id = self._stats_client.create_training_run(
                    name=name, attributes={}, url=url, description=description, tags=tags
                ).id
            except Exception as e:
                logger.warning(f"Failed to create training run: {e}")

        self._log_master(f"Training on {self.device}")
        wandb_policy_name: str | None = None
        while self.agent_step < trainer_cfg.total_timesteps:
            steps_before = self.agent_step
            record_heartbeat()

            with self.torch_profiler:
                self._rollout()
                self._train()

            self.torch_profiler.on_epoch_end(self.epoch)

            # Processing stats
            self._process_stats()

            self._log_status(steps_before)

            # Interval periodic tasks
            self._maybe_save_policy()
            self._maybe_save_training_state()
            wandb_policy_name = self._maybe_upload_policy_record_to_wandb()
            self._maybe_evaluate_policy(wandb_policy_name)
            self._maybe_compute_grad_stats()

            self._on_train_step()
            # end loop over total_timesteps

        self._log_master("Training complete!")
        timing_summary = self.timer.get_all_summaries()

        for name, summary in timing_summary.items():
            self._log_master(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        # Force final saves
        self._maybe_save_policy(force=True)
        self._maybe_save_training_state(force=True)
        wandb_policy_name = self._maybe_upload_policy_record_to_wandb(force=True)

        if self._stats_epoch_start < self.epoch:
            # If we have not just evaluated the latest policy, evaluate it
            self._maybe_evaluate_policy(wandb_policy_name, force=True)

    def _on_train_step(self):
        pass

    def _log_status(self, steps_before: int):
        if not self._master:
            return

        rollout_time = self.timer.get_last_elapsed("_rollout")
        train_time = self.timer.get_last_elapsed("_train")
        stats_time = self.timer.get_last_elapsed("_process_stats")
        steps_calculated = self.agent_step - steps_before

        total_time = train_time + rollout_time + stats_time
        steps_per_sec = steps_calculated / total_time

        train_pct = (train_time / total_time) * 100
        rollout_pct = (rollout_time / total_time) * 100
        stats_pct = (stats_time / total_time) * 100

        console = Console()
        table = Table(
            title=f"[bold cyan]Training Progress - Epoch {self.epoch}[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Progress", style="green", justify="right")
        table.add_column("Rate", style="yellow", justify="left")

        # Add rows
        progress_pct = (self.agent_step / self.trainer_cfg.total_timesteps) * 100
        table.add_row(
            "Agent Steps",
            f"{self.agent_step:,} / {self.trainer_cfg.total_timesteps:,} ({progress_pct:.1f}%)",
            f"[dim]{steps_per_sec:.0f} steps/sec[/dim]",
        )

        table.add_row(
            "Epoch Time",
            f"{total_time:.2f}s",
            f"[dim]Train: {train_pct:.0f}% | Rollout: {rollout_pct:.0f}% | Stats: {stats_pct:.0f}%[/dim]",
        )

        # Log the table
        console.print(table)

    @with_instance_timer("_rollout")
    def _rollout(self):
        """Perform rollout phase of training."""
        experience = self.experience
        trainer_cfg = self.trainer_cfg

        raw_infos = []  # Collect raw info for batch processing later
        experience.reset_for_rollout()

        while not experience.ready_for_training:
            # Check for contiguous env ids constraint
            if trainer_cfg.require_contiguous_env_ids:
                raise ValueError(
                    "We are assuming contiguous eng id is always False. async_factor == num_workers = "
                    f"{trainer_cfg.async_factor} != {trainer_cfg.num_workers}"
                )

            # Perform single rollout step
            # Receive environment data
            o, r, d, t, info, training_env_id, mask, num_steps = get_observation(self.vecenv, self.device, self.timer)
            self.agent_step += num_steps * self._world_size

            # Run policy inference
            actions, selected_action_log_probs, values, lstm_state_to_store = run_policy_inference(
                self.policy, o, experience, training_env_id.start, self.device
            )

            # Store experience
            experience.store(
                obs=o,
                actions=actions,
                logprobs=selected_action_log_probs,
                rewards=r,
                dones=d,
                truncations=t,
                values=values,
                env_id=training_env_id,
                mask=mask,
                lstm_state=lstm_state_to_store,
            )

            # Send actions back to environment
            with self.timer("_rollout.env"):
                self.vecenv.send(actions.cpu().numpy().astype(dtype_actions))

            # Collect info for batch processing
            if info:
                raw_infos.extend(info)

        # Batch process info dictionaries after rollout
        accumulate_rollout_stats(raw_infos, self.stats)

        # TODO: Better way to enable multiple collects
        return self.stats, self.stats

    @with_instance_timer("_train")
    def _train(self):
        """Perform training phase."""
        experience = self.experience
        trainer_cfg = self.trainer_cfg

        self.losses.zero()

        prio_cfg = trainer_cfg.prioritized_experience_replay
        vtrace_cfg = trainer_cfg.vtrace

        # Reset importance sampling ratios
        experience.reset_importance_sampling_ratios()

        # Prioritized sampling parameters
        anneal_beta = calculate_prioritized_sampling_params(
            epoch=self.epoch,
            total_timesteps=trainer_cfg.total_timesteps,
            batch_size=trainer_cfg.batch_size,
            prio_alpha=prio_cfg.prio_alpha,
            prio_beta0=prio_cfg.prio_beta0,
        )

        # Compute advantages using puff_advantage
        advantages = torch.zeros(experience.values.shape, device=self.device)

        # Initial importance sampling ratio is all ones
        initial_importance_sampling_ratio = torch.ones_like(experience.values)

        advantages = compute_advantage(
            experience.values,
            experience.rewards,
            experience.dones,
            initial_importance_sampling_ratio,
            advantages,
            trainer_cfg.ppo.gamma,
            trainer_cfg.ppo.gae_lambda,
            vtrace_cfg.vtrace_rho_clip,
            vtrace_cfg.vtrace_c_clip,
            self.device,
        )

        # Optimizing the policy and value network
        _total_minibatches = experience.num_minibatches * trainer_cfg.update_epochs
        minibatch_idx = 0

        for _epoch in range(trainer_cfg.update_epochs):
            for _ in range(experience.num_minibatches):
                minibatch = experience.sample_minibatch(
                    advantages=advantages,
                    prio_alpha=prio_cfg.prio_alpha,
                    prio_beta=anneal_beta,
                    minibatch_idx=minibatch_idx,
                    total_minibatches=_total_minibatches,
                )

                # Use the helper function to process minibatch update
                loss = process_minibatch_update(
                    policy=self.policy,
                    experience=experience,
                    minibatch=minibatch,
                    advantages=advantages,
                    trainer_cfg=trainer_cfg,
                    kickstarter=self.kickstarter,
                    agent_step=self.agent_step,
                    losses=self.losses,
                    device=self.device,
                )
                self.optimizer.zero_grad()
                loss.backward()
                if (minibatch_idx + 1) % self.experience.accumulate_minibatches == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), trainer_cfg.ppo.max_grad_norm)
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if str(self.device).startswith("cuda"):
                        torch.cuda.synchronize()

                minibatch_idx += 1
                # end loop over minibatches

            self.epoch += 1

            # check early exit if we have reached target_kl
            if trainer_cfg.ppo.target_kl is not None:
                average_approx_kl = self.losses.approx_kl_sum / self.losses.minibatches_processed
                if average_approx_kl > trainer_cfg.ppo.target_kl:
                    break
            # end loop over epochs

        # self.hyperparameter_scheduler.step(self.agent_step)

        # Calculate explained variance using helper function
        self.losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    def _should_run(self, interval: int, force: bool = False) -> bool:
        """Check if a periodic task should run based on interval and force flag."""
        if not self._master or not interval:
            return False

        if force:
            return True

        return self.epoch % interval == 0

    def _maybe_save_training_state(self, force=False):
        """Save training state if on checkpoint interval"""
        # Check interval for all ranks to ensure synchronization
        if not force and self.trainer_cfg.checkpoint.checkpoint_interval:
            if self.epoch % self.trainer_cfg.checkpoint.checkpoint_interval != 0:
                return

        # Now all ranks that should save are here
        # Only master saves training state, but all ranks must participate in barrier
        if not self._master:
            # Non-master ranks need to participate in the barrier below
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        extra_args = {}
        if self.kickstarter.enabled and self.kickstarter.teacher_uri is not None:
            extra_args["teacher_pr_uri"] = self.kickstarter.teacher_uri

        checkpoint = TrainerCheckpoint(
            agent_step=self.agent_step,
            epoch=self.epoch,
            optimizer_state_dict=self.optimizer.state_dict(),
            stopwatch_state=self.timer.save_state(),
            policy_path=self.latest_saved_policy_uri,
            extra_args=extra_args,
        )
        checkpoint.save(self.cfg.run_dir)
        self._log_master(f"Saved training state at epoch {self.epoch}")

        # Synchronize all ranks to ensure the checkpoint is fully saved before continuing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _maybe_save_policy(self, force=False):
        """Save policy locally if on checkpoint interval"""
        # Check interval for all ranks to ensure synchronization
        if not force and self.trainer_cfg.checkpoint.checkpoint_interval:
            if self.epoch % self.trainer_cfg.checkpoint.checkpoint_interval != 0:
                return

        # Now all ranks that should save are here
        # Only master saves policies, but all ranks must participate in barrier
        if not self._master:
            # Non-master ranks need to participate in the barrier below
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        name = self.policy_store.make_model_name(self.epoch)

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), "vecenv.driver_env must be a MettaGridEnv"

        training_time = self.timer.get_elapsed("_rollout") + self.timer.get_elapsed("_train")

        category_scores_map = self.evals.category_scores.copy()
        category_score_values = [v for k, v in category_scores_map.items()]
        overall_score = sum(category_score_values) / len(category_score_values) if category_score_values else 0

        metadata = PolicyMetadata(
            agent_step=self.agent_step,
            epoch=self.epoch,
            run=self.cfg.run,
            action_names=metta_grid_env.action_names,
            generation=self.current_policy_generation,
            initial_uri=self.initial_policy_uri,
            train_time=training_time,
            score=overall_score,
            eval_scores=category_scores_map,
        )

        # Extract the actual policy module from distributed wrapper if needed
        if isinstance(self.policy, DistributedMettaAgent):
            policy_to_save = self.policy.module
        else:
            policy_to_save = self.policy

        # Save the original feature mapping in metadata
        if hasattr(policy_to_save, "get_original_feature_mapping"):
            original_feature_mapping = policy_to_save.get_original_feature_mapping()
            if original_feature_mapping is not None:
                metadata["original_feature_mapping"] = original_feature_mapping
                self._log_master(
                    f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata"
                )

        # Create a policy record and assign our current policy to it
        policy_record = self.policy_store.create_empty_policy_record(name)
        policy_record.metadata = metadata
        policy_record.policy = policy_to_save

        # Save the policy
        self.latest_saved_policy_record = self.policy_store.save(policy_record)
        logger.info(f"Successfully saved policy at epoch {self.epoch}")

        # Clean up old policies to prevent disk space issues
        if self.epoch % 10 == 0:  # Clean up every 10 epochs
            cleanup_old_policies(self.trainer_cfg.checkpoint.checkpoint_dir, keep_last_n=5)

        # Synchronize all ranks to ensure the policy is fully saved before continuing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _maybe_upload_policy_record_to_wandb(self, force: bool = False) -> str | None:
        """Upload policy to wandb if on wandb interval"""
        if not self._should_run(self.trainer_cfg.checkpoint.wandb_checkpoint_interval, force):
            return

        if not self.wandb_run:
            return

        if not self.latest_saved_policy_record:
            logger.warning("No policy record to upload to wandb")
            return

        if not self.wandb_run.name:
            logger.warning("No wandb run name was provided")
            return

        result = self.policy_store.add_to_wandb_run(self.wandb_run.name, self.latest_saved_policy_record)
        logger.info(f"Uploaded policy to wandb at epoch {self.epoch}")
        return result

    def _maybe_update_l2_weights(self, force=False):
        """Update L2 init weights if on update interval"""
        if self._should_run(self.cfg.agent.l2_init_weight_update_interval, force):
            self.policy.update_l2_init_weight_copy()

    def _maybe_evaluate_policy(self, wandb_policy_name: str | None, force: bool = False):
        """Evaluate policy if on evaluation interval"""
        if self._should_run(self.trainer_cfg.simulation.evaluate_interval, force):
            try:
                self._evaluate_policy(wandb_policy_name)
            except Exception as e:
                logger.error(f"Error evaluating policy: {e}")
                logger.error(traceback.format_exc())

            self._stats_epoch_start = self.epoch + 1

    @with_instance_timer("_evaluate_policy", log_level=logging.INFO)
    def _evaluate_policy(self, wandb_policy_name: str | None) -> None:
        if self._stats_run_id is not None and self._stats_client is not None:
            self._stats_epoch_id = self._stats_client.create_epoch(
                run_id=self._stats_run_id,
                start_training_epoch=self._stats_epoch_start,
                end_training_epoch=self.epoch,
                attributes={},
            ).id

        if (
            self.trainer_cfg.simulation.evaluate_remote
            and self.wandb_run
            and self._stats_client
            and self.latest_saved_policy_record
            and wandb_policy_name  # ensures it was uploaded to wandb
        ):
            # Need to upload policy artifact to wandb first and make sure our name
            # reflects that in the version
            if ":" not in wandb_policy_name:
                logger.warning(f"Remote evaluation: {wandb_policy_name} does not specify a version")
            else:
                internal_wandb_policy_name, wandb_uri = wandb_policy_name_to_uri(wandb_policy_name)
                stats_server_policy_id = get_or_create_policy_ids(
                    self._stats_client,
                    [(internal_wandb_policy_name, wandb_uri, self.wandb_run.notes)],
                    self._stats_epoch_id,
                ).get(internal_wandb_policy_name)
                if not stats_server_policy_id:
                    logger.warning(f"Remote evaluation: failed to get or register policy ID for {wandb_policy_name}")
                else:
                    task = asyncio.run(
                        self._stats_client.create_task(
                            TaskCreateRequest(
                                policy_id=stats_server_policy_id,
                                git_hash=self.trainer_cfg.simulation.git_hash,
                                sim_suite=self._sim_suite_config.name,
                            )
                        )
                    )
                    logger.info(f"Remote evaluation: created task {task.id} for policy {wandb_policy_name}")
                    # TODO: need policy evaluator to generate replays and push stats to wandb

        logger.info(f"Simulating policy: {self.latest_saved_policy_uri} with extended config including training task")
        evaluation_results = evaluate_policy(
            policy_record=self.latest_saved_policy_record,
            simulation_suite=self._sim_suite_config,
            device=self.device,
            vectorization=self.cfg.vectorization,
            replay_dir=self.trainer_cfg.simulation.replay_dir,  # Pass replay_dir to enable replay generation
            stats_epoch_id=self._stats_epoch_id,
            wandb_policy_name=wandb_policy_name,
            policy_store=self.policy_store,
            stats_client=self._stats_client,
            logger=logger,
        )
        logger.info("Simulation complete")
        self.evals = evaluation_results.scores

        # Get target metric (for logging) from sweep config
        # and write top-level score for policy selection.
        # In sweep_eval, we use the "score" entry in the policy metadata to select the best policy
        target_metric = getattr(self.cfg, "sweep", {}).get("metric", "reward")  # fallback to reward
        category_scores = list(self.evals.category_scores.values())
        if category_scores and self.latest_saved_policy_record:
            self.latest_saved_policy_record.metadata["score"] = float(np.mean(category_scores))
            logger.info(
                f"Set policy metadata score to "
                f"{self.latest_saved_policy_record.metadata['score']} using {target_metric} metric"
            )

        # Generate and upload replay HTML if we have wandb
        if self.wandb_run is not None and evaluation_results.replay_urls:
            self._upload_replay_html(evaluation_results.replay_urls)

    def _upload_replay_html(self, replay_urls: dict[str, list[str]]):
        """Upload replay HTML to wandb"""
        # Create unified HTML with all replay links on a single line
        if replay_urls:
            # Group replays by base name
            replay_groups = {}

            for sim_name, urls in sorted(replay_urls.items()):
                if "training_task" in sim_name:
                    # Training replays
                    if "training" not in replay_groups:
                        replay_groups["training"] = []
                    replay_groups["training"].extend(urls)
                else:
                    # Evaluation replays - clean up the display name
                    display_name = sim_name.replace("eval/", "")
                    if display_name not in replay_groups:
                        replay_groups[display_name] = []
                    replay_groups[display_name].extend(urls)

            # Build HTML with episode numbers
            links = []
            for name, urls in replay_groups.items():
                if len(urls) == 1:
                    # Single episode - just show the name
                    player_url = "https://metta-ai.github.io/metta/?replayUrl=" + urls[0]
                    links.append(f'<a href="{player_url}" target="_blank">{name}</a>')
                else:
                    # Multiple episodes - show with numbers
                    episode_links = []
                    for i, url in enumerate(urls, 1):
                        player_url = "https://metta-ai.github.io/metta/?replayUrl=" + url
                        episode_links.append(f'<a href="{player_url}" target="_blank">{i}</a>')
                    links.append(f"{name} [{' '.join(episode_links)}]")

            # Join all links with " | " separator and add epoch prefix
            html_content = f"epoch {self.epoch}: " + " | ".join(links)
        else:
            html_content = f"epoch {self.epoch}: No replays available."

        # Log the unified HTML with step parameter for wandb's epoch slider
        link_summary = {"replays/all_links": wandb.Html(html_content)}
        self.wandb_run.log(link_summary, step=self.agent_step)

        # Also log individual link for backward compatibility
        if "eval/training_task" in replay_urls and replay_urls["eval/training_task"]:
            training_url = replay_urls["eval/training_task"][0]  # Use first URL for backward compatibility
            player_url = "https://metta-ai.github.io/metta/?replayUrl=" + training_url
            link_summary = {
                "replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {self.epoch})</a>')
            }
            self.wandb_run.log(link_summary, step=self.agent_step)

    @with_instance_timer("_process_stats")
    def _process_stats(self):
        if not self._master or not self.wandb_run:
            self.stats.clear()
            self.grad_stats.clear()
            return

        # Process training stats using shared function
        processed_stats = process_training_stats(
            raw_stats=self.stats,
            losses=self.losses,
            experience=self.experience,
            trainer_config=self.trainer_cfg,
            kickstarter=self.kickstarter,
        )

        # Update self.stats with mean values for consistency
        self.stats = processed_stats["mean_stats"]

        # Compute weight stats if on interval
        weight_stats = {}
        if self.cfg.agent.analyze_weights_interval != 0 and self.epoch % self.cfg.agent.analyze_weights_interval == 0:
            for metrics in self.policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_stats[f"weights/{key}/{name}"] = value

        # Compute timing stats using shared function
        timing_info = compute_timing_stats(timer=self.timer, agent_step=self.agent_step)

        # Build parameters dictionary
        parameters = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_steps": timing_info["epoch_steps"],
            "num_minibatches": self.experience.num_minibatches,
            "generation": self.current_policy_generation,
            "latest_saved_policy_epoch": self.latest_saved_policy_record.metadata.epoch,
        }

        # Include custom stats from trainer config
        if hasattr(self.trainer_cfg, "stats") and hasattr(self.trainer_cfg.stats, "overview"):
            for k, v in self.trainer_cfg.stats.overview.items():
                if k in self.stats:
                    processed_stats["overview"][v] = self.stats[k]

        # Add hyperparameter values

        system_stats = {}  # self._system_monitor.stats()
        memory_stats = {}  # self._memory_monitor.stats()

        # Build complete stats dictionary for wandb
        all_stats = build_wandb_stats(
            processed_stats=processed_stats,
            timing_info=timing_info,
            weight_stats=weight_stats,
            grad_stats=self.grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters=self.hyperparameters,
            evals=self.evals,
            agent_step=self.agent_step,
            epoch=self.epoch,
        )

        self.wandb_run.log(
            all_stats,
            # WandB can automatically increment step on each call to log, but we force the value here
            # to make WandB reject any non-monotonic data points. This hides duplicate data when resuming
            # from checkpoints and keeps graphs clean. The policy is reset to the checkpoint too so the
            # count of steps that contribute to training the saved policies is consistent.
            step=self.agent_step,
        )

        self.stats.clear()
        self.grad_stats.clear()

    def close(self):
        self.vecenv.close()
        if self._master:
            self._memory_monitor.clear()
            self._system_monitor.stop()

    @property
    def hyperparameters(self):
        return {}
        # return {
        #     "learning_rate": self.optimizer.param_groups[0]["lr"],
        #     "ppo_clip_coef": self.trainer_cfg.ppo.clip_coef,
        #     "ppo_vf_clip_coef": self.trainer_cfg.ppo.vf_clip_coef,
        #     "ppo_ent_coef": self.trainer_cfg.ppo.ent_coef,
        #     "ppo_l2_reg_loss_coef": self.trainer_cfg.ppo.l2_reg_loss_coef,
        #     "ppo_l2_init_loss_coef": self.trainer_cfg.ppo.l2_init_loss_coef,
        # }

    @property
    def latest_saved_policy_uri(self) -> str | None:
        """Get the URI of the latest saved policy, if any."""
        if self.latest_saved_policy_record is None:
            return None
        return self.latest_saved_policy_record.uri

    @property
    def initial_policy_uri(self) -> str | None:
        """Get the URI of the initial policy used to start training."""
        if self.initial_policy_record is None:
            return None
        return self.initial_policy_record.uri

    @property
    def current_policy_generation(self) -> int:
        """Get the current generation number of the policy."""
        if self.initial_policy_record is None:
            return 0
        return self.initial_policy_record.metadata.get("generation", 0) + 1

    def _make_experience_buffer(self):
        vecenv = self.vecenv
        trainer_cfg = self.trainer_cfg

        # Get environment info
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents

        # Calculate minibatch parameters
        max_minibatch_size = trainer_cfg.minibatch_size

        # Get LSTM parameters using helper function
        hidden_size, num_lstm_layers = get_lstm_config(self.policy)

        # Create experience buffer
        self.experience = Experience(
            total_agents=total_agents,
            batch_size=self._batch_size,
            bptt_horizon=trainer_cfg.bptt_horizon,
            minibatch_size=self._minibatch_size,
            max_minibatch_size=max_minibatch_size,
            obs_space=obs_space,
            atn_space=atn_space,
            device=self.device,
            hidden_size=hidden_size,
            cpu_offload=trainer_cfg.cpu_offload,
            num_lstm_layers=num_lstm_layers,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

    def _make_vecenv(self):
        """Create a vectorized environment."""
        trainer_cfg = self.trainer_cfg
        task = self._curriculum.get_task()
        env_cfg = task.env_cfg()

        # TODO: relax someday when we support other observation shapes
        try:
            game_cfg_dict = OmegaConf.to_container(env_cfg.game, resolve=True)

            if not isinstance(game_cfg_dict, dict) or not all(isinstance(k, str) for k in game_cfg_dict.keys()):
                raise TypeError("env_cfg.game must be a dict with string keys")

            game_cfg = PyPolicyGameConfig(**game_cfg_dict)  # type: ignore[arg-type]

        except ValidationError as e:
            raise ValueError(f"env_cfg.game is not compatible with agent requirements: {e}") from e

        num_agents = game_cfg.num_agents

        # Calculate batch sizes using helper function
        self.target_batch_size, self.batch_size, num_envs = calculate_batch_sizes(
            forward_pass_minibatch_target_size=trainer_cfg.forward_pass_minibatch_target_size,
            num_agents=num_agents,
            num_workers=trainer_cfg.num_workers,
            async_factor=trainer_cfg.async_factor,
        )

        logger.info(
            f"target_batch_size: {self.target_batch_size} = "
            f"min ({trainer_cfg.forward_pass_minibatch_target_size} // {num_agents} , {trainer_cfg.num_workers})"
        )

        logger.info(
            f"forward_pass_batch_size: {self.batch_size} = "
            f"({self.target_batch_size} // {trainer_cfg.num_workers}) * {trainer_cfg.num_workers}"
        )

        logger.info(f"num_envs: {num_envs}")

        if num_envs < 1:
            logger.error(
                f"num_envs = batch_size ({self.batch_size}) * async_factor ({trainer_cfg.async_factor}) "
                f"is {num_envs}, which is less than 1! (Increase trainer.forward_pass_minibatch_target_size)"
            )

        self.vecenv = make_vecenv(
            self._curriculum,
            self.cfg.vectorization,
            num_envs=num_envs,
            batch_size=self.batch_size,
            num_workers=trainer_cfg.num_workers,
            zero_copy=trainer_cfg.zero_copy,
            is_training=True,
            run_dir=self.cfg.run_dir,
        )

        if self.cfg.seed is None:
            self.cfg.seed = np.random.randint(0, 1000000)

        # Use rank-specific seed for environment reset to ensure different
        # processes generate uncorrelated environments in distributed training
        rank = int(os.environ.get("RANK", 0))
        self.vecenv.async_reset(self.cfg.seed + rank)

    def _load_policy(self, checkpoint: TrainerCheckpoint | None, policy_store) -> PolicyRecord | None:
        """Try to load policy from checkpoint or config. Returns None if not found."""
        trainer_cfg = self.trainer_cfg

        # Try checkpoint first
        if checkpoint and checkpoint.policy_path:
            self._log_master(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            return policy_store.policy_record(checkpoint.policy_path)

        # Try initial_policy from config
        if trainer_cfg.initial_policy and (initial_uri := trainer_cfg.initial_policy.uri) is not None:
            self._log_master(f"Loading initial policy URI: {initial_uri}")
            return policy_store.policy_record(initial_uri)

        # Try default checkpoint path
        policy_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
        if os.path.exists(policy_path):
            self._log_master(f"Loading policy from checkpoint: {policy_path}")
            return policy_store.policy_record(policy_path)

        return None

    def _create_and_save_policy_record(self, policy_store: PolicyStore, env: MettaGridEnv) -> PolicyRecord:
        """Create a new policy and save it."""
        name = policy_store.make_model_name(self.epoch)
        logger.info(f"Creating new policy record: {name}")

        # Create the policy record with a new policy instance
        pr = policy_store.create_empty_policy_record(name)
        pr.policy = make_policy(env, self.cfg)

        # Save the policy record
        saved_pr = policy_store.save(pr)
        logger.info(f"Successfully saved initial policy to {saved_pr.uri}")

        return saved_pr

    def _maybe_compute_grad_stats(self, force=False):
        """Compute and store gradient statistics if on interval."""
        interval = self.trainer_cfg.grad_mean_variance_interval
        if not self._should_run(interval, force):
            return

        with self.timer("grad_stats"):
            self.grad_stats = compute_gradient_stats(self.policy)

    def _initialize_policy_to_environment(self, policy, metta_grid_env, device):
        """Helper method to initialize a policy to the environment using the appropriate interface."""
        if hasattr(policy, "initialize_to_environment"):
            features = metta_grid_env.get_observation_features()
            policy.initialize_to_environment(
                features, metta_grid_env.action_names, metta_grid_env.max_action_args, device
            )
        else:
            policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)
