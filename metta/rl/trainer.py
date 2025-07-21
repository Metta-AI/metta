import logging
import os
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Set
from uuid import UUID

import numpy as np
import torch
import torch.distributed
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast

from metta.agent.metta_agent import DistributedMettaAgent, make_policy
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch, with_instance_timer
from metta.common.util.fs import wait_for_file
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_stats_db import EvalStatsDB
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functions import (
    accumulate_rollout_stats,
    calculate_batch_sizes,
    calculate_explained_variance,
    calculate_prioritized_sampling_params,
    cleanup_old_policies,
    compute_advantage,
    compute_gradient_stats,
    get_lstm_config,
    perform_rollout_step,
    process_minibatch_update,
    validate_policy_environment_match,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import parse_trainer_config
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite

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
        policy_store: PolicyStore,
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
        self.trainer_cfg = trainer_cfg = parse_trainer_config(cfg)

        # it doesn't make sense to evaluate more often than we checkpoint since we need a saved policy to evaluate
        if (
            trainer_cfg.simulation.evaluate_interval != 0
            and trainer_cfg.simulation.evaluate_interval < trainer_cfg.checkpoint.checkpoint_interval
        ):
            raise ValueError(
                f"evaluate_interval must be at least as large as checkpoint_interval "
                f"({trainer_cfg.simulation.evaluate_interval} < {trainer_cfg.checkpoint.checkpoint_interval})"
            )
        if (
            trainer_cfg.simulation.evaluate_interval != 0
            and trainer_cfg.simulation.evaluate_interval < trainer_cfg.checkpoint.wandb_checkpoint_interval
        ):
            raise ValueError(
                f"evaluate_interval must be at least as large as wandb_checkpoint_interval "
                f"({trainer_cfg.simulation.evaluate_interval} < {trainer_cfg.checkpoint.wandb_checkpoint_interval})"
            )
        # Validate that we save policies locally at least as often as we upload to wandb
        if (
            trainer_cfg.checkpoint.wandb_checkpoint_interval != 0
            and trainer_cfg.checkpoint.checkpoint_interval != 0
            and trainer_cfg.checkpoint.wandb_checkpoint_interval < trainer_cfg.checkpoint.checkpoint_interval
        ):
            raise ValueError(
                f"wandb_checkpoint_interval must be at least as large as checkpoint_interval "
                f"to ensure policies exist locally before uploading to wandb "
                f"({trainer_cfg.checkpoint.wandb_checkpoint_interval} < {trainer_cfg.checkpoint.checkpoint_interval})"
            )

        if trainer_cfg.checkpoint.checkpoint_dir:
            os.makedirs(trainer_cfg.checkpoint.checkpoint_dir, exist_ok=True)

        self.sim_suite_config = sim_suite_config
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

        self.use_amp = trainer_cfg.use_amp

        if self.use_amp and not str(self.device).startswith("cuda"):
            raise ValueError("AMP requires a CUDA device, but device is not CUDA.")

        if self.use_amp:
            self.scaler = GradScaler("cuda")
        else:
            self.scaler = None

        self.torch_profiler = TorchProfiler(self._master, trainer_cfg.profiler, wandb_run, cfg.run_dir)
        self.losses = Losses()
        self.stats = defaultdict(list)
        self.grad_stats = {}
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.evals: dict[str, float] = {}

        self.timer = Stopwatch(logger)
        self.timer.start()

        if self._master:
            self._memory_monitor = MemoryMonitor()
            self._system_monitor = SystemMonitor(
                sampling_interval_sec=1.0,  # Sample every second
                history_size=100,  # Keep last 100 samples
                logger=logger,
                auto_start=True,  # Start monitoring immediately
            )

        curriculum_config = trainer_cfg.curriculum_or_env
        env_overrides = DictConfig(trainer_cfg.env_overrides)
        self._curriculum = curriculum_from_config_path(curriculum_config, env_overrides)
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

        # Note that these fields are specific to MettaGridEnv, which is why we can't keep
        # self.vecenv.driver_env as just the parent class pufferlib.PufferEnv
        actions_names = metta_grid_env.action_names
        actions_max_params = metta_grid_env.max_action_args

        # Load or create policy with distributed coordination
        policy_record = self._load_policy(checkpoint, policy_store)

        if policy_record is not None:
            logging.info(f"Rank {self._rank}: LOADED {policy_record.uri}")
            self.latest_saved_policy_record = policy_record

            # Get the policy from the record
            self.policy = policy_record.policy

            # Restore original_feature_mapping from metadata if available
            if (
                hasattr(self.policy, "restore_original_feature_mapping")
                and "original_feature_mapping" in policy_record.metadata
            ):
                self.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
                logger.info(f"Rank {self._rank}: Restored original_feature_mapping")

            # Initialize the policy to the environment
            self._initialize_policy_to_environment(self.policy, metta_grid_env, self.device)

            self.initial_policy_record = policy_record

        else:
            logger.info(f"Rank {self._rank}: No existing policy found, creating new one")
            # In distributed mode, handle policy creation/loading differently
            if torch.distributed.is_initialized() and not self._master:
                # Non-master ranks wait for master to create and save the policy
                default_policy_path = os.path.join(
                    trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0)
                )
                logger.info(f"Rank {self._rank}: Waiting for master to create policy at {default_policy_path}")

                # Synchronize with master before attempting to load
                torch.distributed.barrier()

                def log_progress(elapsed: float, status: str) -> None:
                    if status == "waiting" and int(elapsed) % 10 == 0 and elapsed > 0:
                        logger.info(f"Rank {self._rank}: Still waiting for policy file... ({elapsed:.0f}s elapsed)")
                    elif status == "found":
                        logger.info(f"Rank {self._rank}: Policy file found, waiting for write to complete...")
                    elif status == "stable":
                        logger.info(f"Rank {self._rank}: Policy file stable after {elapsed:.1f}s")

                if not wait_for_file(default_policy_path, timeout=300, progress_callback=log_progress):
                    raise RuntimeError(f"Rank {self._rank}: Timeout waiting for policy at {default_policy_path}")

                try:
                    policy_record = self.policy_store.policy_record(default_policy_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Rank {self._rank}: Failed to load policy from {default_policy_path}: {e}"
                    ) from e

                self.initial_policy_record = policy_record
                self.latest_saved_policy_record = policy_record
                self.policy = policy_record.policy

                self._initialize_policy_to_environment(self.policy, metta_grid_env, self.device)
            else:
                # Master creates and saves new policy
                policy_record = self._create_and_save_policy_record(policy_store, metta_grid_env)
                self.initial_policy_record = policy_record
                self.latest_saved_policy_record = policy_record
                self.policy = policy_record.policy

                self._initialize_policy_to_environment(self.policy, metta_grid_env, self.device)

                # Synchronize with non-master ranks after saving
                if torch.distributed.is_initialized():
                    logger.info("Master rank: Policy saved, synchronizing with other ranks")
                    torch.distributed.barrier()

        logging.info(f"Rank {self._rank}: USING {self.initial_policy_record.uri}")

        if self._master:
            logger.info(f"MettaTrainer loaded: {self.policy}")

        if trainer_cfg.compile:
            logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=trainer_cfg.compile_mode)

        self.kickstarter = Kickstarter(
            trainer_cfg.kickstart,
            self.device,
            policy_store,
            actions_names,
            actions_max_params,
        )

        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
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

        self.lr_scheduler = None
        if trainer_cfg.lr_scheduler.enabled:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=trainer_cfg.total_timesteps // trainer_cfg.batch_size
            )

        if checkpoint and checkpoint.optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                logger.info("Successfully loaded optimizer state from checkpoint")
            except ValueError:
                logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

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

        if self._master:
            self._memory_monitor.add(self, name="MettaTrainer", track_attributes=True)

        logger.info(f"MettaTrainer initialization complete on device: {self.device}")

    def train(self) -> None:
        logger.info("Starting training")
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

        logger.info(f"Training on {self.device}")
        wandb_policy_name: str | None = None
        while self.agent_step < trainer_cfg.total_timesteps:
            steps_before = self.agent_step

            with self.torch_profiler:
                self._rollout()
                self._train()

            self.torch_profiler.on_epoch_end(self.epoch)

            # Processing stats
            self._process_stats()

            rollout_time = self.timer.get_last_elapsed("_rollout")
            train_time = self.timer.get_last_elapsed("_train")
            stats_time = self.timer.get_last_elapsed("_process_stats")
            steps_calculated = self.agent_step - steps_before

            total_time = train_time + rollout_time + stats_time
            steps_per_sec = steps_calculated / total_time

            train_pct = (train_time / total_time) * 100
            rollout_pct = (rollout_time / total_time) * 100
            stats_pct = (stats_time / total_time) * 100

            logger.info(
                f"Epoch {self.epoch} - "
                f"{steps_per_sec * self._world_size:.0f} steps/sec "
                f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
            )

            # Interval periodic tasks
            self._maybe_record_heartbeat()
            self._maybe_save_policy()
            self._maybe_save_training_state()
            wandb_policy_name = self._maybe_upload_policy_record_to_wandb()
            self._maybe_evaluate_policy(wandb_policy_name)
            self._maybe_generate_replay()
            self._maybe_compute_grad_stats()

            self._on_train_step()
            # end loop over total_timesteps

        logger.info("Training complete!")
        timing_summary = self.timer.get_all_summaries()

        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        # Force final saves
        self._maybe_save_policy(force=True)
        self._maybe_save_training_state(force=True)
        self._maybe_upload_policy_record_to_wandb(force=True)

    def _on_train_step(self):
        pass

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
            num_steps, info = perform_rollout_step(self.policy, self.vecenv, experience, self.device, self.timer)

            self.agent_step += num_steps

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

        amp_context = autocast if self.use_amp else nullcontext

        for _epoch in range(trainer_cfg.update_epochs):
            for _ in range(experience.num_minibatches):
                with amp_context(device_type="cuda"):
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

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation check
                if (minibatch_idx + 1) % self.experience.accumulate_minibatches == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), trainer_cfg.ppo.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), trainer_cfg.ppo.max_grad_norm)
                        self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if str(self.device).startswith("cuda"):
                        torch.cuda.synchronize()
                    self.optimizer.zero_grad()

                minibatch_idx += 1

            self.epoch += 1

            if trainer_cfg.ppo.target_kl is not None:
                average_approx_kl = self.losses.approx_kl_sum / self.losses.minibatches_processed
                if average_approx_kl > trainer_cfg.ppo.target_kl:
                    break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    def _should_run(self, interval: int, force: bool = False) -> bool:
        """Check if a periodic task should run based on interval and force flag."""
        if not self._master or not interval:
            return False

        if force:
            return True

        return self.epoch % interval == 0

    def _maybe_record_heartbeat(self, force=False):
        if not self._should_run(10, force):
            return

        record_heartbeat()

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
            total_agent_step=self.agent_step * self._world_size,
            optimizer_state_dict=self.optimizer.state_dict(),
            stopwatch_state=self.timer.save_state(),
            policy_path=self.latest_saved_policy_uri,
            extra_args=extra_args,
        )
        checkpoint.save(self.cfg.run_dir)
        logger.info(f"Saved training state at epoch {self.epoch}")

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

        category_scores_map = {key.split("/")[0]: value for key, value in self.evals.items() if key.endswith("/score")}
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
                logger.info(
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

    def _maybe_evaluate_policy(self, wandb_policy_name: str | None = None, force: bool = False):
        """Evaluate policy if on evaluation interval"""
        if self._should_run(self.trainer_cfg.simulation.evaluate_interval, force):
            try:
                self._evaluate_policy(wandb_policy_name)
            except Exception as e:
                logger.error(f"Error evaluating policy: {e}")
                logger.error(traceback.format_exc())

            self._stats_epoch_start = self.epoch + 1

    @with_instance_timer("_evaluate_policy", log_level=logging.INFO)
    def _evaluate_policy(self, wandb_policy_name: str | None = None):
        if self._stats_run_id is not None and self._stats_client is not None:
            self._stats_epoch_id = self._stats_client.create_epoch(
                run_id=self._stats_run_id,
                start_training_epoch=self._stats_epoch_start,
                end_training_epoch=self.epoch,
                attributes={},
            ).id

        logger.info(f"Simulating policy: {self.latest_saved_policy_uri} with config: {self.sim_suite_config}")
        sim = SimulationSuite(
            config=self.sim_suite_config,
            policy_pr=self.latest_saved_policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            stats_dir="/tmp/stats",
            stats_client=self._stats_client,
            stats_epoch_id=self._stats_epoch_id,
            wandb_policy_name=wandb_policy_name,
        )
        result = sim.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
        logger.info("Simulation complete")

        # Build evaluation metrics
        self.evals = {}  # used for wandb
        categories: Set[str] = set()
        for sim_name in self.sim_suite_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter(
                "reward", self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
            )
            logger.info(f"{category} score: {score}")
            record_heartbeat()
            if score is None:
                continue
            self.evals[f"{category}/score"] = score

        # Get detailed per-simulation scores
        all_scores = stats_db.simulation_scores(self.latest_saved_policy_record, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            self.evals[f"{category}/{sim_short_name}"] = score

    def _maybe_generate_replay(self, force=False):
        """Generate replay if on replay interval"""
        if self._should_run(self.trainer_cfg.simulation.replay_interval, force):
            self._generate_and_upload_replay()

    @with_instance_timer("_generate_and_upload_replay", log_level=logging.INFO)
    def _generate_and_upload_replay(self):
        replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/arena/advanced",
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        replay_simulator = Simulation(
            name=f"replay_{self.epoch}",
            config=replay_sim_config,
            policy_pr=self.latest_saved_policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            replay_dir=self.trainer_cfg.simulation.replay_dir,
        )
        results = replay_simulator.simulate()

        if self.wandb_run is not None:
            key, version = results.stats_db.key_and_version(self.latest_saved_policy_record)
            replay_urls = results.stats_db.get_replay_urls(key, version)
            if len(replay_urls) > 0:
                replay_url = replay_urls[0]
                player_url = "https://metta-ai.github.io/metta/?replayUrl=" + replay_url
                link_summary = {
                    "replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {self.epoch})</a>')
                }
                self.wandb_run.log(link_summary)

    @with_instance_timer("_process_stats")
    def _process_stats(self):
        if not self._master or not self.wandb_run:
            self.stats.clear()
            self.grad_stats.clear()
            return

        # convert lists of values (collected across all environments and rollout steps on this GPU)
        # into single mean values and standard deviations.
        mean_stats = {}
        for k, v in self.stats.items():
            try:
                mean_stats[k] = np.mean(v)
                # Add standard deviation with .std_dev suffix
                # DISABLED(daveey): this is too noisy and so far not useful
                # mean_stats[f"{k}.std_dev"] = np.std(v)
            except (TypeError, ValueError) as e:
                raise RuntimeError(
                    f"Cannot compute mean for stat '{k}' with value {v!r} (type: {type(v)}). "
                    f"All collected stats must be numeric values or lists of numeric values. "
                    f"Error: {e}"
                ) from e
        self.stats = mean_stats

        weight_stats = {}
        if self.cfg.agent.analyze_weights_interval != 0 and self.epoch % self.cfg.agent.analyze_weights_interval == 0:
            for metrics in self.policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_stats[f"weights/{key}/{name}"] = value

        elapsed_times = self.timer.get_all_elapsed()
        wall_time = self.timer.get_elapsed()
        train_time = elapsed_times.get("_rollout", 0) + elapsed_times.get("_train", 0)

        lap_times = self.timer.lap_all(self.agent_step, exclude_global=False)
        wall_time_for_lap = lap_times.pop("global", 0)

        # X-axis values for wandb
        metric_stats = {
            "metric/agent_step": self.agent_step * self._world_size,
            "metric/epoch": self.epoch,
            "metric/total_time": wall_time,
            "metric/train_time": train_time,
        }

        epoch_steps = self.timer.get_lap_steps()
        assert epoch_steps is not None

        epoch_steps_per_second = epoch_steps / wall_time_for_lap if wall_time_for_lap > 0 else 0
        steps_per_second = self.timer.get_rate(self.agent_step) if wall_time > 0 else 0

        epoch_steps_per_second *= self._world_size
        steps_per_second *= self._world_size

        timing_stats = {
            **{
                f"timing_per_epoch/frac/{op}": lap_elapsed / wall_time_for_lap if wall_time_for_lap > 0 else 0
                for op, lap_elapsed in lap_times.items()
            },
            **{
                f"timing_per_epoch/msec/{op}": lap_elapsed * 1000 if wall_time_for_lap > 0 else 0
                for op, lap_elapsed in lap_times.items()
            },
            "timing_per_epoch/sps": epoch_steps_per_second,
            **{
                f"timing_cumulative/frac/{op}": elapsed / wall_time if wall_time > 0 else 0
                for op, elapsed in elapsed_times.items()
            },
            "timing_cumulative/sps": steps_per_second,
        }

        environment_stats = {f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in self.stats.items()}

        overview = {
            "sps": epoch_steps_per_second,
        }

        # Calculate average reward from all env_task_reward entries
        task_reward_values = [v for k, v in environment_stats.items() if k.startswith("env_task_reward")]
        if task_reward_values:
            mean_reward = sum(task_reward_values) / len(task_reward_values)
            overview["reward"] = mean_reward
            overview["reward_vs_total_time"] = mean_reward

        # include custom stats from trainer config
        if hasattr(self.trainer_cfg, "stats") and hasattr(self.trainer_cfg.stats, "overview"):
            for k, v in self.trainer_cfg.stats.overview.items():
                if k in self.stats:
                    overview[v] = self.stats[k]

        category_scores_map = {key.split("/")[0]: value for key, value in self.evals.items() if key.endswith("/score")}

        for category, score in category_scores_map.items():
            overview[f"{category}_score"] = score

        losses = self.losses.stats()

        # don't plot losses that are unused
        if self.trainer_cfg.ppo.l2_reg_loss_coef == 0:
            losses.pop("l2_reg_loss")
        if self.trainer_cfg.ppo.l2_init_loss_coef == 0:
            losses.pop("l2_init_loss")
        if not self.kickstarter.enabled:
            losses.pop("ks_action_loss")
            losses.pop("ks_value_loss")

        parameters = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_steps": epoch_steps,
            "num_minibatches": self.experience.num_minibatches,
            "generation": self.current_policy_generation,
            "latest_saved_policy_epoch": self.latest_saved_policy_record.metadata.epoch,
        }

        self.wandb_run.log(
            {
                **{f"overview/{k}": v for k, v in overview.items()},
                **{f"losses/{k}": v for k, v in losses.items()},
                **{f"experience/{k}": v for k, v in self.experience.stats().items()},
                **{f"parameters/{k}": v for k, v in parameters.items()},
                **{f"eval_{k}": v for k, v in self.evals.items()},
                **{f"monitor/{k}": v for k, v in self._system_monitor.stats().items()},
                **{f"trainer_memory/{k}": v for k, v in self._memory_monitor.stats().items()},
                **environment_stats,
                **weight_stats,
                **timing_stats,
                **metric_stats,
                **self.grad_stats,
            },
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

        num_agents = self._curriculum.get_task().env_cfg().game.num_agents

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
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            return policy_store.policy_record(checkpoint.policy_path)

        # Try initial_policy from config
        if trainer_cfg.initial_policy and (initial_uri := trainer_cfg.initial_policy.uri) is not None:
            logger.info(f"Loading initial policy URI: {initial_uri}")
            return policy_store.policy_record(initial_uri)

        # Try default checkpoint path
        policy_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
        if os.path.exists(policy_path):
            logger.info(f"Loading policy from checkpoint: {policy_path}")
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


class AbortingTrainer(MettaTrainer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _on_train_step(self):
        if self.wandb_run is None:
            return

        if "abort" not in wandb.Api().run(self.wandb_run.path).tags:
            return

        logger.info("Abort tag detected. Stopping the run.")
        self.trainer_cfg.total_timesteps = int(self.agent_step)
        self.wandb_run.config.update(
            {"trainer.total_timesteps": self.trainer_cfg.total_timesteps}, allow_val_change=True
        )
