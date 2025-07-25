"""Main trainer class that orchestrates all components."""

import logging
import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbContext
from metta.interface.agent import create_or_load_agent
from metta.interface.directories import save_experiment_config
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import TrainerConfig
from metta.rl.util.distributed import setup_device_and_distributed
from metta.rl.util.optimization import maybe_update_l2_weights
from metta.rl.util.policy_management import (
    cleanup_old_policies,
    save_policy_with_metadata,
    validate_policy_environment_match,
    wrap_agent_distributed,
)
from metta.rl.util.rollout import get_lstm_config
from metta.rl.util.stats import compute_timing_stats
from metta.rl.util.utils import check_abort, should_run
from metta.rl.wandb import log_model_parameters, setup_wandb_metrics, upload_env_configs

from .environment_manager import EnvironmentManager
from .evaluation_manager import EvaluationManager
from .optimizer_manager import OptimizerManager
from .rollout_manager import RolloutManager
from .stats_manager import StatsManager
from .training_manager import TrainingManager

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class that orchestrates the training workflow using components."""

    def __init__(
        self,
        trainer_config: TrainerConfig,
        run_dir: str,
        run_name: str,
        checkpoint_dir: str,
        replay_dir: str,
        stats_dir: str,
        device: Optional[torch.device] = None,
        wandb_config: Optional[DictConfig] = None,
        global_config: Optional[DictConfig] = None,
        stats_client: Optional[StatsClient] = None,
    ):
        """Initialize trainer with all components.

        Args:
            trainer_config: Training configuration
            run_dir: Directory for this run
            run_name: Name of this run
            checkpoint_dir: Directory for checkpoints
            replay_dir: Directory for replays
            stats_dir: Directory for statistics
            device: Optional device override
            wandb_config: Optional wandb configuration
            global_config: Optional global configuration
            stats_client: Optional stats client for tracking
        """
        self.trainer_config = trainer_config
        self.run_dir = run_dir
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.replay_dir = replay_dir
        self.stats_dir = stats_dir
        self.stats_client = stats_client

        # Set up device and distributed
        if device is None:
            base_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device, self.is_master, self.world_size, self.rank = setup_device_and_distributed(base_device)
        else:
            self.device = device
            self.is_master = int(os.environ.get("RANK", 0)) == 0
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.rank = int(os.environ.get("RANK", 0))

        # Adjust batch sizes for distributed training
        if torch.distributed.is_initialized() and trainer_config.scale_batches_by_world_size:
            trainer_config.batch_size = trainer_config.batch_size // self.world_size

        # Save config
        # Handle both TrainerConfig objects and DictConfig from Hydra
        if hasattr(trainer_config, "model_dump"):
            # It's a TrainerConfig Pydantic model
            save_experiment_config(
                type("Dirs", (), {"run_dir": run_dir, "run_name": run_name}),
                self.device,
                trainer_config,
            )
        else:
            # It's a DictConfig from Hydra - create a temporary object with model_dump method
            temp_config = type(
                "TempConfig", (), {"model_dump": lambda self: OmegaConf.to_container(trainer_config, resolve=True)}
            )()
            save_experiment_config(
                type("Dirs", (), {"run_dir": run_dir, "run_name": run_name}),
                self.device,
                temp_config,
            )

        # Initialize wandb
        self.wandb_run = None
        self.wandb_ctx = None
        if self.is_master and wandb_config and global_config:
            self.wandb_ctx = WandbContext(wandb_config, global_config)
            self.wandb_run = self.wandb_ctx.__enter__()

        # Create policy store
        policy_store_config = self._create_policy_store_config(wandb_config, global_config)
        self.policy_store = PolicyStore(DictConfig(policy_store_config), self.wandb_run)

        # Create component managers
        self.env_manager = EnvironmentManager(trainer_config, self.device)
        self.optimizer_manager = OptimizerManager(trainer_config.optimizer, self.device)

        # Create timer
        self.timer = Stopwatch(logger)

        # These will be initialized in setup()
        self.agent = None
        self.optimizer = None
        self.experience = None
        self.kickstarter = None
        self.losses = None
        self.rollout_manager = None
        self.training_manager = None
        self.stats_manager = None
        self.evaluation_manager = None
        self.torch_profiler = None
        self.system_monitor = None
        self.memory_monitor = None

        # Training state
        self.agent_step = 0
        self.epoch = 0
        self.latest_saved_policy_record = None
        self.initial_policy_record = None
        self.initial_policy_uri = None
        self.initial_generation = 0
        self.current_policy_generation = 0
        self.wandb_policy_name = None

    def _create_policy_store_config(
        self,
        wandb_config: Optional[DictConfig],
        global_config: Optional[DictConfig],
    ) -> dict:
        """Create policy store configuration."""
        config = {
            "device": str(self.device),
            "policy_cache_size": 10,
            "run": self.run_name,
            "run_dir": self.run_dir,
            "vectorization": "serial",  # Will be updated when env is created
            "trainer": self.trainer_config.model_dump()
            if hasattr(self.trainer_config, "model_dump")
            else OmegaConf.to_container(self.trainer_config, resolve=True),
        }

        # Add wandb config if available
        if self.wandb_run and self.wandb_ctx:
            try:
                wandb_cfg = self.wandb_ctx.cfg
                if isinstance(wandb_cfg, DictConfig):
                    wandb_config_dict = OmegaConf.to_container(wandb_cfg, resolve=True)
                    if isinstance(wandb_config_dict, dict) and wandb_config_dict.get("enabled"):
                        config["wandb"] = {
                            "entity": wandb_config_dict.get("entity"),
                            "project": wandb_config_dict.get("project"),
                        }
            except AttributeError:
                pass

        return config

    def setup(self, vectorization: str = "multiprocessing", seed: Optional[int] = None) -> None:
        """Set up all components for training.

        Args:
            vectorization: Vectorization mode for environment
            seed: Optional seed for environment
        """
        self.timer.start()

        # Create environment
        env = self.env_manager.create_environment(
            vectorization=vectorization,
            seed=seed,
            rank=self.rank,
        )
        metta_grid_env = self.env_manager.driver_env

        # Create or load agent
        self.agent, policy_record, self.agent_step, self.epoch, checkpoint = create_or_load_agent(
            env=env,
            run_dir=self.run_dir,
            policy_store=self.policy_store,
            trainer_config=self.trainer_config,
            device=self.device,
            is_master=self.is_master,
            rank=self.rank,
        )

        # Store initial policy info
        self.initial_policy_record = policy_record
        self.latest_saved_policy_record = policy_record
        self.initial_policy_uri = policy_record.uri if policy_record else None
        self.initial_generation = policy_record.metadata.get("generation", 0) if policy_record else 0
        self.current_policy_generation = self.initial_generation + 1 if policy_record else 0

        # Restore timer state if checkpoint exists
        if checkpoint and checkpoint.stopwatch_state is not None:
            self.timer.load_state(checkpoint.stopwatch_state, resume_running=True)

        # Get LSTM config
        hidden_size, num_lstm_layers = get_lstm_config(self.agent)

        # Validate policy matches environment
        validate_policy_environment_match(self.agent, metta_grid_env)

        # Compile policy if configured
        if self.trainer_config.compile:
            logger.info("Compiling policy")
            self.agent = torch.compile(self.agent, mode=self.trainer_config.compile_mode)

        # Create optimizer
        self.optimizer = self.optimizer_manager.create_optimizer(self.agent)
        self.optimizer_manager.load_state_from_checkpoint(self.optimizer, checkpoint)

        # Wrap agent for distributed training
        self.agent = wrap_agent_distributed(self.agent, self.device)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Set up wandb
        if self.wandb_run and self.is_master:
            setup_wandb_metrics(self.wandb_run)
            log_model_parameters(self.agent, self.wandb_run)

            # Upload environment configs
            curr_obj = self.env_manager.get_curriculum()
            if curr_obj is not None and hasattr(curr_obj, "get_env_cfg_by_bucket"):
                env_configs = curr_obj.get_env_cfg_by_bucket()
                upload_env_configs(env_configs=env_configs, wandb_run=self.wandb_run)

        # Log model info
        if self.is_master:
            num_params = sum(p.numel() for p in self.agent.parameters())
            logger.info(f"Model has {num_params:,} parameters")

        # Create experience buffer
        self.experience = Experience(
            total_agents=self.env_manager.num_agents,
            batch_size=self.trainer_config.batch_size,
            bptt_horizon=self.trainer_config.bptt_horizon,
            minibatch_size=self.trainer_config.minibatch_size,
            max_minibatch_size=self.trainer_config.minibatch_size,
            obs_space=env.single_observation_space,
            atn_space=env.single_action_space,
            device=self.device,
            hidden_size=hidden_size,
            cpu_offload=self.trainer_config.cpu_offload,
            num_lstm_layers=num_lstm_layers,
            agents_per_batch=getattr(env, "agents_per_batch", None),
        )

        # Create kickstarter
        self.kickstarter = Kickstarter(
            self.trainer_config.kickstart,
            str(self.device),
            self.policy_store,
            metta_grid_env,
        )

        # Create losses tracker
        self.losses = Losses()

        # Create monitoring (master only)
        if self.is_master:
            self.memory_monitor = MemoryMonitor()
            self.memory_monitor.add(self.experience, name="Experience", track_attributes=True)
            self.memory_monitor.add(self.agent, name="Agent", track_attributes=False)

            self.system_monitor = SystemMonitor(
                sampling_interval_sec=1.0,
                history_size=100,
                logger=logger,
                auto_start=True,
                external_timer=self.timer,
            )

        # Create component managers
        self.rollout_manager = RolloutManager(env, self.device, self.timer)
        self.training_manager = TrainingManager(self.trainer_config, self.device, self.kickstarter)
        self.stats_manager = StatsManager(
            self.trainer_config,
            self.timer,
            self.is_master,
            self.system_monitor,
            self.memory_monitor,
            self.stats_client,
        )
        self.evaluation_manager = EvaluationManager(
            self.trainer_config,
            self.policy_store,
            self.device,
            self.stats_dir,
            self.is_master,
            self.stats_client,
            self.stats_manager.stats_tracker,
        )

        # Create torch profiler
        self.torch_profiler = TorchProfiler(
            self.is_master,
            self.trainer_config.profiler,
            self.wandb_run,
            self.run_dir,
        )

        # Initialize stats tracking
        self.stats_manager.initialize_stats_tracking(self.wandb_run)

    def train_epoch(self) -> Tuple[float, int]:
        """Run one training epoch (rollout + training).

        Returns:
            Tuple of (steps_per_second, agent_step)
        """
        steps_before = self.agent_step

        # Rollout phase
        rollout_start = time.time()
        raw_infos, self.agent_step = self.rollout_manager.collect_rollouts(self.agent, self.experience, self.agent_step)
        rollout_time = time.time() - rollout_start

        # Process rollout statistics
        self.stats_manager.process_rollout_stats(raw_infos)

        # Training phase
        train_start = time.time()
        self.training_manager.train_on_experience(
            self.agent, self.optimizer, self.experience, self.losses, self.epoch, self.agent_step
        )
        train_time = time.time() - train_start
        self.epoch += 1

        self.torch_profiler.on_epoch_end(self.epoch)

        # Stats processing phase
        stats_start = time.time()

        # Compute timing stats
        timing_info = compute_timing_stats(timer=self.timer, agent_step=self.agent_step)

        # Build and log stats
        if self.is_master:
            current_lr = self.optimizer_manager.get_current_lr(self.optimizer)

            # Compute weight stats
            weight_stats = self.stats_manager.compute_weight_stats(self.agent, self.epoch)

            # Build complete stats
            all_stats = self.stats_manager.build_training_stats(
                losses=self.losses,
                experience=self.experience,
                kickstarter=self.kickstarter,
                agent_step=self.agent_step,
                epoch=self.epoch,
                current_lr=current_lr,
                current_policy_generation=self.current_policy_generation,
                timing_info=timing_info,
            )

            # Add weight stats
            all_stats.update(weight_stats)

            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log(all_stats, step=self.agent_step)

        # Clear stats
        self.stats_manager.clear_stats()
        stats_time = time.time() - stats_start

        # Calculate performance
        steps_calculated = self.agent_step - steps_before
        total_time = train_time + rollout_time + stats_time
        steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

        # Log progress
        self._log_progress(steps_per_sec, train_time, rollout_time, stats_time, total_time)

        return steps_per_sec, self.agent_step

    def _log_progress(
        self,
        steps_per_sec: float,
        train_time: float,
        rollout_time: float,
        stats_time: float,
        total_time: float,
    ) -> None:
        """Log training progress."""
        train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
        rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
        stats_pct = (stats_time / total_time) * 100

        total_timesteps = self.trainer_config.total_timesteps
        if total_timesteps >= 1e9:
            total_steps_str = f"{total_timesteps:.0e}"
        else:
            total_steps_str = f"{total_timesteps:,}"

        logger.info(
            f"Epoch {self.epoch}- "
            f"{steps_per_sec:.0f} SPS- "
            f"step {self.agent_step}/{total_steps_str}- "
            f"({train_pct:.0f}% train- {rollout_pct:.0f}% rollout- {stats_pct:.0f}% stats)"
        )

    def checkpoint(self) -> None:
        """Save checkpoint and policy."""
        if not self.is_master:
            return

        # Create temporary initial_policy_record for save_policy_with_metadata
        temp_initial_policy_record = None
        if self.initial_policy_uri:
            temp_initial_policy_record = type(
                "obj",
                (object,),
                {
                    "uri": self.initial_policy_uri,
                    "metadata": {"generation": self.initial_generation},
                },
            )()

        saved_record = save_policy_with_metadata(
            policy=self.agent,
            policy_store=self.policy_store,
            epoch=self.epoch,
            agent_step=self.agent_step,
            evals=self.stats_manager.eval_scores,
            timer=self.timer,
            initial_policy_record=temp_initial_policy_record,
            run_name=self.run_name,
            is_master=self.is_master,
        )

        if saved_record:
            self.latest_saved_policy_record = saved_record

            # Clean up old policies periodically
            if self.epoch % 10 == 0:
                cleanup_old_policies(self.checkpoint_dir, keep_last_n=5)

        # Save training state
        extra_args = {}
        if self.kickstarter.enabled and self.kickstarter.teacher_uri is not None:
            extra_args["teacher_pr_uri"] = self.kickstarter.teacher_uri

        latest_uri = self.latest_saved_policy_record.uri if self.latest_saved_policy_record else None
        checkpoint = TrainerCheckpoint(
            agent_step=self.agent_step,
            epoch=self.epoch,
            optimizer_state_dict=self.optimizer.state_dict(),
            stopwatch_state=self.timer.save_state(),
            policy_path=latest_uri,
            extra_args=extra_args,
        )
        checkpoint.save(self.run_dir)
        logger.info(f"Saved training state at epoch {self.epoch}")

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(f"Starting training on {self.device}")
        training_start_time = time.time()

        while self.agent_step < self.trainer_config.total_timesteps:
            # Run one epoch
            steps_per_sec, _ = self.train_epoch()

            # Periodic operations
            if should_run(self.epoch, 10, self.is_master):
                record_heartbeat()

            # Update L2 weights if configured
            if hasattr(self.agent, "l2_init_weight_update_interval"):
                maybe_update_l2_weights(
                    agent=self.agent,
                    epoch=self.epoch,
                    interval=getattr(self.agent, "l2_init_weight_update_interval", 0),
                    is_master=self.is_master,
                )

            # Compute gradient statistics
            self.stats_manager.compute_gradient_stats(self.agent, self.epoch)

            # Save checkpoint
            if should_run(self.epoch, self.trainer_config.checkpoint.checkpoint_interval, True):
                self.checkpoint()

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

            # Upload to wandb
            if (
                self.is_master
                and self.wandb_run
                and self.latest_saved_policy_record
                and should_run(self.epoch, self.trainer_config.checkpoint.wandb_checkpoint_interval, True)
            ):
                try:
                    self.wandb_policy_name = self.policy_store.add_to_wandb_run(
                        self.wandb_run.id, self.latest_saved_policy_record
                    )
                    logger.info(f"Uploaded policy to wandb at epoch {self.epoch}")
                except Exception as e:
                    logger.warning(f"Failed to upload policy to wandb: {e}")

            # Check for abort
            if self.is_master and self.wandb_run and should_run(self.epoch, 5, True):
                if check_abort(self.wandb_run, self.trainer_config, self.agent_step):
                    break

            # Evaluation
            if self.evaluation_manager.should_evaluate(self.epoch) and self.latest_saved_policy_record:
                eval_scores = self.evaluation_manager.evaluate_policy(
                    self.latest_saved_policy_record,
                    self.epoch,
                    self.env_manager.get_curriculum(),
                    self.wandb_run,
                    self.wandb_policy_name,
                    self.agent_step,
                )
                self.stats_manager.update_eval_scores(eval_scores)

                # Generate replay
                self.evaluation_manager.generate_replay(
                    self.latest_saved_policy_record,
                    self.epoch,
                    self.env_manager.get_curriculum(),
                    self.wandb_run,
                )

        # Training complete
        total_elapsed = time.time() - training_start_time
        logger.info("Training complete!")
        logger.info(f"Total training time: {total_elapsed:.1f}s")
        logger.info(f"Final epoch: {self.epoch}, Total steps: {self.agent_step}")

        # Log timing summary
        timing_summary = self.timer.get_all_summaries()
        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        # Final evaluation if needed
        if self.evaluation_manager.final_evaluation_needed(self.epoch) and self.latest_saved_policy_record:
            eval_scores = self.evaluation_manager.evaluate_policy(
                self.latest_saved_policy_record,
                self.epoch,
                self.env_manager.get_curriculum(),
                self.wandb_run,
                self.wandb_policy_name,
                self.agent_step,
            )
            self.stats_manager.update_eval_scores(eval_scores)

        # Save final checkpoint
        self.checkpoint()

        # Force upload final policy to wandb
        if self.wandb_run and self.latest_saved_policy_record and self.is_master:
            try:
                self.policy_store.add_to_wandb_run(self.wandb_run.id, self.latest_saved_policy_record, force=True)
                logger.info("Uploaded final policy to wandb")
            except Exception as e:
                logger.warning(f"Failed to upload final policy to wandb: {e}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def cleanup(self) -> None:
        """Clean up resources after training."""
        # Stop monitoring
        if self.is_master:
            if self.system_monitor:
                self.system_monitor.stop()
            if self.memory_monitor:
                self.memory_monitor.clear()

        # Close environment
        self.env_manager.close()

        logger.info(f"\nTraining run complete! Run saved to: {self.run_dir}")

        # Clean up distributed training
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Clean up wandb
        if self.is_master and self.wandb_ctx:
            self.wandb_ctx.__exit__(None, None, None)
