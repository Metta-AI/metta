"""Main trainer facade for coordinating all training components."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed
from heavyball import ForeachMuon
from torchrl.data import Composite

from metta.core.monitoring import cleanup_monitoring, setup_monitoring
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.rl.experience import Experience
from metta.rl.losses import get_loss_experience_spec
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState
from metta.rl.training.core import CoreTrainingLoop
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.evaluator import Evaluator
from metta.rl.utils import log_training_progress

if TYPE_CHECKING:
    from metta.rl.training.training_environment import TrainingEnvironment

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib  # type: ignore[reportUnusedImport]
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer facade that coordinates all training components."""

    def __init__(
        self,
        run_dir: str,
        run_name: str,
        training_env: "TrainingEnvironment",  # Required TrainingEnvironment instance
        policy: Any,  # The policy/agent to train
        system_cfg: SystemConfig,
        trainer_cfg: TrainerConfig,
        device: torch.device,
        distributed_helper: DistributedHelper,
        components: List,
    ):
        """Initialize trainer with all components.

        Args:
            run_dir: Directory for this run
            run_name: Name of the run
            training_env: TrainingEnvironment instance for experience generation
            policy: The policy/agent to train
            system_cfg: System configuration
            trainer_cfg: Trainer configuration
            device: Device to train on
            distributed_helper: Helper for distributed training
            components: List of training components (callbacks)
        """
        self.run_dir = run_dir
        self.run_name = run_name
        self.training_env = training_env
        self.policy = policy
        self.system_cfg = system_cfg
        self.trainer_cfg = trainer_cfg
        self.device = device

        # Use provided distributed helper
        self.distributed_helper = distributed_helper
        self.distributed_helper.setup()

        # Initialize components list
        self._components = components

        # Find evaluator from components (still needed for some internal logic)
        self.evaluator = None
        self.stats_client = None

        # Find components from the list
        for component in self._components:
            if isinstance(component, Evaluator):
                self.evaluator = component

        # Find stats client from stats reporter if available
        from metta.rl.training.stats_reporter import StatsReporter

        for component in self._components:
            if isinstance(component, StatsReporter) and hasattr(component, "_stats_client"):
                self.stats_client = component._stats_client
                break

        # Create timer
        self.timer = Stopwatch(log_level=logger.getEffectiveLevel())

        # Initialize other components
        self.vecenv = None
        self.core_loop = None
        self.optimizer = None
        self.memory_monitor = None
        self.system_monitor = None
        self.latest_losses_stats = {}
        self.latest_grad_stats = {}

    def register_component(self, component) -> None:
        """Register a training component.

        Args:
            component: Training component to register
        """
        from metta.rl.training.component import TrainingComponent

        if isinstance(component, TrainingComponent):
            self._components.append(component)
        else:
            logger.warning(f"Component {component} is not a TrainingComponent, skipping registration")

    def get_latest_policy_uri(self) -> Optional[str]:
        """Get the latest policy checkpoint URI.

        Returns:
            URI of the latest policy checkpoint, or None if no checkpoints exist
        """
        # Find policy checkpointer from components
        from metta.rl.training.policy_checkpointer import PolicyCheckpointer

        for component in self._components:
            if isinstance(component, PolicyCheckpointer):
                return component.get_latest_policy_uri()
        return None

    def _invoke_step_callbacks(self, infos: Dict[str, Any]) -> None:
        """Invoke all registered callbacks for step.

        Args:
            infos: Step information from environment
        """
        for component in self._components:
            try:
                component.on_step(self, infos)
            except Exception as e:
                logger.error(f"Component {component.__class__.__name__} on_step failed: {e}", exc_info=True)

    def _invoke_epoch_callbacks(self) -> None:
        """Invoke all registered callbacks for epoch end."""
        epoch = self.trainer_state.epoch
        for component in self._components:
            if epoch % component.interval == 0:
                try:
                    component.on_epoch_end(self, epoch)
                except Exception as e:
                    logger.error(f"Component {component.__class__.__name__} on_epoch_end failed: {e}", exc_info=True)

    def _invoke_training_complete_callbacks(self) -> None:
        """Invoke all registered callbacks for training completion."""
        for component in self._components:
            try:
                component.on_training_complete(self)
            except Exception as e:
                logger.error(
                    f"Component {component.__class__.__name__} on_training_complete failed: {e}",
                    exc_info=True,
                )

    def _invoke_failure_callbacks(self) -> None:
        """Invoke all registered callbacks for training failure."""
        for component in self._components:
            try:
                component.on_failure(self)
            except Exception as e:
                logger.error(f"Component {component.__class__.__name__} on_failure failed: {e}", exc_info=True)

    def setup(self) -> None:
        """Setup training environment."""
        self.timer.start()

        # Set up the training environment
        from metta.rl.training.training_environment import TrainingEnvironment

        if not isinstance(self.training_env, TrainingEnvironment):
            raise TypeError(f"training_env must be a TrainingEnvironment instance, got {type(self.training_env)}")

        metta_grid_env, target_batch_size, batch_size, num_envs = self.training_env.setup()
        self.vecenv = self.training_env.get_vecenv()

        # Initialize trainer state (will be restored by TrainerCheckpointer if needed)
        self.trainer_state = TrainerState(
            agent_step=0,
            epoch=0,
            update_epoch=0,
            mb_idx=0,
            optimizer=None,  # Will be set after creating optimizer
        )
        self.latest_saved_epoch = 0

        # Compile policy if requested
        if self.trainer_cfg.compile:
            if self.distributed_helper.is_master():
                logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)  # type: ignore

        # Wrap policy for distributed training
        self.policy = self.distributed_helper.wrap_policy(self.policy, self.device)

        # Initialize policy to environment
        self.policy.train()
        features = metta_grid_env.get_observation_features()
        self.policy.initialize_to_environment(
            features, metta_grid_env.action_names, metta_grid_env.max_action_args, self.device
        )

        # Create losses
        losses = self.trainer_cfg.losses.init_losses(self.policy, self.trainer_cfg, self.vecenv, self.device)

        # Create experience buffer
        experience = self._create_experience_buffer(metta_grid_env, losses)

        # Attach experience buffer to losses
        for loss_instance in losses.values():
            loss_instance.attach_replay_buffer(experience)

        # Create optimizer
        self.optimizer = self._create_optimizer(self.trainer_state.__dict__)
        self.trainer_state.optimizer = self.optimizer

        # Create core training loop
        self.core_loop = CoreTrainingLoop(
            policy=self.policy,
            experience=experience,
            losses=losses,
            optimizer=self.optimizer,
            device=self.device,
            accumulate_minibatches=experience.accumulate_minibatches,
        )

        # Setup monitoring (master only)
        if self.distributed_helper.is_master():
            logger.info("Starting training")
            self.memory_monitor, self.system_monitor = setup_monitoring(
                policy=self.policy,
                experience=experience,
                timer=self.timer,
            )

            # Wandb setup will be handled by callbacks if configured

    def _create_experience_buffer(self, env, losses):
        """Create experience buffer with merged specs from policy and losses."""
        # Get specs from policy and losses
        policy_spec = self.policy.get_agent_experience_spec()
        act_space = self.vecenv.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        loss_spec = get_loss_experience_spec(act_space.nvec, act_dtype)

        # Merge all specs
        merged_spec_dict: dict = dict(policy_spec.items())
        for inst in losses.values():
            spec = inst.get_experience_spec()
            merged_spec_dict.update(dict(spec.items()))
        merged_spec_dict.update(dict(loss_spec.items()))

        # Create experience buffer
        return Experience(
            total_agents=self.vecenv.num_agents,
            batch_size=self.trainer_cfg.batch_size,
            bptt_horizon=self.trainer_cfg.bptt_horizon,
            minibatch_size=self.trainer_cfg.minibatch_size,
            max_minibatch_size=self.trainer_cfg.minibatch_size,
            experience_spec=Composite(merged_spec_dict),
            device=self.device,
        )

    def _create_optimizer(self, trainer_state: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer and load state if available."""
        optimizer_type = self.trainer_cfg.optimizer.type

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.trainer_cfg.optimizer.learning_rate,
                betas=(self.trainer_cfg.optimizer.beta1, self.trainer_cfg.optimizer.beta2),
                eps=self.trainer_cfg.optimizer.eps,
                weight_decay=self.trainer_cfg.optimizer.weight_decay,
            )
        elif optimizer_type == "muon":
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=self.trainer_cfg.optimizer.learning_rate,
                betas=(self.trainer_cfg.optimizer.beta1, self.trainer_cfg.optimizer.beta2),
                eps=self.trainer_cfg.optimizer.eps,
                weight_decay=int(self.trainer_cfg.optimizer.weight_decay),
            )
        else:
            raise ValueError(f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}")

        # Load optimizer state if available
        if trainer_state and "optimizer_state" in trainer_state:
            try:
                optimizer.load_state_dict(trainer_state["optimizer_state"])
                logger.info("Successfully loaded optimizer state from checkpoint")
            except ValueError:
                logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

        return optimizer

    def train(self) -> None:
        """Run the main training loop."""
        if not self.core_loop:
            raise RuntimeError("Trainer must be setup before training")

        try:
            while self.trainer_state.agent_step < self.trainer_cfg.total_timesteps:
                self._train_epoch()

        except Exception:
            self._invoke_failure_callbacks()
            raise
        finally:
            self._cleanup()

        # Finalize training
        self._finalize()

    def _train_epoch(self) -> None:
        """Run a single training epoch."""
        steps_before = self.trainer_state.agent_step

        # Start new epoch
        self.core_loop.on_epoch_start(self.trainer_state)

        # Rollout phase
        with self.timer("_rollout"):
            rollout_result = self.core_loop.rollout_phase(self.vecenv, self.trainer_state, self.timer)
            self.trainer_state.agent_step += rollout_result.agent_steps * self.distributed_helper.get_world_size()
            # Invoke step callbacks for each info
            for info in rollout_result.raw_infos:
                self._invoke_step_callbacks(info)

        # Training phase
        with self.timer("_train"):
            losses_stats = self.core_loop.training_phase(
                self.trainer_state,
                self.trainer_cfg.update_epochs,
                max_grad_norm=0.5,
                timer=self.timer,
            )
            self.trainer_state.epoch += self.trainer_cfg.update_epochs

        # Synchronize before proceeding
        self.distributed_helper.synchronize()

        # Store losses stats for callbacks
        self.latest_losses_stats = losses_stats

        # Master-only operations
        if not self.distributed_helper.is_master():
            return

        # Invoke callbacks for epoch end
        self._invoke_epoch_callbacks()

        # Log progress
        log_training_progress(
            epoch=self.trainer_state.epoch,
            agent_step=self.trainer_state.agent_step,
            prev_agent_step=steps_before,
            total_timesteps=self.trainer_cfg.total_timesteps,
            train_time=self.timer.get_last_elapsed("_train"),
            rollout_time=self.timer.get_last_elapsed("_rollout"),
            stats_time=self.timer.get_last_elapsed("_process_stats"),
            run_name=self.run_name,
        )

    def _cleanup(self) -> None:
        """Clean up resources."""
        # Wait for all ranks
        self.distributed_helper.synchronize()

        # Close environment
        if self.vecenv:
            self.vecenv.close()

        # Clean up monitoring
        if self.distributed_helper.is_master():
            cleanup_monitoring(self.memory_monitor, self.system_monitor)

    def _finalize(self) -> None:
        """Finalize training."""
        if not self.distributed_helper.is_master():
            return

        logger.info("Training complete!")

        # Save final checkpoint
        metadata = {
            "agent_step": self.trainer_state.agent_step,
            "total_time": self.timer.get_elapsed(),
            "total_train_time": (
                self.timer.get_all_elapsed().get("_rollout", 0) + self.timer.get_all_elapsed().get("_train", 0)
            ),
            "is_final": True,
            "upload_to_wandb": False,
        }

        # Add final evaluation scores
        if self.evaluator:
            eval_scores = self.evaluator.get_latest_scores()
            if eval_scores.category_scores or eval_scores.simulation_scores:
                metadata.update(
                    {
                        "score": eval_scores.avg_simulation_score,
                        "avg_reward": eval_scores.avg_category_score,
                    }
                )

        # Final saves are handled by component callbacks (on_training_complete)

        # Log timing summary
        if self.distributed_helper.is_master():
            timing_summary = self.timer.get_all_summaries()
            logger.info("Timing Summary:")
            for name, summary in timing_summary.items():
                logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        # Invoke training complete callbacks
        self._invoke_training_complete_callbacks()
