"""Main trainer facade for coordinating all training components."""

import logging
from typing import Any, Dict, Optional, TypeVar

import torch

from metta.agent.policy import Policy
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.component import TrainerCallback, TrainerComponent
from metta.rl.training.context import TrainerContext
from metta.rl.training.core import CoreTrainingLoop
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.experience import Experience
from metta.rl.training.heartbeat import HeartbeatWriter
from metta.rl.training.optimizer import create_optimizer
from metta.rl.training.training_environment import TrainingEnvironment
from metta.rl.utils import log_training_progress

T_Component = TypeVar("T_Component", bound=TrainerComponent)

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
        cfg: TrainerConfig,
        env: TrainingEnvironment,
        policy: Policy,
        device: torch.device,
    ):
        """Initialize trainer with all components.

        Args:
            cfg: Trainer configuration
            env: TrainingEnvironment instance for experience generation
            policy: The policy/agent to train
        """
        self._env = env
        self._policy = policy
        self._cfg = cfg
        self._device = device
        self._distributed_helper = DistributedHelper(self._device)
        self._components: list[TrainerComponent] = []
        self._component_map: dict[type[TrainerComponent], TrainerComponent] = {}

        self._epoch = 0
        self._agent_step = 0
        self.timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self.timer.start()

        self._policy = self._distributed_helper.wrap_policy(self._policy, self._device)
        self._policy.to(self._device)
        self._policy.initialize_to_environment(self._env.meta_data, self._device)
        losses = self._cfg.losses.init_losses(self._policy, self._cfg, self._env, self._device)
        self._policy.to(self._device)

        # Put the torch policy into training mode
        self._policy.train()

        batch_info = self._env.batch_info
        parallel_agents = getattr(self._env, "total_parallel_agents", None)
        if parallel_agents is None:
            parallel_agents = batch_info.num_envs * self._env.meta_data.num_agents

        experience = Experience.from_losses(
            total_agents=parallel_agents,
            batch_size=self._cfg.batch_size,
            bptt_horizon=self._cfg.bptt_horizon,
            minibatch_size=self._cfg.minibatch_size,
            max_minibatch_size=self._cfg.minibatch_size,
            policy_experience_spec=self._policy.get_agent_experience_spec(),
            losses=losses,
            device=self._device,
        )

        self.optimizer = create_optimizer(self._cfg.optimizer, self._policy)

        self.core_loop = CoreTrainingLoop(
            policy=self._policy,
            experience=experience,
            losses=losses,
            optimizer=self.optimizer,
            device=self._device,
            accumulate_minibatches=experience.accumulate_minibatches,
        )

        self._context = TrainerContext(
            trainer=self,
            policy=self._policy,
            env=self._env,
            experience=experience,
            optimizer=self.optimizer,
            config=self._cfg,
            device=self._device,
            stopwatch=self.timer,
            distributed=self._distributed_helper,
            run_dir=None,
            run_name=None,
            get_epoch=lambda: self._epoch,
            set_epoch=lambda value: setattr(self, "_epoch", value),
            get_agent_step=lambda: self._agent_step,
            set_agent_step=lambda value: setattr(self, "_agent_step", value),
        )

        if self._cfg.heartbeat is not None:
            self.register(HeartbeatWriter(epoch_interval=self._cfg.heartbeat.epoch_interval))

    @property
    def context(self) -> TrainerContext:
        """Return the shared trainer context."""

        return self._context

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = value

    @property
    def agent_step(self) -> int:
        return self._agent_step

    @agent_step.setter
    def agent_step(self, value: int) -> None:
        self._agent_step = value

    def train(self) -> None:
        """Run the main training loop."""

        try:
            while self._agent_step < self._cfg.total_timesteps:
                self._train_epoch()

        except Exception:
            self._invoke_callback(TrainerCallback.FAILURE)
            raise

        self._distributed_helper.synchronize()
        self._invoke_callback(TrainerCallback.TRAINING_COMPLETE)

    def _train_epoch(self) -> None:
        """Run a single training epoch."""
        if not self.core_loop:
            raise RuntimeError("Core loop not initialized")

        steps_before = self._agent_step

        # Start new epoch
        self.core_loop.on_epoch_start(self._epoch)

        # Rollout phase
        with self.timer("_rollout"):
            rollout_result = self.core_loop.rollout_phase(self._env, self._epoch)
            self._agent_step += rollout_result.agent_steps * self._distributed_helper.get_world_size()
            # Invoke step callbacks for each info
            for info in rollout_result.raw_infos:
                self._invoke_callback(TrainerCallback.STEP, info)

        # Training phase
        with self.timer("_train"):
            losses_stats = self.core_loop.training_phase(
                epoch=self._epoch,
                training_env_id=slice(0, self._cfg.batch_size),
                update_epochs=self._cfg.update_epochs,
                max_grad_norm=0.5,
            )
            self._epoch += self._cfg.update_epochs

        # Synchronize before proceeding
        self._distributed_helper.synchronize()

        # Store losses stats for callbacks
        self.latest_losses_stats = losses_stats

        # Master-only operations
        if not self._distributed_helper.is_master():
            return

        # Invoke callbacks for epoch end
        self._invoke_callback(TrainerCallback.EPOCH_END)

        # Log progress
        log_training_progress(
            epoch=self._epoch,
            agent_step=self._agent_step,
            prev_agent_step=steps_before,
            total_timesteps=self._cfg.total_timesteps,
            train_time=self.timer.get_last_elapsed("_train"),
            rollout_time=self.timer.get_last_elapsed("_rollout"),
            stats_time=self.timer.get_last_elapsed("_process_stats"),
        )

    @staticmethod
    def load_or_create(
        checkpoint_path: str,
        cfg: TrainerConfig,
        training_env: TrainingEnvironment,
        policy: Policy,
        device: torch.device,
    ) -> "Trainer":
        """Create a trainer from a configuration."""
        return Trainer(cfg, training_env, policy, device)

    def register(self, component: TrainerComponent) -> None:
        """Register a training component.

        Args:
            component: Training component to register
        """
        if component._master_only and not self._distributed_helper.is_master():
            return

        self._components.append(component)
        self._component_map[type(component)] = component
        self._context.register_component(component)
        component.register(self._context)

    def _invoke_callback(self, callback_type: TrainerCallback, infos: Optional[Dict[str, Any]] = None) -> None:
        """Invoke all registered callbacks of the specified type.

        Args:
            callback_type: The type of callback to invoke
            infos: Step information from environment (only used for STEP callback)
        """
        for component in self._components:
            try:
                if callback_type == TrainerCallback.STEP:
                    if component._step_interval != 0 and self._agent_step % component._step_interval == 0:
                        component.on_step(infos)
                elif callback_type == TrainerCallback.EPOCH_END:
                    if component._epoch_interval != 0 and self._epoch % component._epoch_interval == 0:
                        component.on_epoch_end(self._epoch)
                    elif component._epoch_interval == 0:
                        component.on_epoch_end(self._epoch)
                elif callback_type == TrainerCallback.TRAINING_COMPLETE:
                    component.on_training_complete()
                elif callback_type == TrainerCallback.FAILURE:
                    component.on_failure()
            except Exception as e:
                logger.error(
                    f"Component {component.__class__.__name__} {callback_type.value} callback failed: {e}",
                    exc_info=True,
                )

    def restore(self) -> None:
        """Restore trainer state from checkpoints.

        This should be called after setup() to restore any saved state.
        """
        # Find and restore trainer checkpointer state
        from metta.rl.training.trainer_checkpointer import TrainerCheckpointer

        for component in self._components:
            if isinstance(component, TrainerCheckpointer):
                component.restore(self._context)
                break
            # Wandb setup will be handled by callbacks if configured
