import importlib
import typing

import torch

import metta.agent.policy
import metta.common.util.log_config
import metta.rl.system_config
import metta.rl.trainer_config
import metta.rl.training.component as training_component
import metta.rl.training.component_context as training_component_context
import metta.rl.training.context_checkpointer as training_context_checkpointer
import metta.rl.training.core as training_core
import metta.rl.training.distributed_helper as training_distributed_helper
import metta.rl.training.experience as training_experience
import metta.rl.training.training_environment as training_environment
import metta.rl.training.optimizer
import mettagrid.profiling.stopwatch

try:
    importlib.import_module("pufferlib._C")
except ImportError:
    raise ImportError("Failed to import C/CUDA kernel. Try: pip install --no-build-isolation") from None

logger = metta.common.util.log_config.getRankAwareLogger(__name__)


class Trainer:
    """Main trainer facade that coordinates all training components."""

    def __init__(
        self,
        cfg: metta.rl.trainer_config.TrainerConfig,
        env: training_environment.TrainingEnvironment,
        policy: metta.agent.policy.Policy,
        device: torch.device,
        distributed_helper: typing.Optional[training_distributed_helper.DistributedHelper] = None,
        run_name: typing.Optional[str] = None,
    ):
        """Initialize trainer with all components.

        Args:
            cfg: Trainer configuration
            env: TrainingEnvironment instance for experience generation
            policy: The policy/agent to train
            distributed_helper: Optional helper managing torch.distributed lifecycle
        """
        self._env = env
        self._policy = policy
        self._cfg = cfg
        self._device = device
        if self._cfg.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Torch autograd anomaly detection enabled; backward will be slower.")
        if distributed_helper is None:
            distributed_helper = training_distributed_helper.DistributedHelper(
                metta.rl.system_config.SystemConfig(device=self._device.type)
            )
        self._distributed_helper = distributed_helper
        self._run_name = run_name
        self._components: list[training_component.TrainerComponent] = []
        self.timer = mettagrid.profiling.stopwatch.Stopwatch(log_level=logger.getEffectiveLevel())
        self.timer.start()

        self._policy.to(self._device)
        self._policy.initialize_to_environment(self._env.policy_env_info, self._device)
        self._policy.train()

        self._policy = self._distributed_helper.wrap_policy(self._policy, self._device)
        self._policy.to(self._device)
        losses = self._cfg.losses.init_losses(self._policy, self._cfg, self._env, self._device)
        self._policy.train()

        batch_info = self._env.batch_info

        parallel_agents = getattr(self._env, "total_parallel_agents", None)
        if parallel_agents is None:
            parallel_agents = batch_info.num_envs * self._env.policy_env_info.num_agents

        self._experience = training_experience.Experience.from_losses(
            total_agents=parallel_agents,
            batch_size=self._cfg.batch_size,
            bptt_horizon=self._cfg.bptt_horizon,
            minibatch_size=self._cfg.minibatch_size,
            max_minibatch_size=self._cfg.minibatch_size,
            policy_experience_spec=self._policy.get_agent_experience_spec(),
            losses=losses,
            device=self._device,
        )

        self.optimizer = metta.rl.training.optimizer.create_optimizer(self._cfg.optimizer, self._policy)
        self._is_schedulefree = metta.rl.training.optimizer.is_schedulefree_optimizer(self.optimizer)

        self._state = training_component_context.TrainerState()

        # Extract curriculum from environment if available
        curriculum = getattr(self._env, "_curriculum", None)

        self._context = training_component_context.ComponentContext(
            state=self._state,
            policy=self._policy,
            env=self._env,
            experience=self._experience,
            optimizer=self.optimizer,
            config=self._cfg,
            stopwatch=self.timer,
            distributed=self._distributed_helper,
            run_name=self._run_name,
            curriculum=curriculum,
        )
        self._context.get_train_epoch_fn = lambda: self._train_epoch_callable
        self._context.set_train_epoch_fn = self._set_train_epoch_callable

        self._train_epoch_callable: typing.Callable[[], None] = self._run_epoch

        self.core_loop = training_core.CoreTrainingLoop(
            policy=self._policy,
            experience=self._experience,
            losses=losses,
            optimizer=self.optimizer,
            device=self._device,
            context=self._context,
        )

        self._losses = losses
        self._context.losses = losses

        for loss in losses.values():
            loss.attach_context(self._context)

        self._prev_agent_step_for_step_callbacks: int = 0

    @property
    def context(self) -> training_component_context.ComponentContext:
        """Return the shared trainer context."""

        return self._context

    def train(self) -> None:
        """Run the main training loop."""

        try:
            while self._state.agent_step < self._cfg.total_timesteps:
                self._train_epoch_callable()

        except Exception:
            self._invoke_callback(training_component.TrainerCallback.FAILURE)
            raise

        self._distributed_helper.synchronize()
        self._invoke_callback(training_component.TrainerCallback.TRAINING_COMPLETE)

    def _set_train_epoch_callable(self, fn: typing.Callable[[], None]) -> None:
        self._train_epoch_callable = fn

    def _run_epoch(self) -> None:
        """Run a single training epoch."""
        self._context.reset_for_epoch()

        # Start new epoch
        self.core_loop.on_epoch_start(self._context)

        # Rollout phase
        with self.timer("_rollout"):
            # Ensure ScheduleFree optimizer is in eval mode during rollout
            if self._is_schedulefree:
                self.optimizer.eval()

            rollout_result = self.core_loop.rollout_phase(self._env, self._context)
            self._context.training_env_id = rollout_result.training_env_id
            world_size = self._distributed_helper.get_world_size()
            previous_agent_step = self._context.agent_step
            if rollout_result.agent_steps:
                self._context.record_rollout(rollout_result.agent_steps, world_size)
            if rollout_result.raw_infos:
                self._prev_agent_step_for_step_callbacks = previous_agent_step
        self._invoke_callback(training_component.TrainerCallback.STEP, rollout_result.raw_infos)

        # Training phase
        with self.timer("_train"):
            if self._context.training_env_id is None:
                raise RuntimeError("Training environment slice unavailable for training phase")

            # ScheduleFree optimizer is in train mode for training phase
            if self._is_schedulefree:
                self.optimizer.train()

            losses_stats, epochs_trained = self.core_loop.training_phase(
                context=self._context,
                update_epochs=self._cfg.update_epochs,
                max_grad_norm=0.5,
            )
            self._context.advance_epoch(epochs_trained)
        # Synchronize before proceeding
        self._distributed_helper.synchronize()

        # Store losses stats for callbacks
        self._context.latest_losses_stats = losses_stats

        # Invoke callbacks for epoch end on every rank. Components that should
        # only run on the master process must set `_master_only` so they aren't
        # registered on other ranks.
        self._invoke_callback(training_component.TrainerCallback.EPOCH_END)

        # Progress logging handled by ProgressLogger component

    @staticmethod
    def load_or_create(
        checkpoint_path: str,
        cfg: metta.rl.trainer_config.TrainerConfig,
        training_env: training_environment.TrainingEnvironment,
        policy: metta.agent.policy.Policy,
        device: torch.device,
        distributed_helper: typing.Optional[training_distributed_helper.DistributedHelper] = None,
        run_name: typing.Optional[str] = None,
    ) -> "Trainer":
        """Create a trainer from a configuration.

        Args:
            distributed_helper: Optional helper to reuse existing process group
        """
        return Trainer(
            cfg,
            training_env,
            policy,
            device,
            distributed_helper=distributed_helper,
            run_name=run_name,
        )

    def register(self, component: training_component.TrainerComponent) -> None:
        """Register a training component.

        Args:
            component: Training component to register
        """
        if component._master_only and not self._distributed_helper.is_master():
            return

        self._components.append(component)
        component.register(self._context)

    def _invoke_callback(
        self,
        callback_type: training_component.TrainerCallback,
        infos: typing.Optional[list[dict[str, typing.Any]]] = None,
    ) -> None:
        """Invoke all registered callbacks of the specified type.

        Args:
            callback_type: The type of callback to invoke
            infos: Step information from environment (only used for STEP callback)
        """
        current_step = self._context.agent_step
        previous_step = getattr(self, "_prev_agent_step_for_step_callbacks", current_step)
        current_epoch = self._context.epoch

        for component in self._components:
            try:
                if callback_type == training_component.TrainerCallback.STEP:
                    if (
                        component.should_handle_step(current_step=current_step, previous_step=previous_step)
                        and infos is not None
                    ):
                        component.on_step(infos)
                elif callback_type == training_component.TrainerCallback.EPOCH_END:
                    if component.should_handle_epoch(current_epoch):
                        component.on_epoch_end(current_epoch)
                elif callback_type == training_component.TrainerCallback.TRAINING_COMPLETE:
                    component.on_training_complete()
                elif callback_type == training_component.TrainerCallback.FAILURE:
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
        for component in self._components:
            if isinstance(component, training_context_checkpointer.ContextCheckpointer):
                component.restore(self._context)
                break
            # Wandb setup will be handled by callbacks if configured
