import contextlib
import importlib
from typing import Any, Callable, Optional

import torch

from metta.agent.policy import Policy
from metta.agent.util.distribution_utils import configure_sampling_backend
from metta.common.util.log_config import getRankAwareLogger
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import (
    ComponentContext,
    ContextCheckpointer,
    CoreTrainingLoop,
    DistributedHelper,
    Experience,
    TrainerCallback,
    TrainerComponent,
    TrainerState,
    TrainingEnvironment,
)
from metta.rl.training.optimizer import create_optimizer
from mettagrid.profiling.stopwatch import Stopwatch

try:
    importlib.import_module("pufferlib._C")
except ImportError as exc:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. "
        "If you have non-default PyTorch, try installing with --no-build-isolation"
    ) from exc

torch.set_float32_matmul_precision("high")
logger = getRankAwareLogger(__name__)


class Trainer:
    """Main trainer facade that coordinates all training components."""

    def __init__(
        self,
        cfg: TrainerConfig,
        env: TrainingEnvironment,
        policy: Policy,
        device: torch.device,
        distributed_helper: Optional[DistributedHelper] = None,
        run_name: Optional[str] = None,
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
        if distributed_helper is None:
            distributed_helper = DistributedHelper(self._device)
        self._distributed_helper = distributed_helper
        self._run_name = run_name
        self._components: list[TrainerComponent] = []
        self.timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self.timer.start()

        self._policy.to(self._device)
        self._policy.initialize_to_environment(self._env.meta_data, self._device)

        compile_mode = self._cfg.compile
        use_compile = compile_mode != "disabled"
        effective_mode = compile_mode if use_compile else "reduce-overhead"

        configure_sampling_backend(use_compile, effective_mode)

        if use_compile:
            self._policy = torch.compile(self._policy, mode=effective_mode)

        self._policy.train()

        self._policy = self._distributed_helper.wrap_policy(self._policy, self._device)
        self._policy.to(self._device)
        losses = self._cfg.losses.init_losses(self._policy, self._cfg, self._env, self._device)
        self._policy.train()

        batch_info = self._env.batch_info

        parallel_agents = getattr(self._env, "total_parallel_agents", None)
        if parallel_agents is None:
            parallel_agents = batch_info.num_envs * self._env.meta_data.num_agents

        storage_device = torch.device("cpu") if self._cfg.cpu_offload else self._device
        pin_memory = self._cfg.cpu_offload and self._device.type == "cuda"

        self._experience = Experience.from_losses(
            total_agents=parallel_agents,
            batch_size=self._cfg.batch_size,
            bptt_horizon=self._cfg.bptt_horizon,
            minibatch_size=self._cfg.minibatch_size,
            max_minibatch_size=self._cfg.minibatch_size,
            policy_experience_spec=self._policy.get_agent_experience_spec(),
            losses=losses,
            device=self._device,
            storage_device=storage_device,
            pin_memory=pin_memory,
        )

        self.optimizer = create_optimizer(self._cfg.optimizer, self._policy)

        self._state = TrainerState()

        # Extract curriculum from environment if available
        curriculum = getattr(self._env, "_curriculum", None)

        self._context = ComponentContext(
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

        amp_dtype_name = self._cfg.precision
        amp_dtype = getattr(torch, amp_dtype_name, torch.float32)
        amp_supported = amp_dtype in (torch.float16, torch.bfloat16)
        amp_requested = self._cfg.amp and self._device.type == "cuda"
        amp_enabled = amp_requested and amp_supported

        if amp_requested and not amp_supported:
            logger.warning(
                "Disabling AMP: precision '%s' is not supported for CUDA autocast (use 'float16' or 'bfloat16').",
                amp_dtype_name,
            )

        if amp_enabled:

            def make_autocast() -> contextlib.AbstractContextManager:
                return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

        else:

            def make_autocast() -> contextlib.AbstractContextManager:
                return contextlib.nullcontext()

        self._context.set_autocast_factory(make_autocast)
        self._context.amp_enabled = amp_enabled
        self._context.amp_dtype = amp_dtype if amp_enabled else torch.float32

        self._train_epoch_callable: Callable[[], None] = self._run_epoch

        self.core_loop = CoreTrainingLoop(
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

        self._lr_scheduler = None
        if self._cfg.lr_scheduler == "cosine":
            total_epochs = max(1, self._cfg.total_timesteps // self._cfg.batch_size)
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs)
        self._context.lr_scheduler = self._lr_scheduler

        self._prev_agent_step_for_step_callbacks: int = 0

    @property
    def context(self) -> ComponentContext:
        """Return the shared trainer context."""

        return self._context

    def train(self) -> None:
        """Run the main training loop."""

        try:
            while self._state.agent_step < self._cfg.total_timesteps:
                self._train_epoch_callable()

        except Exception:
            self._invoke_callback(TrainerCallback.FAILURE)
            raise

        self._distributed_helper.synchronize()
        self._invoke_callback(TrainerCallback.TRAINING_COMPLETE)

    def _set_train_epoch_callable(self, fn: Callable[[], None]) -> None:
        self._train_epoch_callable = fn

    def _run_epoch(self) -> None:
        """Run a single training epoch."""
        self._context.reset_for_epoch()

        # Start new epoch
        self.core_loop.on_epoch_start(self._context)

        # Rollout phase
        with self.timer("_rollout"):
            rollout_result = self.core_loop.rollout_phase(self._env, self._context)
            self._context.training_env_id = rollout_result.training_env_id
            world_size = self._distributed_helper.get_world_size()
            previous_agent_step = self._context.agent_step
            if rollout_result.agent_steps:
                self._context.record_rollout(rollout_result.agent_steps, world_size)
            if rollout_result.raw_infos:
                self._prev_agent_step_for_step_callbacks = previous_agent_step
                self._invoke_callback(TrainerCallback.STEP, rollout_result.raw_infos)

        # Training phase
        with self.timer("_train"):
            if self._context.training_env_id is None:
                raise RuntimeError("Training environment slice unavailable for training phase")
            losses_stats, epochs_trained = self.core_loop.training_phase(
                context=self._context,
                update_epochs=self._cfg.update_epochs,
                max_grad_norm=0.5,
            )
            self._context.advance_epoch(epochs_trained)
            if self._lr_scheduler is not None and epochs_trained > 0:
                # Advance the scheduler once per outer training epoch to align with the configured horizon.
                self._lr_scheduler.step()

        # Synchronize before proceeding
        self._distributed_helper.synchronize()

        # Store losses stats for callbacks
        self._context.latest_losses_stats = losses_stats

        # Invoke callbacks for epoch end on every rank. Components that should
        # only run on the master process must set `_master_only` so they aren't
        # registered on other ranks.
        self._invoke_callback(TrainerCallback.EPOCH_END)

        # Progress logging handled by ProgressLogger component

    @staticmethod
    def load_or_create(
        checkpoint_path: str,
        cfg: TrainerConfig,
        training_env: TrainingEnvironment,
        policy: Policy,
        device: torch.device,
        distributed_helper: Optional[DistributedHelper] = None,
        run_name: Optional[str] = None,
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

    def register(self, component: TrainerComponent) -> None:
        """Register a training component.

        Args:
            component: Training component to register
        """
        if component._master_only and not self._distributed_helper.is_master():
            return

        self._components.append(component)
        component.register(self._context)

    def _invoke_callback(self, callback_type: TrainerCallback, infos: Optional[list[dict[str, Any]]] = None) -> None:
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
                if callback_type == TrainerCallback.STEP:
                    if (
                        component.should_handle_step(current_step=current_step, previous_step=previous_step)
                        and infos is not None
                    ):
                        component.on_step(infos)
                elif callback_type == TrainerCallback.EPOCH_END:
                    if component.should_handle_epoch(current_epoch):
                        component.on_epoch_end(current_epoch)
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
        for component in self._components:
            if isinstance(component, ContextCheckpointer):
                component.restore(self._context)
                break
            # Wandb setup will be handled by callbacks if configured
