import importlib
from typing import Any, Callable, Optional

import torch
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.common.util.log_config import getRankAwareLogger
from metta.rl.binding_config import LossProfileConfig, PolicyBindingConfig
from metta.rl.binding_controller import BindingControllerPolicy
from metta.rl.policy_registry import PolicyRegistry
from metta.rl.system_config import SystemConfig
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
from metta.rl.training.optimizer import create_optimizer, is_schedulefree_optimizer
from mettagrid.profiling.stopwatch import Stopwatch

try:
    importlib.import_module("pufferlib._C")
except ImportError:
    raise ImportError("Failed to import C/CUDA kernel. Try: pip install --no-build-isolation") from None

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
        if self._cfg.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Torch autograd anomaly detection enabled; backward will be slower.")
        if distributed_helper is None:
            distributed_helper = DistributedHelper(SystemConfig(device=self._device.type))
        self._distributed_helper = distributed_helper
        self._run_name = run_name
        self._components: list[TrainerComponent] = []
        self.timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self.timer.start()
        self._policy_registry = PolicyRegistry()

        self._policy.to(self._device)
        self._policy.initialize_to_environment(self._env.policy_env_info, self._device)
        self._policy.train()

        binding_state = self._build_binding_state(self._policy)
        if len(binding_state["bindings"]) > 1 or not binding_state["bindings"][0].use_trainer_policy:
            self._policy = BindingControllerPolicy(
                binding_lookup=binding_state["binding_lookup"],
                bindings=binding_state["bindings"],
                binding_policies=binding_state["binding_policies"],
                policy_env_info=self._env.policy_env_info,
                device=self._device,
            )

        self._policy = self._distributed_helper.wrap_policy(self._policy, self._device)
        self._policy.to(self._device)
        losses = self._cfg.losses.init_losses(self._policy, self._cfg, self._env, self._device)
        self._policy.train()

        batch_info = self._env.batch_info

        parallel_agents = getattr(self._env, "total_parallel_agents", None)
        if parallel_agents is None:
            parallel_agents = batch_info.num_envs * self._env.policy_env_info.num_agents

        policy_experience_spec = self._extend_policy_experience_spec(
            self._policy.get_agent_experience_spec(),
            has_multiple_bindings=len(binding_state["bindings"]) > 1,
        )

        self._experience = Experience.from_losses(
            total_agents=parallel_agents,
            batch_size=self._cfg.batch_size,
            bptt_horizon=self._cfg.bptt_horizon,
            minibatch_size=self._cfg.minibatch_size,
            max_minibatch_size=self._cfg.minibatch_size,
            policy_experience_spec=policy_experience_spec,
            losses=losses,
            device=self._device,
        )

        self.optimizer = create_optimizer(self._cfg.optimizer, self._policy)
        self._is_schedulefree = is_schedulefree_optimizer(self.optimizer)

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
        self._context.binding_id_per_agent = binding_state["binding_ids"]
        self._context.loss_profile_id_per_agent = binding_state["loss_profile_ids"]
        self._context.trainable_agent_mask = binding_state["trainable_mask"]
        self._context.binding_id_lookup = binding_state["binding_lookup"]
        self._context.loss_profile_lookup = binding_state["loss_profile_lookup"]
        self._context.policy_bindings = binding_state["bindings"]

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
                self._invoke_callback(TrainerCallback.STEP, rollout_result.raw_infos)
            self._invoke_callback(TrainerCallback.ROLLOUT_END)

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

    # ------------------------------------------------------------------
    # Binding setup helpers
    # ------------------------------------------------------------------
    def _build_binding_state(self, trainer_policy: Policy) -> dict[str, Any]:
        """Prepare per-agent binding and loss-profile metadata."""

        bindings_cfg = list(self._cfg.policy_bindings or [])
        # Ensure at least one binding uses the trainer-provided policy
        has_trainer_binding = any(b.use_trainer_policy for b in bindings_cfg)
        if not bindings_cfg or not has_trainer_binding:
            bindings_cfg.insert(0, PolicyBindingConfig(id="main", use_trainer_policy=True, trainable=True))

        trainer_binding_count = sum(1 for b in bindings_cfg if b.use_trainer_policy)
        if trainer_binding_count > 1:
            raise ValueError("Only one binding may set use_trainer_policy=True")

        binding_lookup: dict[str, int] = {}
        binding_policies: dict[int, Policy] = {}
        for binding in bindings_cfg:
            if binding.id in binding_lookup:
                raise ValueError(f"Duplicate policy binding id '{binding.id}'")
            binding_lookup[binding.id] = len(binding_lookup)
            if binding.use_trainer_policy:
                binding_policies[binding_lookup[binding.id]] = trainer_policy
            else:
                binding_policies[binding_lookup[binding.id]] = self._policy_registry.get(
                    binding,
                    self._env.policy_env_info,
                    self._device if binding.device is None else torch.device(binding.device),
                )

        # Loss profiles (config-only in Phase 0)
        loss_profiles = dict(self._cfg.loss_profiles)
        if not loss_profiles:
            loss_profiles = {"default": LossProfileConfig(losses=[])}
        loss_profile_lookup = {name: idx for idx, name in enumerate(loss_profiles.keys())}

        default_binding_profile = next(iter(loss_profiles.keys()))

        num_agents = self._env.policy_env_info.num_agents
        agent_binding_map = self._cfg.agent_binding_map
        if agent_binding_map is None:
            agent_binding_map = [bindings_cfg[0].id for _ in range(num_agents)]
        if len(agent_binding_map) != num_agents:
            raise ValueError(
                f"agent_binding_map must have length num_agents ({num_agents}); got {len(agent_binding_map)}"
            )

        binding_ids = []
        loss_profile_ids = []
        trainable_mask = []
        for idx, binding_id_str in enumerate(agent_binding_map):
            if binding_id_str not in binding_lookup:
                raise ValueError(f"agent_binding_map[{idx}] references unknown binding id '{binding_id_str}'")
            b_idx = binding_lookup[binding_id_str]
            binding = bindings_cfg[b_idx]
            binding_ids.append(b_idx)

            profile_name = binding.loss_profile or default_binding_profile
            if profile_name not in loss_profile_lookup:
                # Auto-register profile if referenced but not defined
                loss_profile_lookup[profile_name] = len(loss_profile_lookup)
            loss_profile_ids.append(loss_profile_lookup[profile_name])
            trainable_mask.append(bool(binding.trainable))

        binding_tensor = torch.tensor(binding_ids, dtype=torch.long)
        loss_profile_tensor = torch.tensor(loss_profile_ids, dtype=torch.long)
        trainable_tensor = torch.tensor(trainable_mask, dtype=torch.bool)

        if len(bindings_cfg) > 1:
            logger.warning(
                "Multiple policy bindings configured; Phase 0 annotates rollout with binding metadata but control "
                "is still provided by the primary trainer policy."
            )

        return {
            "bindings": bindings_cfg,
            "binding_lookup": binding_lookup,
            "loss_profile_lookup": loss_profile_lookup,
            "binding_ids": binding_tensor,
            "loss_profile_ids": loss_profile_tensor,
            "trainable_mask": trainable_tensor,
            "binding_policies": binding_policies,
        }

    def _extend_policy_experience_spec(self, base_spec: Composite, has_multiple_bindings: bool) -> Composite:
        """Append binding/loss-profile metadata to the policy experience spec when needed."""

        if not has_multiple_bindings:
            return base_spec

        extras = {
            "binding_id": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            "loss_profile_id": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            "is_trainable_agent": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool),
        }
        merged = dict(base_spec.items())
        merged.update(extras)
        return Composite(merged)

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
                elif callback_type == TrainerCallback.ROLLOUT_END:
                    component.on_rollout_end()
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
