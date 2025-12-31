import importlib
from typing import Any, Callable, Optional

import torch
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.common.util.log_config import getRankAwareLogger
from metta.rl.slot_config import LossProfileConfig, PolicySlotConfig
from metta.rl.slot_controller import SlotControllerPolicy
from metta.rl.slot_registry import SlotRegistry
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
        self._slot_registry = SlotRegistry()

        self._policy.to(self._device)
        self._policy.initialize_to_environment(self._env.policy_env_info, self._device)
        self._policy.train()

        slot_state = self._build_slot_state(self._policy)
        if len(slot_state["slots"]) > 1 or not slot_state["slots"][0].use_trainer_policy:
            self._policy = SlotControllerPolicy(
                slot_lookup=slot_state["slot_lookup"],
                slots=slot_state["slots"],
                slot_policies=slot_state["slot_policies"],
                policy_env_info=self._env.policy_env_info,
                agent_slot_map=slot_state["slot_ids"],
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
            include_slot_metadata=True,
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
            sampling_config=self._cfg.sampling,
        )

        self.optimizer = create_optimizer(self._cfg.optimizer, self._policy)
        self._is_schedulefree = is_schedulefree_optimizer(self.optimizer)

        self._state = TrainerState()
        reward_centering = self._cfg.advantage.reward_centering
        self._state.avg_reward = torch.full(
            (parallel_agents,),
            float(reward_centering.initial_reward_mean),
            device=self._device,
            dtype=torch.float32,
        )

        # Extract curriculum from environment if available
        curriculum = getattr(self._env, "_curriculum", None)

        self._train_epoch_callable: Callable[[], None] = self._run_epoch

        self._context = ComponentContext(
            state=self._state,
            policy=self._policy,
            env=self._env,
            experience=self._experience,
            optimizer=self.optimizer,
            config=self._cfg,
            stopwatch=self.timer,
            distributed=self._distributed_helper,
            get_train_epoch_fn=lambda: self._train_epoch_callable,
            set_train_epoch_fn=self._set_train_epoch_callable,
            run_name=self._run_name,
            curriculum=curriculum,
        )
        self._context.slot_id_per_agent = slot_state["slot_ids"]
        self._context.loss_profile_id_per_agent = slot_state["loss_profile_ids"]
        self._context.trainable_agent_mask = slot_state["trainable_mask"]
        self._context.slot_id_lookup = slot_state["slot_lookup"]
        self._context.loss_profile_lookup = slot_state["loss_profile_lookup"]
        self._context.policy_slots = slot_state["slots"]
        self._assign_loss_profiles(losses, slot_state["loss_profile_lookup"])

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

    def _assign_loss_profiles(self, losses: dict[str, Any], loss_profile_lookup: dict[str, int]) -> None:
        """Attach resolved loss profile ids to losses based on config."""

        if not loss_profile_lookup:
            return

        # Build reverse map: profile -> loss names declared in loss_profiles config
        configured_profile_losses: dict[str, set[str]] = {}
        for profile_name, profile_cfg in self._cfg.loss_profiles.items():
            configured_profile_losses[profile_name] = set(getattr(profile_cfg, "losses", []))

        for loss_name, loss_obj in losses.items():
            profiles: set[int] = set()

            cfg_attr = getattr(self._cfg.losses, loss_name, None)
            if cfg_attr is None and loss_name == "action_supervisor":
                cfg_attr = getattr(self._cfg.losses, "supervisor", None)

            explicit = getattr(cfg_attr, "profiles", None)
            if explicit:
                profiles |= {loss_profile_lookup[name] for name in explicit if name in loss_profile_lookup}

            for profile_name, losses_for_profile in configured_profile_losses.items():
                if loss_name in losses_for_profile and profile_name in loss_profile_lookup:
                    profiles.add(loss_profile_lookup[profile_name])

            if profiles:
                loss_obj.loss_profiles = profiles

    def _set_trainable_flag(self, policy: Policy, trainable: bool) -> None:
        """Set requires_grad according to slot.trainable."""

        for param in policy.parameters():
            param.requires_grad = trainable

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

    def _build_slot_state(self, trainer_policy: Policy) -> dict[str, Any]:
        slots_cfg = list(self._cfg.policy_slots or [])
        has_trainer_slot = any(b.use_trainer_policy for b in slots_cfg)
        if not slots_cfg or not has_trainer_slot:
            slots_cfg.insert(0, PolicySlotConfig(id="main", use_trainer_policy=True, trainable=True))

        trainer_slot_count = sum(1 for b in slots_cfg if b.use_trainer_policy)
        if trainer_slot_count > 1:
            raise ValueError("Only one slot may set use_trainer_policy=True")

        slot_lookup: dict[str, int] = {}
        slot_policies: dict[int, Policy] = {}
        for slot in slots_cfg:
            if slot.id in slot_lookup:
                raise ValueError(f"Duplicate policy slot id '{slot.id}'")
            slot_lookup[slot.id] = len(slot_lookup)
            if slot.use_trainer_policy:
                self._set_trainable_flag(trainer_policy, slot.trainable)
                slot_policies[slot_lookup[slot.id]] = trainer_policy
            else:
                loaded_policy = self._slot_registry.get(slot, self._env.policy_env_info, self._device)
                self._set_trainable_flag(loaded_policy, slot.trainable)
                slot_policies[slot_lookup[slot.id]] = loaded_policy

        # Loss profiles (config-only in Phase 0)
        loss_profiles = dict(self._cfg.loss_profiles)
        if not loss_profiles:
            loss_profiles = {"default": LossProfileConfig(losses=[])}
        loss_profile_lookup = {name: idx for idx, name in enumerate(loss_profiles.keys())}

        default_slot_profile = next(iter(loss_profiles))

        num_agents = self._env.policy_env_info.num_agents
        agent_slot_map = self._cfg.agent_slot_map
        if agent_slot_map is None:
            agent_slot_map = [slots_cfg[0].id for _ in range(num_agents)]
        if len(agent_slot_map) != num_agents:
            raise ValueError(f"agent_slot_map must have length num_agents ({num_agents}); got {len(agent_slot_map)}")

        slot_ids = []
        loss_profile_ids = []
        trainable_mask = []
        for idx, slot_id_str in enumerate(agent_slot_map):
            if slot_id_str not in slot_lookup:
                raise ValueError(f"agent_slot_map[{idx}] references unknown slot id '{slot_id_str}'")
            b_idx = slot_lookup[slot_id_str]
            slot = slots_cfg[b_idx]
            slot_ids.append(b_idx)

            profile_name = slot.loss_profile or default_slot_profile
            if profile_name not in loss_profile_lookup:
                # Auto-register profile if referenced but not defined
                loss_profile_lookup[profile_name] = len(loss_profile_lookup)
            loss_profile_ids.append(loss_profile_lookup[profile_name])
            trainable_mask.append(bool(slot.trainable))

        slot_tensor = torch.tensor(slot_ids, dtype=torch.long)
        loss_profile_tensor = torch.tensor(loss_profile_ids, dtype=torch.long)
        trainable_tensor = torch.tensor(trainable_mask, dtype=torch.bool)

        return {
            "slots": slots_cfg,
            "slot_lookup": slot_lookup,
            "loss_profile_lookup": loss_profile_lookup,
            "slot_ids": slot_tensor,
            "loss_profile_ids": loss_profile_tensor,
            "trainable_mask": trainable_tensor,
            "slot_policies": slot_policies,
        }

    def _extend_policy_experience_spec(self, base_spec: Composite, include_slot_metadata: bool) -> Composite:
        """Append slot/loss-profile metadata to the policy experience spec when requested."""

        if not include_slot_metadata:
            return base_spec

        extras = {
            "slot_id": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            "loss_profile_id": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            "is_trainable_agent": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool),
        }
        merged = dict(base_spec.items())
        merged.update(extras)
        return Composite(merged)

    def _invoke_callback(self, callback_type: TrainerCallback, infos: Optional[list[dict[str, Any]]] = None) -> None:
        """Invoke all registered callbacks of the specified type."""
        current_step = self._context.agent_step
        previous_step = getattr(self, "_prev_agent_step_for_step_callbacks", current_step)
        current_epoch = self._context.epoch

        for component in self._components:
            if callback_type == TrainerCallback.STEP:
                if component.should_handle_step(current_step=current_step, previous_step=previous_step) and infos:
                    component.on_step(infos)
            elif callback_type == TrainerCallback.EPOCH_END and component.should_handle_epoch(current_epoch):
                component.on_epoch_end(current_epoch)
            elif callback_type == TrainerCallback.ROLLOUT_END:
                component.on_rollout_end()
            elif callback_type == TrainerCallback.TRAINING_COMPLETE:
                component.on_training_complete()
            elif callback_type == TrainerCallback.FAILURE:
                component.on_failure()

    def restore(self) -> None:
        """Restore trainer state from checkpoints.

        This should be called after setup() to restore any saved state.
        """
        for component in self._components:
            if isinstance(component, ContextCheckpointer):
                component.restore(self._context)
                break
            # Wandb setup will be handled by callbacks if configured
