import logging
from typing import Any

import torch
from pydantic import ConfigDict
from tensordict import NonTensorData, TensorDict

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from metta.rl.utils import add_dummy_loss_for_unused_params, forward_policy_for_training
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class RolloutResult(Config):
    """Results from a rollout phase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_infos: list[dict[str, Any]]
    agent_steps: int
    training_env_id: slice


class CoreTrainingLoop:
    """Handles the core training loop with rollout and training phases."""

    def __init__(
        self,
        policy: Policy,
        experience: Experience,
        losses: dict[str, Loss],
        device: torch.device,
        context: ComponentContext,
    ):
        """Initialize core training loop.

        Args:
            policy: The policy to train
            experience: Experience buffer for storing rollouts
            losses: Dictionary of loss instances to use
            optimizer: Optimizer for policy updates
            device: Device to run on
        """
        self.policy = policy
        self.experience = experience
        self.losses = losses
        self.device = device
        self.accumulate_minibatches = experience.accumulate_minibatches
        self.context = context
        self.last_action = None

        # Cache environment indices to avoid reallocating per rollout batch
        self._env_index_cache = experience._range_tensor.to(device=device, dtype=torch.long)
        self._metadata_cache: dict[tuple[str, tuple[int, ...], int, str], torch.Tensor] = {}

        # Get policy spec for experience buffer
        self.policy_spec = policy.get_agent_experience_spec()

    def rollout_phase(
        self,
        env: TrainingEnvironment,
        context: ComponentContext,
    ) -> RolloutResult:
        """Perform rollout phase to collect experience.

        Args:
            env: Vectorized environment to collect from
            context: Shared trainer context providing rollout state

        Returns:
            RolloutResult with collected info
        """
        raw_infos: list[dict[str, Any]] = []
        self.experience.reset_for_rollout()

        # Notify losses of rollout start
        for loss in self.losses.values():
            loss.on_rollout_start(context)

        # Get buffer for storing experience
        buffer_step = self.experience.buffer[self.experience.row_slot_ids, self.experience.t_in_row - 1]
        buffer_step = buffer_step.select(*self.policy_spec.keys())

        total_steps = 0
        last_env_id: slice | None = None

        while not self.experience.ready_for_training:
            # Get observation from environment
            with context.stopwatch("_rollout.env_wait"):
                o, r, d, t, ta, info, training_env_id, _, num_steps = env.get_observations()
            last_env_id = training_env_id
            # Prepare data for policy
            with context.stopwatch("_rollout.td_prep"):
                td = buffer_step[training_env_id].clone()
                target_device = td.device
                td["env_obs"] = o.to(device=target_device, non_blocking=True)

                rewards = r.to(device=target_device, non_blocking=True)
                td["rewards"] = rewards
                agent_ids = self._gather_env_indices(training_env_id, td.device)
                td["training_env_ids"] = agent_ids.unsqueeze(1)

                avg_reward = context.state.avg_reward
                baseline = avg_reward[agent_ids]
                td["reward_baseline"] = baseline

                # CRITICAL FIX for MPS: Convert dtype BEFORE moving to device, and use blocking transfer
                # MPS has two bugs:
                # 1. bool->float32 conversion during .to(device=mps, dtype=float32) produces NaN
                # 2. non_blocking=True causes race conditions with uninitialized data
                # Solution: Convert dtype on CPU first, then use blocking transfer to MPS
                if target_device.type == "mps":
                    td["dones"] = d.to(dtype=torch.float32).to(device=target_device, non_blocking=False)
                    td["truncateds"] = t.to(dtype=torch.float32).to(device=target_device, non_blocking=False)
                else:
                    # On CUDA/CPU, combined conversion is safe and faster
                    td["dones"] = d.to(device=target_device, dtype=torch.float32, non_blocking=True)
                    td["truncateds"] = t.to(device=target_device, dtype=torch.float32, non_blocking=True)
                td["teacher_actions"] = ta.to(device=target_device, dtype=torch.long, non_blocking=True)
                # Row-aligned state: provide row slot id and position within row
                row_ids = self.experience.row_slot_ids[training_env_id].to(device=target_device, dtype=torch.long)
                t_in_row = self.experience.t_in_row[training_env_id].to(device=target_device, dtype=torch.long)
                td["row_id"] = row_ids
                td["t_in_row"] = t_in_row
                self.add_last_action_to_td(td)

                self._ensure_rollout_metadata(td)

            # Allow losses to mutate td (policy inference, bookkeeping, etc.)
            with context.stopwatch("_rollout.inference"):
                context.training_env_id = training_env_id
                trajectory_isolator = getattr(context, "trajectory_isolator", None)
                if trajectory_isolator is not None:
                    trajectory_isolator.route_rollout(td, training_env_id=training_env_id, context=context)
                for loss in self.losses.values():
                    loss.rollout(td, context)

            self.experience.store(data_td=td, env_id=training_env_id)

            avg_reward = context.state.avg_reward
            beta = float(context.config.advantage.reward_centering.beta)
            with torch.no_grad():
                rewards_f32 = td["rewards"].to(dtype=torch.float32)
                avg_reward[agent_ids] = baseline + beta * (rewards_f32 - baseline)
            context.state.avg_reward = avg_reward

            assert "actions" in td, "No loss performed inference - at least one loss must generate actions"
            raw_actions = td["actions"].detach()
            if raw_actions.dim() != 1:
                raise ValueError(
                    "Policies must emit a single discrete action id per agent; "
                    f"received tensor of shape {tuple(raw_actions.shape)}"
                )

            actions_column = raw_actions.view(-1, 1)

            if self.last_action.device != actions_column.device:
                self.last_action = self.last_action.to(device=actions_column.device)

            if self.last_action.dtype != actions_column.dtype:
                actions_column = actions_column.to(dtype=self.last_action.dtype)

            target_buffer = self.last_action[training_env_id]
            if target_buffer.shape != actions_column.shape:
                msg = "last_action buffer shape mismatch: target=%s actions=%s raw=%s" % (
                    target_buffer.shape,
                    actions_column.shape,
                    tuple(td["actions"].shape),
                )
                logger.error(msg, exc_info=True)
                raise RuntimeError(msg)

            target_buffer.copy_(actions_column)

            # Ship actions to the environment
            with context.stopwatch("_rollout.send"):
                env.send_actions(td["actions"].cpu().numpy())

            infos_list: list[dict[str, Any]] = list(info) if info else []
            if infos_list:
                raw_infos.extend(infos_list)

            total_steps += num_steps

        context.training_env_id = last_env_id
        return RolloutResult(raw_infos=raw_infos, agent_steps=total_steps, training_env_id=last_env_id)

    def _gather_env_indices(self, training_env_id: slice, device: torch.device) -> torch.Tensor:
        env_indices = self._env_index_cache[training_env_id]
        if env_indices.device != device:
            env_indices = env_indices.to(device=device)
        return env_indices

    def _ensure_rollout_metadata(self, td: TensorDict) -> None:
        """Populate metadata fields needed downstream while reusing cached tensors."""

        batch_elems = td.batch_size.numel()
        device = td.device
        if "batch" not in td.keys():
            td.set("batch", self._get_constant_tensor("batch", (batch_elems,), batch_elems, device))
        if "bptt" not in td.keys():
            td.set("bptt", self._get_constant_tensor("bptt", (batch_elems,), 1, device))

    def _get_constant_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        value: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not shape:
            shape = (1,)
        key = (name, shape, int(value), str(device))
        cached = self._metadata_cache.get(key)
        if cached is None or cached.device != device:
            if value == 1:
                tensor = torch.ones(shape, dtype=torch.long, device=device)
            else:
                tensor = torch.full(shape, value, dtype=torch.long, device=device)
            self._metadata_cache[key] = tensor
            return tensor
        return cached

    def training_phase(
        self,
        context: ComponentContext,
        update_epochs: int,
        max_grad_norm: float = 0.5,
    ) -> tuple[dict[str, float], int]:
        """Perform training phase on collected experience.

        Args:
            context: Shared trainer context providing training state
            update_epochs: Number of epochs to train for
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary of loss statistics
        """
        self.experience.reset_importance_sampling_ratios()

        for loss in self.losses.values():
            loss.zero_loss_tracker()

        epochs_trained = 0

        for _ in range(update_epochs):
            if "values" in self.experience.buffer.keys():
                centered_rewards = self.experience.buffer["rewards"] - self.experience.buffer["reward_baseline"]
                advantages = compute_advantage(
                    self.experience.buffer["values"],
                    centered_rewards,
                    self.experience.buffer["dones"],
                    torch.ones_like(self.experience.buffer["values"]),
                    torch.zeros_like(self.experience.buffer["values"], device=self.device),
                    self.context.config.advantage.gamma,
                    self.context.config.advantage.gae_lambda,
                    self.device,
                    self.context.config.advantage.vtrace_rho_clip,
                    self.context.config.advantage.vtrace_c_clip,
                )
            else:
                # Value-free setups still need a tensor shaped like the buffer for sampling.
                advantages = torch.zeros(self.experience.buffer.batch_size, device=self.device, dtype=torch.float32)
            self.experience.buffer["advantages_full"] = advantages

            stop_update_epoch = False
            for mb_idx in range(self.experience.num_minibatches):
                optimizer = getattr(self.policy, "optimizer", None)
                if optimizer is None:
                    raise RuntimeError(
                        "CoreTrainingLoop.training_phase requires an optimizer attached to the active policy "
                        "(policy.optimizer is None)."
                    )
                if mb_idx % self.accumulate_minibatches == 0:
                    optimizer.zero_grad()

                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                stop_update_epoch_mb = False

                shared_loss_mb_data = self.experience.sample(
                    mb_idx=mb_idx,
                    epoch=context.epoch,
                    total_timesteps=context.config.total_timesteps,
                    batch_size=context.config.batch_size,
                    advantages=advantages,
                )
                if mb_idx == 0:
                    shared_loss_mb_data["advantages_full"] = NonTensorData(advantages)

                policy_td = shared_loss_mb_data["sampled_mb"]
                policy_td = forward_policy_for_training(self.policy, policy_td, self.policy_spec)
                shared_loss_mb_data["policy_td"] = policy_td

                used_keys: set[str] = set()
                for _loss_name, loss_obj in self.losses.items():
                    if loss_obj._loss_gate_allows("train", context):
                        used_keys.update(loss_obj.policy_output_keys(policy_td))
                    loss_val, shared_loss_mb_data, loss_requests_stop = loss_obj.train(
                        shared_loss_mb_data, context, mb_idx
                    )
                    total_loss = total_loss + loss_val
                    stop_update_epoch_mb = stop_update_epoch_mb or loss_requests_stop

                if stop_update_epoch_mb:
                    stop_update_epoch = True
                    break

                # Ensure all policy outputs participate in the graph even if some heads
                # aren't used by the active losses (e.g., BC-only runs). This avoids
                # DDP unused-parameter errors without relying on find_unused_parameters.
                total_loss = add_dummy_loss_for_unused_params(total_loss, td=policy_td, used_keys=used_keys)

                total_loss.backward()

                # Optimizer step with gradient accumulation
                if (mb_idx + 1) % self.accumulate_minibatches == 0:
                    # Get max_grad_norm from first loss that has it
                    actual_max_grad_norm = max_grad_norm
                    for loss_obj in self.losses.values():
                        if hasattr(loss_obj.cfg, "max_grad_norm"):
                            actual_max_grad_norm = loss_obj.cfg.max_grad_norm
                            break

                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), actual_max_grad_norm)
                    optimizer.step()

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                # Notify losses of minibatch end
                for loss_obj in self.losses.values():
                    loss_obj.on_mb_end(context, mb_idx)

            epochs_trained += 1
            if stop_update_epoch:
                break

        # Notify losses of training phase end
        for loss_obj in self.losses.values():
            loss_obj.on_train_phase_end(context)

        # Collect statistics from all losses
        losses_stats = {}
        for _loss_name, loss_obj in self.losses.items():
            losses_stats.update(loss_obj.stats())

        return losses_stats, epochs_trained

    def on_epoch_start(self, context: ComponentContext | None = None) -> None:
        """Called at the start of each epoch.

        Args:
            context: Shared trainer context providing epoch state
        """
        for loss in self.losses.values():
            loss.on_epoch_start(context)

    def add_last_action_to_td(self, td: TensorDict) -> None:
        env_ids = td["training_env_ids"].squeeze(-1)

        max_env_id = int(env_ids.max().item())
        target_length = max_env_id + 1

        if self.last_action is None:
            self.last_action = torch.zeros(target_length, 1, dtype=torch.int32, device=td.device)
        else:
            if self.last_action.size(0) < target_length:
                pad_shape = (target_length - self.last_action.size(0), self.last_action.size(1))
                pad_tensor = torch.zeros(pad_shape, dtype=self.last_action.dtype, device=self.last_action.device)
                self.last_action = torch.cat((self.last_action, pad_tensor), dim=0)
            if self.last_action.device != td.device:
                self.last_action = self.last_action.to(device=td.device)

        td["last_actions"] = self.last_action[env_ids].detach()
