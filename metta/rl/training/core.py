import logging
from typing import Any

import torch
from pydantic import ConfigDict
from tensordict import NonTensorData, TensorDict

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, compute_delta_lambda
from metta.rl.loss.loss import Loss
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from metta.rl.utils import add_dummy_loss_for_unused_params, ensure_sequence_metadata, forward_policy_for_training
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
        optimizer: torch.optim.Optimizer,
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
        self.optimizer = optimizer
        self.device = device
        self.accumulate_minibatches = experience.accumulate_minibatches
        self.context = context
        self.last_action = torch.zeros(
            experience.total_agents,
            1,
            dtype=torch.int32,
            device=device,
        )
        # Cache environment indices to avoid reallocating per rollout batch
        self._env_index_cache = experience._range_tensor.to(device=device)
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
                agent_ids = self._env_index_cache[training_env_id]
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
                row_ids = self.experience.row_slot_ids[training_env_id]
                t_in_row = self.experience.t_in_row[training_env_id]
                td["row_id"] = row_ids
                td["t_in_row"] = t_in_row
                self.add_last_action_to_td(td)

                ensure_sequence_metadata(td, batch_size=td.batch_size.numel(), time_steps=1)
                self._inject_slot_metadata(td, training_env_id)

            # Allow losses to mutate td (policy inference, bookkeeping, etc.)
            with context.stopwatch("_rollout.inference"):
                context.training_env_id = training_env_id
                for loss in self.losses.values():
                    loss.rollout(td, context)

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

    def _inject_slot_metadata(self, td: TensorDict, training_env_id: slice) -> None:
        ctx = self.context
        slot_ids = ctx.slot_id_per_agent
        if slot_ids is None:
            return

        num_agents = ctx.env.policy_env_info.num_agents
        if slot_ids.numel() != num_agents:
            raise RuntimeError(f"slot_id_per_agent expected {num_agents} entries, got {slot_ids.numel()}")

        batch_elems = td.batch_size.numel()
        if batch_elems % num_agents:
            raise RuntimeError(
                f"Batch elements ({batch_elems}) must be divisible by num_agents ({num_agents}) for slot metadata"
            )

        num_envs = batch_elems // num_agents
        device = td.device

        def _expand(meta: torch.Tensor) -> torch.Tensor:
            meta = meta.to(device=device)
            return meta.view(1, num_agents).expand(num_envs, num_agents)

        td.set("slot_id", _expand(slot_ids))

        loss_profile_ids = ctx.loss_profile_id_per_agent
        if loss_profile_ids is not None:
            td.set("loss_profile_id", _expand(loss_profile_ids))

        trainable_mask = ctx.trainable_agent_mask
        if trainable_mask is not None:
            td.set("is_trainable_agent", _expand(trainable_mask))
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

        advantage_cfg = context.config.advantage
        ppo_critic = self.losses.get("ppo_critic")
        use_delta_lambda = (
            ppo_critic is not None
            and getattr(ppo_critic.cfg, "critic_update", None) == "gtd_lambda"
            and ppo_critic._loss_gate_allows("train", context)
        )
        advantage_method = "delta_lambda" if use_delta_lambda else "vtrace"

        epochs_trained = 0

        for _ in range(update_epochs):
            if "values" in self.experience.buffer.keys():
                values_for_adv = self.experience.buffer["values"]
                if values_for_adv.dim() > 2:
                    values_for_adv = values_for_adv.mean(dim=-1)
                centered_rewards = self.experience.buffer["rewards"] - self.experience.buffer["reward_baseline"]
                advantages_full = compute_advantage(
                    values_for_adv,
                    centered_rewards,
                    self.experience.buffer["dones"],
                    torch.ones_like(values_for_adv),
                    torch.zeros_like(values_for_adv, device=self.device),
                    advantage_cfg.gamma,
                    advantage_cfg.gae_lambda,
                    self.device,
                    advantage_cfg.vtrace_rho_clip,
                    advantage_cfg.vtrace_c_clip,
                )
            else:
                # Value-free setups still need a tensor shaped like the buffer for sampling.
                advantages_full = torch.zeros(
                    self.experience.buffer.batch_size,
                    device=self.device,
                    dtype=torch.float32,
                )
            self.experience.buffer["advantages_full"] = advantages_full

            stop_update_epoch = False
            for mb_idx in range(self.experience.num_minibatches):
                if mb_idx % self.accumulate_minibatches == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                stop_update_epoch_mb = False

                shared_loss_mb_data = self.experience.sample(
                    mb_idx=mb_idx,
                    epoch=context.epoch,
                    total_timesteps=context.config.total_timesteps,
                    batch_size=context.config.batch_size,
                    advantages=advantages_full,
                )
                if mb_idx == 0:
                    shared_loss_mb_data["advantages_full"] = NonTensorData(advantages_full)

                policy_td = shared_loss_mb_data["sampled_mb"]
                policy_td = forward_policy_for_training(self.policy, policy_td, self.policy_spec)
                shared_loss_mb_data["policy_td"] = policy_td

                sampled_mb = shared_loss_mb_data["sampled_mb"]
                if "act_log_prob" in sampled_mb.keys() and "act_log_prob" in policy_td.keys():
                    old_logprob = sampled_mb["act_log_prob"]
                    new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
                    logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
                    shared_loss_mb_data["importance_sampling_ratio"] = logratio.exp()

                if advantage_method == "delta_lambda":
                    if "values" not in sampled_mb.keys():
                        raise RuntimeError("delta_lambda advantages require minibatch['values']")

                    new_values = policy_td["values"]
                    if new_values.dim() == 3 and new_values.shape[-1] == 1:
                        new_values = new_values.squeeze(-1)
                    new_values = new_values.reshape(sampled_mb["values"].shape)

                    centered_rewards = sampled_mb["rewards"] - sampled_mb["reward_baseline"]
                    shared_loss_mb_data["advantages_pg"] = compute_delta_lambda(
                        values=new_values,
                        rewards=centered_rewards,
                        dones=sampled_mb["dones"],
                        gamma=float(advantage_cfg.gamma),
                        gae_lambda=float(advantage_cfg.gae_lambda),
                    )
                else:
                    values_for_adv = sampled_mb["values"] if "values" in sampled_mb.keys() else None
                    if values_for_adv is not None:
                        if values_for_adv.dim() > 2:
                            values_for_adv = values_for_adv.mean(dim=-1)

                        importance_sampling_ratio = shared_loss_mb_data.get("importance_sampling_ratio", None)
                        if importance_sampling_ratio is None:
                            importance_sampling_ratio = torch.ones_like(values_for_adv)

                        with torch.no_grad():
                            centered_rewards = sampled_mb["rewards"] - sampled_mb["reward_baseline"]
                            shared_loss_mb_data["advantages_pg"] = compute_advantage(
                                values_for_adv,
                                centered_rewards,
                                sampled_mb["dones"],
                                importance_sampling_ratio,
                                shared_loss_mb_data["advantages"].clone(),
                                advantage_cfg.gamma,
                                advantage_cfg.gae_lambda,
                                self.device,
                                advantage_cfg.vtrace_rho_clip,
                                advantage_cfg.vtrace_c_clip,
                            )
                    else:
                        shared_loss_mb_data["advantages_pg"] = shared_loss_mb_data["advantages"]

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
                    self.optimizer.step()

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

        if self.last_action.device != td.device:
            self.last_action = self.last_action.to(device=td.device)

        td["last_actions"] = self.last_action[env_ids].detach()
