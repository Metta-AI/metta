import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from pydantic import ConfigDict
from tensordict import TensorDict
from torch.cuda.amp import GradScaler, autocast

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from mettagrid.config import Config

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
        *,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.float32,
        grad_scaler: Optional[GradScaler] = None,
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
        self.last_action = None
        self._amp_enabled = amp_enabled and self.device.type == "cuda"
        self._amp_dtype = amp_dtype if self._amp_enabled else torch.float32
        self._grad_scaler: Optional[GradScaler] = grad_scaler if self._amp_enabled else None

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
        buffer_step = self.experience.buffer[self.experience.ep_indices, self.experience.ep_lengths - 1]
        buffer_step = buffer_step.select(*self.policy_spec.keys())

        total_steps = 0
        last_env_id: slice | None = None

        while not self.experience.ready_for_training:
            # Get observation from environment
            with context.stopwatch("rollout.env_fetch"):
                o, r, d, t, info, training_env_id, _, num_steps = env.get_observations()
            last_env_id = training_env_id

            # Prepare data for policy
            td = buffer_step[training_env_id].clone()
            target_device = td.device
            with context.stopwatch("rollout.tensor_transfer"):
                td["env_obs"] = o.to(device=target_device, non_blocking=True)
                td["rewards"] = r.to(device=target_device, non_blocking=True)
                td["dones"] = d.to(device=target_device, dtype=torch.float32, non_blocking=True)
                td["truncateds"] = t.to(device=target_device, dtype=torch.float32, non_blocking=True)
                td["training_env_ids"] = self._gather_env_indices(training_env_id, td.device).unsqueeze(1)
            self.add_last_action_to_td(td, env)

            self._ensure_rollout_metadata(td, training_env_id)

            # Allow losses to mutate td (policy inference, bookkeeping, etc.)
            context.training_env_id = training_env_id
            for loss in self.losses.values():
                loss.rollout(td, context)

            assert "actions" in td, "No loss performed inference - at least one loss must generate actions"
            self.last_action[training_env_id] = td["actions"].detach()

            # Ship actions to the environment
            env.send_actions(td["actions"].cpu().numpy())

            infos_list: list[dict[str, Any]] = list(info) if info else []
            if infos_list:
                raw_infos.extend(infos_list)

            total_steps += num_steps

        if last_env_id is None:
            raise RuntimeError("Rollout completed without receiving any environment data")

        fetch_time = context.stopwatch.get_elapsed("rollout.env_fetch")
        rollout_time = context.stopwatch.get_elapsed("_rollout")
        if rollout_time > 0 and fetch_time > 0:
            fetch_ratio = fetch_time / rollout_time
            if fetch_ratio > 0.1:
                logger.info(
                    "Rollout env fetch consumed %.1f%% of rollout time (%.3fs / %.3fs)",
                    fetch_ratio * 100,
                    fetch_time,
                    rollout_time,
                )
            else:
                logger.debug(
                    "Rollout env fetch ratio %.2f%% (%.3fs of %.3fs)",
                    fetch_ratio * 100,
                    fetch_time,
                    rollout_time,
                )

        context.training_env_id = last_env_id
        return RolloutResult(raw_infos=raw_infos, agent_steps=total_steps, training_env_id=last_env_id)

    def _gather_env_indices(self, training_env_id: slice, device: torch.device) -> torch.Tensor:
        env_indices = self._env_index_cache[training_env_id]
        if env_indices.device != device:
            env_indices = env_indices.to(device=device)
        return env_indices

    def _ensure_rollout_metadata(self, td: TensorDict, training_env_id: slice) -> None:
        """Populate metadata fields needed downstream while reusing cached tensors."""

        batch_elems = td.batch_size.numel()
        device = td.device
        diag_enabled = os.getenv("TRANSFORMER_DIAG", "0") == "1"

        if "batch" not in td.keys():
            td.set("batch", self._get_constant_tensor("batch", (batch_elems,), batch_elems, device))
        if "bptt" not in td.keys():
            td.set("bptt", self._get_constant_tensor("bptt", (batch_elems,), 1, device))

        if diag_enabled:
            batch_tensor = td.get("batch")
            bptt_tensor = td.get("bptt")
            logger.info(
                "[TRANSFORMER_DIAG] rollout metadata batch=%s bptt=%s",
                batch_tensor[0].item() if batch_tensor is not None and batch_tensor.numel() else None,
                bptt_tensor[0].item() if bptt_tensor is not None and bptt_tensor.numel() else None,
            )

        training_env_shape = tuple(int(dim) for dim in td.batch_size) or (batch_elems,)
        env_start = training_env_id.start if training_env_id.start is not None else 0

        if "training_env_id" not in td.keys():
            td.set(
                "training_env_id",
                self._get_constant_tensor("training_env_id", training_env_shape, env_start, device),
            )
        if "training_env_id_start" not in td.keys():
            td.set(
                "training_env_id_start",
                self._get_constant_tensor("training_env_id_start", training_env_shape, env_start, device),
            )

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
        training_env_id = context.training_env_id
        assert training_env_id is not None, "Training environment ID is required"

        # Initialize shared loss data
        shared_loss_mb_data = self.experience.give_me_empty_md_td()
        for loss_name in self.losses.keys():
            shared_loss_mb_data[loss_name] = self.experience.give_me_empty_md_td()

        # Reset loss tracking
        shared_loss_mb_data.zero_()
        self.experience.reset_importance_sampling_ratios()

        for loss in self.losses.values():
            loss.zero_loss_tracker()

        epochs_trained = 0

        for _ in range(update_epochs):
            stop_update_epoch = False
            for mb_idx in range(self.experience.num_minibatches):
                if mb_idx % self.accumulate_minibatches == 0:
                    self.optimizer.zero_grad()

                total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                stop_update_epoch_mb = False

                autocast_enabled = self._amp_enabled and self.device.type == "cuda"
                with autocast(
                    device_type="cuda",
                    dtype=self._amp_dtype,
                    enabled=autocast_enabled,
                ):
                    for _loss_name, loss_obj in self.losses.items():
                        loss_val, shared_loss_mb_data, loss_requests_stop = loss_obj.train(
                            shared_loss_mb_data, context, mb_idx
                        )
                        total_loss = total_loss + loss_val
                        stop_update_epoch_mb = stop_update_epoch_mb or loss_requests_stop

                if stop_update_epoch_mb:
                    stop_update_epoch = True
                    break

                if self._grad_scaler is not None and self._grad_scaler.is_enabled():
                    self._grad_scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                # Optimizer step with gradient accumulation
                if (mb_idx + 1) % self.accumulate_minibatches == 0:
                    # Get max_grad_norm from first loss that has it
                    actual_max_grad_norm = max_grad_norm
                    for loss_obj in self.losses.values():
                        if hasattr(loss_obj.loss_cfg, "max_grad_norm"):
                            actual_max_grad_norm = loss_obj.loss_cfg.max_grad_norm
                            break

                    if self._grad_scaler is not None and self._grad_scaler.is_enabled():
                        self._grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), actual_max_grad_norm)
                        self._grad_scaler.step(self.optimizer)
                        self._grad_scaler.update()
                    else:
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

    def on_epoch_start(self, context: ComponentContext) -> None:
        """Called at the start of each epoch.

        Args:
            context: Shared trainer context providing epoch state
        """
        for loss in self.losses.values():
            loss.on_new_training_run(context)

    def add_last_action_to_td(self, td: TensorDict, env: TrainingEnvironment) -> None:
        env_ids = td["training_env_ids"]
        if env_ids.dim() == 2:
            env_ids = td["training_env_ids"].squeeze(-1)
        if self.last_action is None or len(self.last_action) < env_ids.max() + 1:
            act_space = env.single_action_space
            act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
            self.last_action = torch.zeros(env_ids.max() + 1, len(act_space.nvec), dtype=act_dtype, device=td.device)
        td["last_actions"] = self.last_action[env_ids].detach()
