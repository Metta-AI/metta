import logging
from typing import Any, Dict, List

import torch
from pydantic import ConfigDict

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss
from metta.rl.training.component_context import ComponentContext
from metta.rl.training.experience import Experience
from metta.rl.training.training_environment import TrainingEnvironment
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class RolloutResult(Config):
    """Results from a rollout phase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_infos: List[Dict[str, Any]]
    events: List[tuple[int, List[Dict[str, Any]]]]
    agent_steps: int
    training_env_id: slice


class CoreTrainingLoop:
    """Handles the core training loop with rollout and training phases."""

    def __init__(
        self,
        policy: Policy,
        experience: Experience,
        losses: Dict[str, Loss],
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
        raw_infos: List[Dict[str, Any]] = []
        events: List[tuple[int, List[Dict[str, Any]]]] = []
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
            o, r, d, t, info, training_env_id, _, num_steps = env.get_observations()
            last_env_id = training_env_id

            # Prepare data for policy
            td = buffer_step[training_env_id].clone()
            target_device = td.device
            td["env_obs"] = o.to(device=target_device, non_blocking=True)
            td["rewards"] = r.to(device=target_device, non_blocking=True)
            td["dones"] = d.to(device=target_device, dtype=torch.float32, non_blocking=True)
            td["truncateds"] = t.to(device=target_device, dtype=torch.float32, non_blocking=True)
            env_indices = torch.arange(training_env_id.start, training_env_id.stop, dtype=torch.long, device=td.device)
            td["training_env_ids"] = env_indices.unsqueeze(1)
            td["training_env_id"] = torch.full(
                td.batch_size,
                training_env_id.start,
                dtype=torch.long,
                device=td.device,
            )
            td["training_env_id_start"] = torch.full(
                td.batch_size,
                training_env_id.start,
                dtype=torch.long,
                device=td.device,
            )
            batch_elems = td.batch_size.numel()
            device = td.device
            if "batch" not in td.keys():
                td.set("batch", torch.full((batch_elems,), batch_elems, dtype=torch.long, device=device))
            if "bptt" not in td.keys():
                td.set("bptt", torch.ones(batch_elems, dtype=torch.long, device=device))

            # Allow losses to mutate td (policy inference, bookkeeping, etc.)
            context.training_env_id = training_env_id
            for loss in self.losses.values():
                loss.rollout(td, context)

            assert "actions" in td, "No loss performed inference - at least one loss must generate actions"

            # Ship actions to the environment
            env.send_actions(td["actions"].cpu().numpy())

            infos_list: List[Dict[str, Any]] = list(info) if info else []
            if infos_list:
                raw_infos.extend(infos_list)

            events.append((num_steps, infos_list))

            total_steps += num_steps

        if last_env_id is None:
            raise RuntimeError("Rollout completed without receiving any environment data")

        context.training_env_id = last_env_id
        return RolloutResult(raw_infos=raw_infos, events=events, agent_steps=total_steps, training_env_id=last_env_id)

    def training_phase(
        self,
        context: ComponentContext,
        update_epochs: int,
        max_grad_norm: float = 0.5,
    ) -> tuple[Dict[str, float], int]:
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

                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                stop_update_epoch_mb = False

                for _loss_name, loss_obj in self.losses.items():
                    loss_val, shared_loss_mb_data, loss_requests_stop = loss_obj.train(
                        shared_loss_mb_data, context, mb_idx
                    )
                    total_loss = total_loss + loss_val
                    stop_update_epoch_mb = stop_update_epoch_mb or loss_requests_stop

                if stop_update_epoch_mb:
                    stop_update_epoch = True
                    break

                total_loss.backward()

                # Optimizer step with gradient accumulation
                if (mb_idx + 1) % self.accumulate_minibatches == 0:
                    # Get max_grad_norm from first loss that has it
                    actual_max_grad_norm = max_grad_norm
                    for loss_obj in self.losses.values():
                        if hasattr(loss_obj.loss_cfg, "max_grad_norm"):
                            actual_max_grad_norm = loss_obj.loss_cfg.max_grad_norm
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

    def on_epoch_start(self, context: ComponentContext) -> None:
        """Called at the start of each epoch.

        Args:
            context: Shared trainer context providing epoch state
        """
        for loss in self.losses.values():
            loss.on_new_training_run(context)
