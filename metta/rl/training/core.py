"""Core training loop for rollout and training phases."""

import logging
from typing import Any, Dict, List

import torch
from pydantic import ConfigDict

from metta.agent.policy import Policy
from metta.mettagrid.config import Config
from metta.rl.loss.loss import Loss
from metta.rl.training.experience import Experience
from metta.rl.training.training_environment import TrainingEnvironment

logger = logging.getLogger(__name__)


class RolloutResult(Config):
    """Results from a rollout phase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_infos: List[Dict[str, Any]]
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
        accumulate_minibatches: int = 1,
    ):
        """Initialize core training loop.

        Args:
            policy: The policy to train
            experience: Experience buffer for storing rollouts
            losses: Dictionary of loss instances to use
            optimizer: Optimizer for policy updates
            device: Device to run on
            accumulate_minibatches: Number of minibatches to accumulate before optimizer step
        """
        self.policy = policy
        self.experience = experience
        self.losses = losses
        self.optimizer = optimizer
        self.device = device
        self.accumulate_minibatches = accumulate_minibatches

        # Get policy spec for experience buffer
        self.policy_spec = policy.get_agent_experience_spec()

    def rollout_phase(
        self,
        env: TrainingEnvironment,
        epoch: int,
    ) -> RolloutResult:
        """Perform rollout phase to collect experience.

        Args:
            env: Vectorized environment to collect from
            trainer_state: Current trainer state

        Returns:
            RolloutResult with collected info
        """
        raw_infos = []
        self.experience.reset_for_rollout()

        # Notify losses of rollout start
        for loss in self.losses.values():
            loss.on_rollout_start(epoch)

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
            td["env_obs"] = o.to(td.device)
            td["rewards"] = r.to(td.device)
            td["dones"] = d.float().to(td.device)
            td["truncateds"] = t.float().to(td.device)
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
            B = td.batch_size.numel()
            td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))

            # Run rollout hooks for all losses
            # Each loss can modify td and potentially run inference
            for loss in self.losses.values():
                loss.rollout(td, training_env_id, epoch)

            # At least one loss should have performed inference
            assert "actions" in td, "No loss performed inference - at least one loss must generate actions"

            # Send actions to environment
            env.send_actions(td["actions"].cpu().numpy())

            if info:
                raw_infos.extend(info)

            total_steps += num_steps

        if last_env_id is None:
            raise RuntimeError("Rollout completed without receiving any environment data")

        return RolloutResult(raw_infos=raw_infos, agent_steps=total_steps, training_env_id=last_env_id)

    def training_phase(
        self,
        epoch: int,
        training_env_id: slice,
        update_epochs: int,
        max_grad_norm: float = 0.5,
    ) -> tuple[Dict[str, float], int]:
        """Perform training phase on collected experience.

        Args:
            epoch: Current epoch
            training_env_id: Training environment ID
            update_epochs: Number of epochs to train for
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary of loss statistics
        """
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
        stop_update_epoch = False

        for _ in range(update_epochs):
            for mb_idx in range(self.experience.num_minibatches):
                # Compute total loss from all losses
                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

                for _loss_name, loss_obj in self.losses.items():
                    loss_val, shared_loss_mb_data, stop_update_epoch = loss_obj.train(
                        shared_loss_mb_data, training_env_id, epoch, mb_idx
                    )
                    total_loss = total_loss + loss_val

                # Backward pass
                self.optimizer.zero_grad()
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

                    clip_target = self.policy
                    if hasattr(self.policy, "module"):
                        clip_target = self.policy.module  # DDP unwrap
                    if hasattr(clip_target, "clip_weights"):
                        clip_target.clip_weights()

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                # Notify losses of minibatch end
                for loss_obj in self.losses.values():
                    loss_obj.on_mb_end(epoch, mb_idx)

                if stop_update_epoch:
                    break

            epochs_trained += 1
            if stop_update_epoch:
                break

        # Notify losses of training phase end
        for loss_obj in self.losses.values():
            loss_obj.on_train_phase_end(epoch)

        # Collect statistics from all losses
        losses_stats = {}
        for _loss_name, loss_obj in self.losses.items():
            losses_stats.update(loss_obj.stats())

        return losses_stats, epochs_trained

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch
        """
        for loss in self.losses.values():
            loss.on_new_training_run()
