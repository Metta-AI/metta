"""Loss computation functions for PPO training."""

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

from metta.agent.metta_agent import DistributedMettaAgent
from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed

logger = logging.getLogger(__name__)


class Loss:
    """
    Control flow:
    - just one Loss class is used for all losses
    - Loss gets trainer state in addition to the rest
    - losses are all called
    - all losses have a function to determine if they are active based on trainer state
    - losses can instantiate their own hyper scheduler
    losses are added in Loss and returned
    """

    def __init__(self, policy: DistributedMettaAgent, trainer_cfg: Any, device: torch.device, losses: Losses):
        self.policy = policy
        self.trainer_cfg = trainer_cfg
        self.device = device
        self.losses = losses

    def get_experience_spec(self) -> TensorDict:
        raise NotImplementedError("get_experience_spec not implemented for base class")

    def find_components_recursively(self, leaf: str, target: str) -> list[str]:
        """Recursively walk the MettaAgent and find the component names between a single leaf and a single target
        component. It includes the leaf but not the target in the list.
        Run this function for each leaf and target pair if necessary."""

        def _check_component_name(node: str, target: str, keys: list[str]) -> None:
            sources = getattr(self.policy.components[node], "_sources", None)
            if sources is None:
                return
            for source in sources:
                if source["name"] != target:
                    keys.append(source["name"])
                    _check_component_name(source["name"], target, keys)
                else:
                    keys.append(target)
                    return

        keys = []
        _check_component_name(leaf, target, keys)

        return keys


class PPO(Loss):
    def __init__(
        self,
        policy: DistributedMettaAgent,
        experience: Optional[Experience],
        trainer_cfg: Any,
        device: torch.device,
        losses: Losses,
    ):
        super().__init__(policy, trainer_cfg, device, losses)
        self.experience = experience

    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {
                "env_obs": torch.zeros(*self.policy.agent_attributes["obs_shape"], dtype=torch.uint8),
                "rewards": torch.zeros((), dtype=torch.float32),
                "dones": torch.zeros((), dtype=torch.float32),
                "truncateds": torch.zeros((), dtype=torch.float32),
                "actions": torch.zeros(self.policy.agent_attributes["action_space"].shape, dtype=torch.int32),
                "logprobs": torch.zeros((), dtype=torch.float32),
                "values": torch.zeros((), dtype=torch.float32),
            },
            batch_size=[],
        )

    def __call__(
        self,
        minibatch: TensorDict,
        train_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
        kickstarter: Any,
        agent_step: int,
    ) -> Tensor:
        """Process a single minibatch update and return the total loss."""
        # The policy's training forward pass returns a TD with required tensors for loss calculation.
        new_logprobs = train_td["action_log_prob"].reshape(minibatch["logprobs"].shape)
        entropy = train_td["entropy"]
        newvalue = train_td["value"]
        full_logprobs = train_td["log_probs"]

        logratio = new_logprobs - minibatch["logprobs"]
        importance_sampling_ratio = logratio.exp()

        # Re-compute advantages with new ratios (V-trace)
        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            self.trainer_cfg.ppo.gamma,
            self.trainer_cfg.ppo.gae_lambda,
            self.trainer_cfg.vtrace.vtrace_rho_clip,
            self.trainer_cfg.vtrace.vtrace_c_clip,
            self.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, self.trainer_cfg.ppo.norm_adv)
        adv = prio_weights * adv

        # Compute losses
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
            minibatch,
            new_logprobs,
            entropy,
            newvalue,
            importance_sampling_ratio,
            adv,
        )

        # Kickstarter losses
        ks_action_loss, ks_value_loss = kickstarter.loss(
            agent_step,
            full_logprobs,
            newvalue,
            minibatch["env_obs"],
            teacher_lstm_state=[],
        )

        # L2 init loss
        l2_init_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.trainer_cfg.ppo.l2_init_loss_coef > 0:
            l2_init_loss = self.trainer_cfg.ppo.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

        # Total loss
        loss = (
            pg_loss
            - self.trainer_cfg.ppo.ent_coef * entropy_loss
            + v_loss * self.trainer_cfg.ppo.vf_coef
            + l2_init_loss
            + ks_action_loss
            + ks_value_loss
        )

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape), "ratio": importance_sampling_ratio},
            batch_size=minibatch.batch_size,
        )
        self.experience.update(indices, update_td)

        # Update loss tracking
        self.losses.policy_loss_sum += pg_loss.item()
        self.losses.value_loss_sum += v_loss.item()
        self.losses.entropy_sum += entropy_loss.item()
        self.losses.approx_kl_sum += approx_kl.item()
        self.losses.clipfrac_sum += clipfrac.item()
        self.losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
        self.losses.ks_action_loss_sum += ks_action_loss.item()
        self.losses.ks_value_loss_sum += ks_value_loss.item()
        self.losses.importance_sum += importance_sampling_ratio.mean().item()
        self.losses.minibatches_processed += 1
        self.losses.current_logprobs_sum += new_logprobs.mean().item()

        return loss

    def compute_ppo_losses(
        self,
        minibatch: TensorDict,
        new_logprobs: Tensor,
        entropy: Tensor,
        newvalue: Tensor,
        importance_sampling_ratio: Tensor,
        adv: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute PPO losses for policy and value functions."""
        # Policy loss
        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(
            importance_sampling_ratio, 1 - self.trainer_cfg.ppo.clip_coef, 1 + self.trainer_cfg.ppo.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
        if self.trainer_cfg.ppo.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
            vf_clip_coef = self.trainer_cfg.ppo.vf_clip_coef
            v_clipped = minibatch["values"] + torch.clamp(
                newvalue_reshaped - minibatch["values"],
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprobs - minibatch["logprobs"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.trainer_cfg.ppo.clip_coef).float().mean()

        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


class Contrastive(Loss):
    def __init__(
        self,
        policy: DistributedMettaAgent,
        experience: Optional[Experience],
        trainer_cfg: Any,
        device: torch.device,
        losses: Losses,
    ):
        super().__init__(policy, trainer_cfg, device, losses)
        self.experience = experience

    def get_experience_spec(self) -> TensorDict:
        pass

    def __call__(
        self,
        minibatch: TensorDict,
        train_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
        kickstarter: Any,
        agent_step: int,
    ) -> Tensor:
        neg_example = self.experience.buffer["obs"][indices - 1]
        td = self.experience.buffer[indices]["obs"]

        new_td = self.policy.components["pred_output"](td)
        pred = new_td["pred"]
        # zero the last element in pred
        pred[:, -1] = 0

        # --- sample positive example from the buffer ---
        pos_example = train_td["_core_"].detach()
        # shift pos_example by 1
        pos_example = pos_example[:, 1:, :]
        padding = torch.zeros(
            pos_example.shape[0], 1, pos_example.shape[2], device=pos_example.device, dtype=pos_example.dtype
        )
        pos_example = torch.cat([pos_example, padding], dim=1)

        # --- sample negative example from the buffer ---
        neg_example = train_td["_core_"].detach()
        indices = torch.randperm(neg_example.shape[0])
        neg_example = neg_example[indices]
        neg_example = neg_example[:, 1:, :]
        padding = torch.zeros(
            neg_example.shape[0], 1, neg_example.shape[2], device=neg_example.device, dtype=neg_example.dtype
        )
        neg_example = torch.cat([neg_example, padding], dim=1)

        # Hinge loss for contrastive learning
        # TODO: make margin configurable
        margin = 1.0
        loss_pos = F.cross_entropy(pred, pos_example)
        loss_neg = F.cross_entropy(pred, neg_example)
        loss = loss_pos + F.relu(margin - loss_neg)

        return loss


class ValueDetached(Loss):
    def __init__(
        self,
        policy: DistributedMettaAgent,
        experience: Optional[Experience],
        trainer_cfg: Any,
        device: torch.device,
        losses: Losses,
    ):
        super().__init__(policy, trainer_cfg, device, losses)
        self.experience = experience
        self.detached_component = "_core_"
        self.intermediate_components = self.find_components_recursively(self.detached_component, "pred_output")

    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {"values": torch.zeros((), dtype=torch.float32)},
            batch_size=[],
        )

    def __call__(
        self,
        minibatch: TensorDict,
        train_td: TensorDict,
        indices: Tensor,
    ) -> Tensor:
        # delete the intermediate components from the train_td
        for component in self.intermediate_components:
            del train_td[component]

        train_td[self.detached_component] = train_td[self.detached_component].detach()
        train_td = self.policy(train_td)

        newvalue = train_td["values"]
        newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
        if self.trainer_cfg.ppo.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
            vf_clip_coef = self.trainer_cfg.ppo.vf_clip_coef
            v_clipped = minibatch["values"] + torch.clamp(
                newvalue_reshaped - minibatch["values"],
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape)},
            batch_size=minibatch.batch_size,
        )
        self.experience.update(indices, update_td)

        return v_loss
