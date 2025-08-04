"""Loss computation functions for PPO training."""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.rl.experience import Experience
from metta.rl.trainer import MettaTrainer
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.util.batch_utils import calculate_prioritized_sampling_params

logger = logging.getLogger(__name__)

'''----Updates to Policy in metta_agent.py:----
Policy gets one experience buffer and a list of loss objects. It also needs a method of updating losses when loaded.
policy.experience = Experience()
policy.losses: List[BaseLoss] = []

Upon loading, the policy needs to re-instantiate the losses. Policy method:
    def setup_losses(self, loss_configs: List[Dict], env: Any = None):
        """
        Initializes loss functions after the policy has been created.
        This allows passing in external dependencies like the environment.
        """
        print(f"\n--- Setting up losses for policy '{self.id}' ---")
        self.losses = [] # Clear any existing losses if re-initializing
        for loss_config in loss_configs:
            LossClass = loss_config.pop('class')
            self.losses.append(LossClass(policy=self, cfg=loss_config, device=self.device))

    def update_experience_spec(self, experience_spec: TensorDict):
        for loss in self.losses:
            spec = loss.get_experience_spec()
            if spec:
                self.experience.spec = self.experience.spec.merge(spec)
#----End updates to Policy in metta_agent.py:----


# ====Training loop in new file metta_loop.py====
# ----initialization: most is nicely abstracted by trainer.py, not here----
import MettaTrainer

# initialize the policies which inits the trainers and buffers (one per policy) and losses (one or many per policy)
for policy in policies:
    policy.trainer = MettaTrainer(policy, cfg, device) # this class has the optimizer, roll_out and train methods,
    # infra configs for experience preallocation, stats management, PolicyStore, etc.
    experience_spec = policy.get_agent_experience_spec()
    for loss in policy.losses:
        self.losses.append(LossClass(policy=self, cfg=config_copy, device=self.device, trainer=self.trainer))
        experience_spec.merge(loss.get_experience_spec())
    policy.experience = Experience(experience_spec, policy.trainer) # one buffer combining needs of all losses and agent
# ----End initialization:----

# ----Training loop is simply:----
while Curriculum.should_run():
    for policy in policies:
        policy.trainer.epoch += 1
        for loss in policy.losses:
            policy.loss.roll_out()
        for mb in range(policy.trainer.trainer_cfg.batch_size):
            shared_training_td = TensorDict({}, batch_size=[]).mb = mb # empty tensordict with batch size of batch_size
            for loss in policy.losses:
                losses, shared_training_td = policy.loss.train(shared_training_td)
            policy.trainer.optimize(losses)
        policy.trainer.on_epoch_end()
    Curriculum.update()
Curriculum.step() # update envs, policy losses, etc. according to curriculum schedule
# ----End training loop in new file metta_loop.py:----
'''


class BaseLoss(ABC):
    """
    The Loss class acts as a manager for different loss computations.

    It is initialized with the shared trainer state (policy, config, device, etc.)
    and dynamically instantiates the required loss components (e.g., PPO, Contrastive)
    based on the configuration. Each component holds a reference to this manager
    to access the shared state, favoring composition over inheritance.
    """

    def __init__(
        self,
        policy: MettaAgent | DistributedMettaAgent,
        cfg: Any,
        device: torch.device,
        trainer: MettaTrainer,
    ):
        self.policy = policy
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.cfg = cfg
        self.device = device
        self.trainer = trainer
        self.loss_tracker = trainer.losses

    @abstractmethod
    def get_experience_spec(self) -> TensorDict:
        pass

    def roll_out(self) -> None:
        """Uses trainer to work with the env to generate experience."""
        while not self.policy.experience.full:
            # expecting trainer.roll_out() to get obs from env, run policy, get td from policy, populate with env attr
            # like rewards, dones, truncateds, etc., then for policy.experience to take what it wants from the td
            self.policy.trainer.roll_out()

    @abstractmethod
    def train(self, shared_loss_data: TensorDict) -> tuple[Tensor, TensorDict]:
        """This is primarily computing loss and feeding to the optimizer."""
        pass

    # helper method for losses that wish to detach grads from tensors at various components in the policy
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


# ----Example EMA for BYOL loss class:----
class EMA(BaseLoss):
    def __init__(
        self, policy: MettaAgent | DistributedMettaAgent, cfg: Any, device: torch.device, trainer: MettaTrainer
    ):
        super().__init__(policy, cfg, device, trainer)
        self.target_model = copy.deepcopy(self.policy)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.ema_decay = cfg.get("ema_decay", 0.996)

    def update_target_model(self):
        """Update target model with exponential moving average"""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.policy.parameters(), strict=False
            ):
                target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def train(self, shared_loss_data: TensorDict) -> tuple[Tensor, TensorDict]:
        self.update_target_model()
        policy_td = shared_loss_data["PPO"].select(*self.policy_experience_spec.keys(include_nested=True))
        pred = self.policy.components["pred_output"](policy_td)  # here, we run this head on our own
        with torch.no_grad():
            target_pred = self.target_model.components["pred_output"](policy_td)
            target_pred = target_pred.detach()
        shared_loss_data["BYOL"]["target_pred"] = target_pred  # add these in case other losses want them next
        shared_loss_data["BYOL"]["pred"] = pred
        loss = F.mse_loss(pred, target_pred) * self.cfg.byol_loss_coef
        return loss, shared_loss_data


# ----Teacher-led kickstarter ----
class TeacherLedKickstarter(BaseLoss):
    def __init__(
        self, policy: MettaAgent | DistributedMettaAgent, cfg: Any, device: torch.device, trainer: MettaTrainer
    ):
        super().__init__(policy, cfg, device, trainer)
        self.teacher_policy = self.policy.trainer.policy_store.get_policy(self.cfg.kickstarter.teacher_URI)
        self.teacher_policy_experience_spec = self.teacher_policy.get_agent_experience_spec()
        self.teacher_policy.experience = Experience(self.teacher_policy_experience_spec, self.teacher_policy.trainer)

    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {
                "env_obs": torch.zeros((), dtype=torch.float32),
                "full_logprobs": torch.zeros((), dtype=torch.float32),
            },
            batch_size=[],
        )

    def roll_out(self) -> None:
        self.trainer.policy = self.teacher_policy
        while not self.teacher_policy.experience.full:
            self.trainer.roll_out()
        self.trainer.policy = self.policy

    def train(self, shared_loss_data: TensorDict) -> tuple[Tensor, TensorDict]:
        if self.trainer.epoch >= self.cfg.s_l_kickstarter.start_epoch:
            # runs only for a few epochs, then switches to student-led
            return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data

        teacher_obs = self.teacher_policy.experience["env_obs"]
        teacher_full_logprobs = self.teacher_policy.experience["full_logprobs"]
        student_td = self.teacher_policy.experience.buffer.select(
            *self.policy_experience_spec.keys(include_nested=True)
        )
        student_td = self.policy(student_td)
        student_full_logprobs = student_td["full_logprobs"]
        loss = F.cross_entropy(teacher_full_logprobs, student_full_logprobs)
        self.policy.experience["teacher_obs"] = teacher_obs  # adding just in case other losses want them next
        self.policy.experience["teacher_full_logprobs"] = teacher_full_logprobs  # in case other losses want them next
        return loss, shared_loss_data


# ----Student-led kickstarter SO MUCH SMALLER THAN BEFORE!!! ----
class StudentLedKickstarter(BaseLoss):
    def __init__(
        self, policy: MettaAgent | DistributedMettaAgent, cfg: Any, device: torch.device, trainer: MettaTrainer
    ):
        super().__init__(policy, cfg, device, trainer)
        self.teacher_policy = self.policy.trainer.policy_store.get_policy(self.cfg.kickstarter.teacher_URI)
        self.teacher_policy_experience_spec = self.teacher_policy.get_agent_experience_spec()
        self.teacher_policy.experience = Experience(self.teacher_policy_experience_spec, self.teacher_policy.trainer)

    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {"full_logprobs": torch.zeros((), dtype=torch.float32)},
            batch_size=[],
        )

    def train(self, shared_loss_data: TensorDict) -> tuple[Tensor, TensorDict]:
        if self.trainer.epoch < self.cfg.s_l_kickstarter.start_epoch:
            # runs only after some desired number of epochs. Easy to control and very legible.
            return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data

        teacher_policy_td = self.policy.experience.buffer.select(*self.policy_experience_spec.keys(include_nested=True))
        with torch.no_grad():
            teacher_policy_td = self.teacher_policy(teacher_policy_td)
        teacher_full_logprobs = teacher_policy_td["full_logprobs"]
        student_full_logprobs = self.policy.experience["full_logprobs"]
        loss = F.cross_entropy(teacher_full_logprobs, student_full_logprobs)
        self.policy.experience["teacher_full_logprobs"] = (
            teacher_full_logprobs  # add these in case other losses want them next
        )
        return loss, shared_loss_data


class PPO(BaseLoss):
    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {
                "rewards": torch.zeros((), dtype=torch.float32),
                "dones": torch.zeros((), dtype=torch.float32),
                "truncateds": torch.zeros((), dtype=torch.float32),
                "actions": torch.zeros(self.manager.policy.agent_attributes["action_space"].shape, dtype=torch.int32),
                "logprobs": torch.zeros((), dtype=torch.float32),
                "values": torch.zeros((), dtype=torch.float32),
            },
            batch_size=[],
        )

    def train(
        self,
        shared_loss_data: TensorDict,
        epoch: int,
        mb_idx: int,
    ) -> Tensor:
        """Process a single minibatch update and return the total loss."""
        # The policy's training forward pass returns a TD with required tensors for loss calculation.

        if self.manager.mb_idx == 0:
            self.anneal_beta = calculate_prioritized_sampling_params(
                epoch=self.manager.epoch,
                total_timesteps=self.manager.trainer_cfg.total_timesteps,
                batch_size=self.manager.trainer_cfg.batch_size,
                prio_alpha=self.manager.trainer_cfg.prioritized_experience_replay.prio_alpha,
                prio_beta0=self.manager.trainer_cfg.prioritized_experience_replay.prio_beta0,
            )
            self.prio_weights = self.anneal_beta["weights"]
            self.prio_beta = self.anneal_beta["beta"]

            advantages = torch.zeros_like(self.manager.experience.buffer["values"])
            initial_importance_sampling_ratio = torch.ones_like(self.manager.experience.buffer["values"])
            advantages = compute_advantage(
                self.manager.experience.buffer["values"],
                self.manager.experience.buffer["rewards"],
                self.manager.experience.buffer["dones"],
                initial_importance_sampling_ratio,
                advantages,
                self.manager.trainer_cfg.ppo.gamma,
                self.manager.trainer_cfg.ppo.gae_lambda,
                self.manager.trainer_cfg.vtrace.vtrace_rho_clip,
                self.manager.trainer_cfg.vtrace.vtrace_c_clip,
                self.manager.device,
            )

        minibatch, indices, prio_weights = self.manager.experience.sample_minibatch(
            advantages=advantages,
            prio_alpha=self.manager.trainer_cfg.prioritized_experience_replay.prio_alpha,
            prio_beta=self.prio_beta,
        )

        # select what policy wants from the minibatch and pass it in the forward
        self.manager.train_td = minibatch.select(*self.manager.policy_spec.keys(include_nested=True))
        self.manager.train_td.meta["indices"] = indices  # add indices since we buffer envs
        self.manager.train_td = self.manager.policy(self.manager.train_td, action=minibatch["actions"])
        # design choice: save the policies td to the manager level so other losses can use the same output if they like.
        # if not then they can sample their own from experience and run the policy forward again or just other leafs.

        new_logprobs = self.manager.train_td["action_log_prob"].reshape(minibatch["logprobs"].shape)
        entropy = self.manager.train_td["entropy"]
        newvalue = self.manager.train_td["value"]

        logratio = new_logprobs - minibatch["logprobs"]
        importance_sampling_ratio = logratio.exp()

        # Re-compute advantages with new ratios (V-trace)
        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            self.manager.trainer_cfg.ppo.gamma,
            self.manager.trainer_cfg.ppo.gae_lambda,
            self.manager.trainer_cfg.vtrace.vtrace_rho_clip,
            self.manager.trainer_cfg.vtrace.vtrace_c_clip,
            self.manager.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, self.manager.trainer_cfg.ppo.norm_adv)
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

        # # Kickstarter losses
        # ks_action_loss, ks_value_loss = kickstarter.loss(
        #     agent_step,
        #     full_logprobs,
        #     newvalue,
        #     minibatch["env_obs"],
        #     teacher_lstm_state=[],
        # )

        # Total loss
        loss = (
            pg_loss
            - self.manager.trainer_cfg.ppo.ent_coef * entropy_loss
            + v_loss * self.manager.trainer_cfg.ppo.vf_coef
        )

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape), "ratio": importance_sampling_ratio},
            batch_size=minibatch.batch_size,
        )
        if self.manager.experience:
            self.manager.experience.update(indices, update_td)

        # Update loss tracking
        self.manager.loss_tracker.policy_loss_sum += pg_loss.item()
        self.manager.loss_tracker.value_loss_sum += v_loss.item()
        self.manager.loss_tracker.entropy_sum += entropy_loss.item()
        self.manager.loss_tracker.approx_kl_sum += approx_kl.item()
        self.manager.loss_tracker.clipfrac_sum += clipfrac.item()
        self.manager.loss_tracker.importance_sum += importance_sampling_ratio.mean().item()
        self.manager.loss_tracker.minibatches_processed += 1
        self.manager.loss_tracker.current_logprobs_sum += new_logprobs.mean().item()

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
            importance_sampling_ratio,
            1 - self.manager.trainer_cfg.ppo.clip_coef,
            1 + self.manager.trainer_cfg.ppo.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
        if self.manager.trainer_cfg.ppo.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
            vf_clip_coef = self.manager.trainer_cfg.ppo.vf_clip_coef
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
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.manager.trainer_cfg.ppo.clip_coef).float().mean()

        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


class Contrastive(BaseLoss):
    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {"_core_": torch.zeros((), dtype=torch.float32)},
            batch_size=[],
        )

    def train(self, shared_loss_data: TensorDict) -> tuple[Tensor, TensorDict]:
        policy_td = shared_loss_data["PPO"].select(*self.policy_experience_spec.keys(include_nested=True))
        new_td = self.policy.components["pred_output"](policy_td)  # run just the pred_output head on our own
        pred = new_td["pred_output"]
        # zero the last element in pred
        pred[:, -1] = 0

        # --- sample positive example from the buffer ---
        pos_example = shared_loss_data["PPO"]["_core_"].detach()
        # shift pos_example by 1
        pos_example = pos_example[:, 1:, :]
        padding = torch.zeros(
            pos_example.shape[0], 1, pos_example.shape[2], device=pos_example.device, dtype=pos_example.dtype
        )
        pos_example = torch.cat([pos_example, padding], dim=1)

        # --- Feature: sample negative example from the entire buffer, not just the minibatch ---
        total_buffer_size = self.policy.experience["_core_"].shape[0]
        minibatch_indices = shared_loss_data["PPO"].indices
        minibatch_size = pos_example.shape[0]

        # Create a mask to exclude indices near the minibatch indices for temporal distance
        exclude_mask = torch.zeros(total_buffer_size, dtype=torch.bool, device=self.device)
        window_size = 10  # temporal window to exclude around minibatch indices
        for idx in minibatch_indices:
            start_idx = max(0, idx - window_size)
            end_idx = min(total_buffer_size, idx + window_size + 1)
            exclude_mask[start_idx:end_idx] = True

        # Get available indices (those not excluded)
        available_indices = torch.where(~exclude_mask)[0]

        # Randomly sample minibatch_size indices from available indices
        if len(available_indices) >= minibatch_size:
            neg_indices = available_indices[torch.randperm(len(available_indices))[:minibatch_size]]
        else:
            # Fallback: if not enough available indices, use random indices from full buffer
            neg_indices = torch.randint(0, total_buffer_size, (minibatch_size,), device=self.device)

        neg_example = self.policy.experience["_core_"][neg_indices]
        neg_example = neg_example[:, 1:, :]
        padding = torch.zeros(
            neg_example.shape[0], 1, neg_example.shape[2], device=neg_example.device, dtype=neg_example.dtype
        )
        neg_example = torch.cat([neg_example, padding], dim=1)

        # Hinge loss for contrastive learning
        loss_pos = F.cross_entropy(pred, pos_example)
        loss_neg = F.cross_entropy(pred, neg_example)
        loss = loss_pos + F.relu(self.cfg.contrastive.margin - loss_neg)

        return loss, shared_loss_data


class ValueDetached:
    def __init__(self, manager: "Loss"):
        # this class can now store experience for longer than an epoch
        self.v_experience = Experience()  # need to feed in epoch length etc.

    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {"values": torch.zeros((), dtype=torch.float32)},  # don't worry, Experience will catch that this is a dup
            batch_size=[],
        )

    def __call__(
        self,
        minibatch: TensorDict,
        train_td: TensorDict,
        indices: Tensor,
    ) -> Tensor:
        # feature: this loss can run every n epochs if it likes
        if self.trainer.epoch % self.trainer.trainer_cfg.contrastive.update_every_n_epochs != 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        # deletes the intermediate components from the train_td
        for component in self.intermediate_components:
            del train_td[component]

        train_td[self.detached_component] = train_td[self.detached_component].detach()
        train_td = self.policy(train_td)

        newvalue = train_td["values"]
        newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
        if self.cfg.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
            vf_clip_coef = self.cfg.vf_clip_coef
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
        if self.policy.experience:
            self.policy.experience.update(indices, update_td)

        self.policy.trainer.optimize()

        return
