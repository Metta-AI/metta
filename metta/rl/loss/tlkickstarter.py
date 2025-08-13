from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.rl.loss.base_loss import BaseLoss
from metta.rl.loss.loss_tracker import LossTracker
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState


class TLKickstarter(BaseLoss):
    """Teacher-led kickstarter."""

    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "begin_at_step",
        "end_at_step",
        "anneal_ratio",
        "anneal_duration",
        "ramp_down_start_step",
        "anneal_factor",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
        policy_store: PolicyStore,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, loss_tracker, policy_store)
        self.action_loss_coef = self.policy_cfg.losses.TLKickstarter.action_loss_coef
        self.value_loss_coef = self.policy_cfg.losses.TLKickstarter.value_loss_coef
        self.begin_at_step = self.policy_cfg.losses.TLKickstarter.begin_at_step
        self.end_at_step = self.policy_cfg.losses.TLKickstarter.end_at_step
        self.anneal_ratio = self.policy_cfg.losses.TLKickstarter.anneal_ratio

        # load teacher policy
        policy_record = self.policy_store.policy_record(self.policy_cfg.losses.TLKickstarter.teacher_uri)
        self.teacher_policy: PolicyAgent = policy_record.policy
        if hasattr(self.teacher_policy, "initialize_to_environment"):
            features = self.vec_env.driver_env.get_observation_features()
            self.teacher_policy.initialize_to_environment(
                features, self.vec_env.driver_env.action_names, self.vec_env.driver_env.max_action_args, self.device
            )

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

        self.anneal_factor = 1.0

        kickstart_steps = self.end_at_step - self.begin_at_step

        if self.anneal_ratio > 0:
            self.anneal_duration = kickstart_steps * self.anneal_ratio
            self.ramp_down_start_step = kickstart_steps - self.anneal_duration
        else:
            self.anneal_duration = 0
            self.ramp_down_start_step = kickstart_steps

    def get_experience_spec(self) -> Composite:
        loss_spec = Composite(
            full_log_probs=UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32),
        )
        merged_spec_dict: dict = dict(self.teacher_policy_spec.items())
        merged_spec_dict.update(dict(loss_spec.items()))
        return Composite(merged_spec_dict)

    def losses_to_track(self) -> list[str]:
        return ["ks_action_loss", "ks_value_loss"]

    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        agent_step = trainer_state.agent_step
        if agent_step < self.begin_at_step or agent_step >= self.end_at_step:
            loss = ks_action_loss + ks_value_loss
            return loss, shared_loss_data

        if self.anneal_ratio > 0 and agent_step >= self.end_at_step:
            # Ramp down
            progress = (agent_step - self.ramp_down_start_step) / self.anneal_duration
            self.anneal_factor = 1.0 - progress

        td = shared_loss_data["sampled_mb"].select(self.teacher_policy_spec).clone()
        with torch.no_grad():
            self.teacher_policy(td)
        teacher_value = td["value"]
        teacher_normalized_logits = td["full_log_probs"]

        student_normalized_logits = shared_loss_data["policy_td"]["full_log_probs"]
        student_value = shared_loss_data["policy_td"]["value"]

        ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
        ks_action_loss *= self.action_loss_coef * self.anneal_factor

        ks_value_loss += (
            ((teacher_value.squeeze() - student_value) ** 2).mean() * self.value_loss_coef * self.anneal_factor
        )

        loss = ks_action_loss + ks_value_loss

        return loss, shared_loss_data
