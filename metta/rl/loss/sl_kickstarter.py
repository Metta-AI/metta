from typing import Any

import einops
import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.base_loss import BaseLoss
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState


class SLKickstarter(BaseLoss):
    """Student-led kickstarter."""

    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "anneal_ratio",
        "anneal_duration",
        "ramp_down_start_epochs",
        "anneal_factor",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        instance_name: str,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, checkpoint_manager, instance_name)
        self.action_loss_coef = self.loss_cfg.action_loss_coef
        self.value_loss_coef = self.loss_cfg.value_loss_coef
        self.anneal_ratio = self.loss_cfg.anneal_ratio

        # load teacher policy
        self.teacher_policy: PolicyAgent = CheckpointManager.load_from_uri(self.loss_cfg.teacher_uri, device)
        if hasattr(self.teacher_policy, "initialize_to_environment"):
            features = self.vec_env.driver_env.get_observation_features()
            self.teacher_policy.initialize_to_environment(
                features, self.vec_env.driver_env.action_names, self.vec_env.driver_env.max_action_args, self.device
            )

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

        self.anneal_factor = 1.0

        kickstart_epochs = self.train_end_epoch - self.train_start_epoch

        if self.anneal_ratio > 0:
            self.anneal_duration = kickstart_epochs * self.anneal_ratio
            self.ramp_down_start_epochs = kickstart_epochs - self.anneal_duration
        else:
            self.anneal_duration = 0
            self.ramp_down_start_epochs = kickstart_epochs

    def get_experience_spec(self) -> Composite:
        if not hasattr(self.teacher_policy, "action_max_params"):
            raise ValueError(
                "SL Kickstarter cannot determine size of teacher logits. Teacher policy must have "
                "action_max_params attribute"
            )
        if not hasattr(self.policy, "action_max_params"):
            raise ValueError(
                "SL Kickstarter cannot determine size of student logits. Student policy must have "
                "action_max_params attribute"
            )

        num_teacher_params = self.teacher_policy.action_max_params
        num_teacher_actions = sum([x + 1 for x in num_teacher_params])
        num_student_actions = self.policy.action_max_params
        num_student_actions = sum([x + 1 for x in num_student_actions])
        assert num_teacher_actions == num_student_actions, "Teacher and student must have the same number of actions"

        loss_spec = Composite(
            full_log_probs=UnboundedContinuous(shape=torch.Size([num_student_actions]), dtype=torch.float32),
        )
        merged_spec_dict: dict = dict(self.teacher_policy_spec.items())
        merged_spec_dict.update(dict(loss_spec.items()))
        return Composite(merged_spec_dict)

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        current_epoch = trainer_state.epoch

        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if self.anneal_ratio > 0 and current_epoch > self.ramp_down_start_epochs:
            # Ramp down
            progress = (current_epoch - self.ramp_down_start_epochs) / self.anneal_duration
            self.anneal_factor = max(1.0 - progress, 0.0)

        td = shared_loss_data["sampled_mb"].select(*self.teacher_policy_spec.keys()).clone()

        self.teacher_policy.on_train_mb_start()
        with torch.no_grad():
            policy_out = self.teacher_policy(td)
        teacher_value = policy_out["values"]
        teacher_normalized_logits = policy_out["full_log_probs"]

        student_normalized_logits = shared_loss_data["policy_td"]["full_log_probs"]
        student_value = shared_loss_data["policy_td"]["value"]

        student_normalized_logits = einops.rearrange(student_normalized_logits, "b t l -> (b t) l")
        ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
        ks_action_loss *= self.action_loss_coef * self.anneal_factor

        ks_value_loss += (
            ((teacher_value.squeeze() - student_value) ** 2).mean() * self.value_loss_coef * self.anneal_factor
        )

        # clamp losses
        ks_action_loss = torch.clamp(ks_action_loss, min=0.0)  # av this should never go negative yet it seems to!!!
        ks_value_loss = torch.clamp(ks_value_loss, min=-0.001)

        loss = ks_action_loss + ks_value_loss

        self.loss_tracker["sl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["sl_ks_value_loss"].append(float(ks_value_loss.item()))

        return loss, shared_loss_data
