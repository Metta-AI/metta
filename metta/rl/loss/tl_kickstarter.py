from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

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

        # load teacher policy
        policy_record = self.policy_store.policy_record(self.policy_cfg.losses.TLKickstarter.teacher_uri)
        self.teacher_policy: PolicyAgent = policy_record.policy
        if hasattr(self.teacher_policy, "initialize_to_environment"):
            features = self.vec_env.driver_env.get_observation_features()
            self.teacher_policy.initialize_to_environment(
                features, self.vec_env.driver_env.action_names, self.vec_env.driver_env.max_action_args, self.device
            )

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

    def get_experience_spec(self) -> Composite:
        return self.teacher_policy_spec

    def run_rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        self.teacher_policy(td)

    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        if not self.should_run_train(trainer_state.agent_step):
            return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data

        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        teacher_value = shared_loss_data["sampled_mb"]["values"]
        teacher_normalized_logits = shared_loss_data["sampled_mb"]["full_log_probs"]

        student_normalized_logits = shared_loss_data["policy_td"]["full_log_probs"]
        student_value = shared_loss_data["policy_td"]["value"]

        ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
        ks_action_loss *= self.action_loss_coef

        ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * self.value_loss_coef

        loss = ks_action_loss + ks_value_loss

        return loss, shared_loss_data
