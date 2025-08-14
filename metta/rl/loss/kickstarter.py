from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import Composite

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.rl.loss.base_loss import BaseLoss
from metta.rl.loss.loss_tracker import LossTracker
from metta.rl.trainer_config import TrainerConfig


class TLKickstarter(BaseLoss):
    """Teacher-led kickstarter."""

    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "begin_at_step",
        "end_at_step",
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

    def rollout(self, td: TensorDict) -> None:
        self.teacher_policy(td)

    # need train()
