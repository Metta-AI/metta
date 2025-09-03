from typing import Any

import einops
import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.base_loss import BaseLoss
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
        checkpoint_manager: CheckpointManager,
        instance_name: str,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, checkpoint_manager, instance_name)
        self.action_loss_coef = self.loss_cfg.action_loss_coef
        self.value_loss_coef = self.loss_cfg.value_loss_coef

        # load teacher policy
        self.teacher_policy: PolicyAgent = CheckpointManager.load_from_uri(self.loss_cfg.teacher_uri, device)
        if hasattr(self.teacher_policy, "initialize_to_environment"):
            features = self.vec_env.driver_env.get_observation_features()
            self.teacher_policy.initialize_to_environment(
                features, self.vec_env.driver_env.action_names, self.vec_env.driver_env.max_action_args, self.device
            )

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

    def get_experience_spec(self) -> Composite:
        return self.teacher_policy_spec

    def on_rollout_start(self, trainer_state: TrainerState) -> None:
        self.teacher_policy.on_rollout_start()
        return

    def run_rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        with torch.no_grad():
            self.teacher_policy(td)

        self.replay.store(data_td=td, env_id=trainer_state.training_env_id)

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        teacher_value = shared_loss_data["sampled_mb"]["values"].detach()
        teacher_normalized_logits = shared_loss_data["sampled_mb"]["full_log_probs"].detach()

        student_normalized_logits = shared_loss_data["policy_td"]["full_log_probs"]
        student_value = shared_loss_data["policy_td"]["value"]

        # av - counterintuitive that forward KL works - test reverse KL
        ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
        ks_action_loss *= self.action_loss_coef

        student_value = einops.rearrange(student_value, "b t 1 -> b (t 1)")
        ks_value_loss += ((teacher_value.detach() - student_value) ** 2).mean() * self.value_loss_coef

        loss = ks_action_loss + ks_value_loss

        self.loss_tracker["tl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["tl_ks_value_loss"].append(float(ks_value_loss.item()))

        return loss, shared_loss_data
