from typing import Any

import einops
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import ComponentContext
from mettagrid.base_config import Config


class TLKickstarterConfig(Config):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)

    def update_hypers(self, context: ComponentContext) -> None:
        for sched in self.schedule:
            sched.apply(obj=self, epoch=context.epoch, agent_step=context.agent_step)

    def create(
        self,
        policy: Policy,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create TLKickstarter loss instance."""
        return TLKickstarter(policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config)


class TLKickstarter(Loss):
    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any = None,
    ):
        # Get loss config from trainer_cfg if not provided
        if loss_config is None:
            loss_config = getattr(trainer_cfg.losses, instance_name, None)
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.action_loss_coef = self.loss_cfg.action_loss_coef
        self.value_loss_coef = self.loss_cfg.value_loss_coef

        # load teacher policy
        from metta.rl.checkpoint_manager import CheckpointManager

        self.teacher_policy: Policy = CheckpointManager.load_from_uri(self.loss_cfg.teacher_uri, device)
        if hasattr(self.teacher_policy, "initialize_to_environment"):
            driver_env = self.env.driver_env
            features = driver_env.observation_features
            self.teacher_policy.initialize_to_environment(
                features, driver_env.action_names, driver_env.max_action_args, self.device
            )

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

        # get the teacher policy experience spec
        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        policy_td = shared_loss_data["policy_td"]

        # Teacher forward pass
        teacher_td = policy_td.select(*self.teacher_policy_spec.keys(include_nested=True)).clone()
        teacher_td = self.teacher_policy(teacher_td, action=None)
        teacher_action_logits = teacher_td["action_logits"].to(dtype=torch.float32)
        teacher_value = teacher_td["values"].to(dtype=torch.float32)

        # Student forward pass
        student_td = policy_td.select(*self.policy_experience_spec.keys(include_nested=True)).clone()
        student_td = self.policy(student_td, action=None)
        student_action_logits = student_td["action_logits"].to(dtype=torch.float32)
        student_value = student_td["values"].to(dtype=torch.float32)

        # action loss
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss += ((teacher_action_logits.detach() - student_action_logits) ** 2).mean() * self.action_loss_coef

        # value loss
        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        teacher_value = einops.rearrange(teacher_value, "b t 1 -> b (t 1)")
        student_value = einops.rearrange(student_value, "b t 1 -> b (t 1)")
        ks_value_loss += ((teacher_value.detach() - student_value) ** 2).mean() * self.value_loss_coef

        loss = ks_action_loss + ks_value_loss

        self.loss_tracker["tl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["tl_ks_value_loss"].append(float(ks_value_loss.item()))

        return loss, shared_loss_data, False
