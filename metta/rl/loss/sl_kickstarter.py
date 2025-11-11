from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SLKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)
    student_forward: bool = Field(default=False)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "SLKickstarter":
        """Create SLKickstarter loss instance."""
        return SLKickstarter(policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config)


class SLKickstarter(Loss):
    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "temperature",
        "teacher_policy_spec",
        "student_forward",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any = None,
    ) -> None:
        # Get loss config from trainer_cfg if not provided
        if loss_config is None:
            loss_config = getattr(trainer_cfg.losses, instance_name, None)
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.action_loss_coef = self.cfg.action_loss_coef
        self.value_loss_coef = self.cfg.value_loss_coef
        self.temperature = self.cfg.temperature
        self.student_forward = self.cfg.student_forward

        game_rules = getattr(self.env, "policy_env_info", None)
        if game_rules is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        # Lazy import to avoid circular dependency
        from metta.rl.checkpoint_manager import CheckpointManager

        self.teacher_policy = CheckpointManager.load_from_uri(self.cfg.teacher_uri, game_rules, self.device)

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

    def get_experience_spec(self) -> Composite:
        return self.teacher_policy_spec

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]

        # Teacher forward pass
        teacher_td = minibatch.select(*self.teacher_policy_spec.keys(include_nested=True)).clone()
        B, TT = teacher_td.batch_size
        teacher_td = teacher_td.reshape(B * TT)
        teacher_td.set("bptt", torch.full((B * TT,), TT, device=teacher_td.device, dtype=torch.long))
        teacher_td.set("batch", torch.full((B * TT,), B, device=teacher_td.device, dtype=torch.long))
        teacher_td = self.teacher_policy(teacher_td, action=None)

        # Student forward pass
        if self.student_forward:
            student_td = minibatch.select(*self.policy.get_agent_experience_spec().keys(include_nested=True)).clone()
            B, TT = student_td.batch_size
            student_td = student_td.reshape(B * TT)
            student_td.set("bptt", torch.full((B * TT,), TT, device=student_td.device, dtype=torch.long))
            student_td.set("batch", torch.full((B * TT,), B, device=student_td.device, dtype=torch.long))
            student_td = self.policy(student_td, action=None)
        else:
            student_td = shared_loss_data["policy_td"].reshape(B * TT)

        temperature = self.temperature
        teacher_logits = teacher_td["logits"].to(dtype=torch.float32)
        student_logits = student_td["logits"].to(dtype=torch.float32)

        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_probs = torch.exp(student_log_probs)

        ks_action_loss = (temperature**2) * (
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        )

        # Value loss
        student_value = student_td["values"].to(dtype=torch.float32)
        teacher_value = teacher_td["values"].to(dtype=torch.float32).detach()
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.action_loss_coef + ks_value_loss * self.value_loss_coef

        self.loss_tracker["sl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["sl_ks_value_loss"].append(float(ks_value_loss.item()))

        return loss, shared_loss_data, False
