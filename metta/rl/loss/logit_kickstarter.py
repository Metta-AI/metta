from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.teacher_policy import load_teacher_policy
from metta.rl.training import ComponentContext

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class LogitKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)
    teacher_led_proportion: float = Field(default=1.0, ge=0, le=1.0)  # at 0.0, it's purely student-led

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "LogitKickstarter":
        """Create LogitKickstarter loss instance."""
        return LogitKickstarter(policy, trainer_cfg, vec_env, device, instance_name, self)


class LogitKickstarter(Loss):
    """This also injects the teacher's logits into the student's observations."""

    __slots__ = (
        "teacher_policy",
        "extended_policy_env_info",
        "logit_feature_ids",
        "num_actions",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "LogitKickstarterConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)

        # Determine action space size
        act_space = self.env.single_action_space
        self.num_actions = int(act_space.n)
<<<<<<< HEAD
=======

>>>>>>> 34f6ccf001 (Centralize teacher policy loading for kickstarter losses (#4500))
        self.teacher_policy = load_teacher_policy(self.env, policy_uri=self.cfg.teacher_uri, device=self.device)

    def get_experience_spec(self) -> Composite:
        # Get action space size for logits shape
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        logits_f32 = UnboundedContinuous(shape=torch.Size([self.num_actions]), dtype=torch.float32)

        return Composite(
            teacher_logits=logits_f32,
            teacher_values=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            teacher_td = td.clone()
            self.teacher_policy.forward(teacher_td)
            teacher_actions = teacher_td["actions"]
            td["teacher_logits"] = teacher_td["logits"]
            td["teacher_values"] = teacher_td["values"]

            self.policy.forward(td)

        # Store experience
        env_slice = self._training_env_id(context)
        self.replay.store(data_td=td, env_id=env_slice)

        if torch.rand(1) < self.cfg.teacher_led_proportion:
            # overwrite student actions w teacher actions with some probability. anneal this.
            td["actions"] = teacher_actions

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"logits", "values"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        B, TT = minibatch.batch_size

        student_td = shared_loss_data["policy_td"].reshape(B * TT)  # we should do this without reshaping

        # action loss
        temperature = self.cfg.temperature
        teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).reshape(B * TT, -1).detach()
        student_logits = student_td["logits"].to(dtype=torch.float32)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_probs = torch.exp(student_log_probs)
        ks_action_loss = (temperature**2) * (
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        )

        # value loss
        teacher_value = minibatch["teacher_values"].to(dtype=torch.float32).reshape(B * TT).detach()
        student_value = student_td["values"].to(dtype=torch.float32)
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

        self.loss_tracker["ks_act_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["ks_val_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["ks_act_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["ks_val_loss_coef"].append(float(self.cfg.value_loss_coef))

        return loss, shared_loss_data, False
