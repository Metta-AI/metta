from typing import Any

import einops
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.loss_config import LossSchedule
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import ComponentContext
from mettagrid.base_config import Config


class SLKickstarterConfig(Config):
    schedule: LossSchedule | None = None
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    anneal_ratio: float = Field(default=0.995, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create SLKickstarter loss instance."""
        return SLKickstarter(policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config)


class SLKickstarter(Loss):
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
        self.anneal_ratio = self.loss_cfg.anneal_ratio

        # load teacher policy
        from metta.rl.checkpoint_manager import CheckpointManager

        game_rules = getattr(self.env, "game_rules", getattr(self.env, "meta_data", None))
        if game_rules is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        self.teacher_policy = CheckpointManager.load_from_uri(self.loss_cfg.teacher_uri, game_rules, self.device)

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

        # get the teacher policy experience spec
        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

        # Calculate annealing schedule
        self.anneal_duration = self.trainer_cfg.total_timesteps // self.trainer_cfg.update_epochs
        # Start ramping down at 50% of training
        self.ramp_down_start_epochs = int(0.50 * self.anneal_duration)

        # Pre-compute the annealing factor
        # We want to reach 1e-3 of the original value by the end
        final_value = 0.001
        self.anneal_factor = final_value ** (1.0 / (self.anneal_duration - self.ramp_down_start_epochs))

    def get_experience_spec(self) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=(), dtype=torch.float32)

        return Composite(
            # kickstarter loss data
            teacher_action=UnboundedContinuous(
                shape=(int(self.teacher_policy_spec["action"].shape[0]),), dtype=torch.int32
            ),
            teacher_value=scalar_f32,
        )

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
        teacher_action = teacher_td["action"].to(dtype=torch.int32).detach()
        teacher_value = teacher_td["values"].to(dtype=torch.float32).detach()

        # Student forward pass
        student_td = policy_td.select(*self.policy_experience_spec.keys(include_nested=True)).clone()
        student_td = self.policy(student_td, action=None)
        student_action = student_td["action"].to(dtype=torch.int32)
        student_value = student_td["values"].to(dtype=torch.float32)

        # Calculate annealing coefficient
        update_epoch = getattr(context, "update_epoch", context.epoch)
        if update_epoch < self.ramp_down_start_epochs:
            anneal_coef = self.action_loss_coef  # Full strength
        else:
            # Exponential decay after ramp_down_start
            epochs_since_ramp = update_epoch - self.ramp_down_start_epochs
            anneal_coef = self.action_loss_coef * (self.anneal_factor**epochs_since_ramp)

        # Action loss (only for matching actions)
        matching_actions = (teacher_action == student_action).float()
        ks_action_loss = (1.0 - matching_actions).mean() * anneal_coef

        # Value loss
        teacher_value = einops.rearrange(teacher_value, "b t 1 -> b (t 1)")
        student_value = einops.rearrange(student_value, "b t 1 -> b (t 1)")
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean() * self.value_loss_coef

        # Clamp losses to avoid negative values
        ks_action_loss = torch.clamp(ks_action_loss, min=0.0)  # av this should never go negative yet it seems to!!!
        ks_value_loss = torch.clamp(ks_value_loss, min=-0.001)

        loss = ks_action_loss + ks_value_loss

        self.loss_tracker["sl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["sl_ks_value_loss"].append(float(ks_value_loss.item()))

        return loss, shared_loss_data, False
