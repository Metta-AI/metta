import typing

import einops
import torch
import torch.nn.functional as F
import pydantic
import tensordict

import metta.agent.policy
import metta.rl.checkpoint_manager
import metta.rl.loss.loss
import metta.rl.trainer_config
import metta.rl.training


class TLKickstarterConfig(metta.rl.loss.loss.LossConfig):
    teacher_uri: str = pydantic.Field(default="")
    action_loss_coef: float = pydantic.Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = pydantic.Field(default=1.0, ge=0, le=1.0)
    temperature: float = pydantic.Field(default=2.0, gt=0)

    def create(
        self,
        policy: metta.agent.policy.Policy,
        trainer_cfg: metta.rl.trainer_config.TrainerConfig,
        vec_env: typing.Any,
        device: torch.device,
        instance_name: str,
        loss_config: typing.Any,
    ) -> "TLKickstarter":
        """Create TLKickstarter loss instance."""
        return TLKickstarter(policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config)


class TLKickstarter(metta.rl.loss.loss.Loss):
    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "temperature",
        "teacher_policy_spec",
    )

    def __init__(
        self,
        policy: metta.agent.policy.Policy,
        trainer_cfg: metta.rl.trainer_config.TrainerConfig,
        vec_env: typing.Any,
        device: torch.device,
        instance_name: str,
        loss_config: typing.Any = None,
    ):
        # Get loss config from trainer_cfg if not provided
        if loss_config is None:
            loss_config = getattr(trainer_cfg.losses, instance_name, None)
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.action_loss_coef = self.cfg.action_loss_coef
        self.value_loss_coef = self.cfg.value_loss_coef
        self.temperature = self.cfg.temperature
        game_rules = getattr(self.env, "game_rules", getattr(self.env, "meta_data", None))
        if game_rules is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        self.teacher_policy = metta.rl.checkpoint_manager.CheckpointManager.load_from_uri(self.cfg.teacher_uri, game_rules, self.device)

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

        # get the teacher policy experience spec
        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

    def run_train(
        self,
        shared_loss_data: tensordict.TensorDict,
        context: metta.rl.training.ComponentContext,
        mb_idx: int,
    ) -> tuple[torch.Tensor, tensordict.TensorDict, bool]:
        policy_td = shared_loss_data["policy_td"]

        # Teacher forward pass
        teacher_td = policy_td.select(*self.teacher_policy_spec.keys(include_nested=True)).clone()
        teacher_td = self.teacher_policy(teacher_td, action=None)

        # Student forward pass
        student_td = policy_td.select(*self.policy_experience_spec.keys(include_nested=True)).clone()
        student_td = self.policy(student_td, action=None)

        # action loss
        temperature = self.temperature
        teacher_logits = teacher_td["logits"].to(dtype=torch.float32)
        student_logits = student_td["logits"].to(dtype=torch.float32)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_probs = torch.exp(student_log_probs)
        ks_action_loss = (temperature**2) * (
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        )

        # value loss
        teacher_value = teacher_td["values"].to(dtype=torch.float32).detach()
        student_value = student_td["values"].to(dtype=torch.float32)
        teacher_value = einops.rearrange(teacher_value, "b t 1 -> b (t 1)")
        student_value = einops.rearrange(student_value, "b t 1 -> b (t 1)")
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.action_loss_coef + ks_value_loss * self.value_loss_coef

        self.loss_tracker["tl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["tl_ks_value_loss"].append(float(ks_value_loss.item()))

        return loss, shared_loss_data, False
