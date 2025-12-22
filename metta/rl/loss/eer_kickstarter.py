from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.mpt_policy import MptPolicy
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class EERKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    r_lambda: float = Field(default=0.1, ge=0)  # scale the teacher log likelihoods that are added to rewards

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "EERKickstarter":
        """Create EERKickstarter loss instance."""
        return EERKickstarter(policy, trainer_cfg, vec_env, device, instance_name, self)


class EERKickstarter(Loss):
    """Expected Entropy Regularization Kickstarter. See "Distilling Policy Distillation."
    Note that this needs to be tweaked if we are to use it with prio sampling. For now, use sequential sampling.
    """

    __slots__ = ("teacher_policy",)

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "EERKickstarterConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)

        policy_env_info = getattr(self.env, "policy_env_info", None)
        if policy_env_info is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")
        teacher_spec = policy_spec_from_uri(self.cfg.teacher_uri, device=str(self.device))
        self.teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec)
        if isinstance(self.teacher_policy, MptPolicy):
            self.teacher_policy = self.teacher_policy._policy

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        num_actions = act_space.n

        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        logits_f32 = UnboundedContinuous(shape=torch.Size([num_actions]), dtype=torch.float32)

        return Composite(
            teacher_full_log_probs=logits_f32,
            teacher_values=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            teacher_td = td.clone()
            self.teacher_policy.forward(teacher_td)
            td["teacher_full_log_probs"] = teacher_td["full_log_probs"]
            td["teacher_values"] = teacher_td["values"]

            self.policy.forward(td)

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"full_log_probs", "values"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        student_td = shared_loss_data["policy_td"]

        teacher_full_log_probs = minibatch["teacher_full_log_probs"]
        student_action = minibatch["actions"]

        teach_act_log_probs = teacher_full_log_probs.gather(dim=2, index=student_action.unsqueeze(2)).squeeze(2)
        dones = minibatch["dones"] > 0.5
        truncateds = minibatch["truncateds"] > 0.5
        terminations = dones | truncateds
        teach_act_log_probs[terminations] = 0.0

        teach_act_log_probs_shift = torch.zeros_like(teach_act_log_probs)
        teach_act_log_probs_shift[:-1] = teach_act_log_probs[1:]

        rewards = minibatch["rewards"]
        rewards = rewards + self.cfg.r_lambda * teach_act_log_probs_shift
        minibatch["rewards"] = rewards

        advantages = compute_advantage(
            minibatch["values"],
            rewards,
            minibatch["dones"],
            torch.ones_like(minibatch["values"]),
            torch.zeros_like(minibatch["values"], device=self.device),
            self.trainer_cfg.advantage.gamma,
            self.trainer_cfg.advantage.gae_lambda,
            self.device,
        )

        minibatch["advantages"] = advantages

        # action loss (cross entropy)
        student_full_log_probs = student_td["full_log_probs"]
        teacher_full_log_probs = minibatch["teacher_full_log_probs"]
        ks_action_loss = (student_full_log_probs.exp() * teacher_full_log_probs).sum(dim=-1).mean()

        # value loss
        teacher_value = minibatch["teacher_values"].to(dtype=torch.float32).detach()
        student_value = student_td["values"].to(dtype=torch.float32)
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

        self.loss_tracker["ks_act_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["ks_val_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["ks_act_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["ks_val_loss_coef"].append(float(self.cfg.value_loss_coef))

        return loss, shared_loss_data, False
