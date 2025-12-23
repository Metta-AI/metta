from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
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
    r_lambda: float = Field(default=0.01, ge=0)  # scale the teacher log likelihoods that are added to rewards

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

    Implements:
    1. Reward shaping: r' = r + lambda * log(pi_teacher(a|s))
       This corresponds to minimizing Expected Entropy Regularized objective.
    2. Auxiliary Distillation Loss: KL(pi_student || pi_teacher) minimization term.
    """

    __slots__ = ("teacher_policy", "last_teacher_log_probs", "has_last_probs")

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

        # Cache for teacher log probs from previous step, needed for reward shaping R_{t-1} + log(pi(A_{t-1}))
        # We need this because run_rollout receives R_t (reward for action at t-1), but computes pi(S_t).
        # So we must use the cached pi(S_{t-1}) to shape R_t.
        num_agents = self.env.total_parallel_agents
        num_actions = self.env.single_action_space.n
        self.last_teacher_log_probs = torch.zeros((num_agents, num_actions), device=self.device)
        self.has_last_probs = torch.zeros(num_agents, dtype=torch.bool, device=self.device)

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

            # Store teacher outputs for auxiliary loss and value loss
            td["teacher_full_log_probs"] = teacher_td["full_log_probs"]
            td["teacher_values"] = teacher_td["values"]

            self.policy.forward(td)

            # --- Reward Shaping ---
            # td["rewards"] contains R_{t-1}. We want to add r_lambda * log(pi_teacher(A_{t-1}|S_{t-1})).
            # We use cached teacher probs from the previous step.
            env_slice = context.training_env_id
            indices = torch.arange(env_slice.start, env_slice.stop, device=self.device)

            valid_mask = self.has_last_probs[indices]
            if valid_mask.any():
                last_probs = self.last_teacher_log_probs[indices]

                # Get actions taken at t-1: (batch,)
                last_actions = td["last_actions"]
                if last_actions.dim() > 1:
                    last_actions = last_actions.squeeze(-1)
                last_actions = last_actions.long()

                # Gather log prob of taken action: (batch,)
                intrinsic_reward = last_probs.gather(1, last_actions.unsqueeze(1)).squeeze(1)

                # Add to rewards in place (modifies the buffer view)
                td["rewards"] += self.cfg.r_lambda * intrinsic_reward * valid_mask.float()

            # Update cache for next step
            self.last_teacher_log_probs[indices] = teacher_td["full_log_probs"]
            self.has_last_probs[indices] = True

        # Store experience
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

        student_full_log_probs = student_td["full_log_probs"]
        teacher_full_log_probs = minibatch["teacher_full_log_probs"]
        ks_action_loss = -(student_full_log_probs.exp() * teacher_full_log_probs).sum(dim=-1).mean()

        # Value loss
        teacher_value = minibatch["teacher_values"].to(dtype=torch.float32).detach()
        student_value = student_td["values"].to(dtype=torch.float32)
        ks_value_loss_vec = (teacher_value.detach() - student_value) ** 2
        ks_value_loss = ks_value_loss_vec.mean()

        shared_loss_data["ks_val_loss_vec"] = ks_value_loss_vec

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

        # track losses for plotting
        self.loss_tracker["ks_act_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["ks_val_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["ks_act_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["ks_val_loss_coef"].append(float(self.cfg.value_loss_coef))

        return loss, shared_loss_data, False
