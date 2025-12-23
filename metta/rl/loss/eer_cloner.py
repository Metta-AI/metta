from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedDiscrete

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig
from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext


class EERClonerConfig(LossConfig):
    action_loss_coef: float = Field(default=1, ge=0)
    r_lambda: float = Field(default=0.01, ge=0)  # scale the teacher log likelihoods that are added to rewards

    # Probability floor for teacher actions not taken (to avoid log(0) = -inf). note that this gets mult by r_lambda too
    teacher_prob_floor: float = Field(default=0.01, ge=1e-6, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "EERCloner":
        """Create EERCloner loss instance."""
        return EERCloner(policy, trainer_cfg, vec_env, device, instance_name, self)


class EERCloner(Loss):
    __slots__ = ("last_teacher_actions", "has_last_actions")

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "EERClonerConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)

        # Cache for teacher actions from previous step, needed for reward shaping R_{t-1} + log(pi(A_{t-1}))
        num_agents = self.env.total_parallel_agents
        self.last_teacher_actions = torch.zeros(num_agents, device=self.device, dtype=torch.long)
        self.has_last_actions = torch.zeros(num_agents, dtype=torch.bool, device=self.device)

    def get_experience_spec(self) -> Composite:
        return Composite(teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long))

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        env_slice = self._training_env_id(context)

        # --- Reward Shaping ---
        # td["rewards"] contains R_{t-1}. We want to add r_lambda * log(pi_teacher(A_{t-1}|S_{t-1})).
        # For cloner, the teacher output is just an action index.
        # We treat this as a deterministic distribution (or peaked):
        # If A_{t-1} == TeacherAction_{t-1}: log(prob) then loss is log(1) = 0
        # If A_{t-1} != TeacherAction_{t-1}: log(prob) then loss is log(epsilon)

        indices = torch.arange(env_slice.start, env_slice.stop, device=self.device)

        valid_mask = self.has_last_actions[indices]

        if valid_mask.any():
            # Get cached teacher actions from t-1
            last_teacher_acts = self.last_teacher_actions[indices]

            # Get actions actually taken at t-1
            last_actions = td["last_actions"]
            if last_actions.dim() > 1:
                last_actions = last_actions.squeeze(-1)
            last_actions = last_actions.long()

            # Compare: 1.0 if match, prob_floor if mismatch
            matches = (last_teacher_acts == last_actions).float()
            probs = matches * (1.0 - self.cfg.teacher_prob_floor) + self.cfg.teacher_prob_floor

            # Compute log likelihood
            intrinsic_reward = torch.log(probs)

            # Add to rewards in place
            td["rewards"] += self.cfg.r_lambda * intrinsic_reward * valid_mask.float()

        teacher_actions = td["teacher_actions"]
        self.last_teacher_actions[indices] = teacher_actions
        self.has_last_actions[indices] = True
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"full_log_probs"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        policy_td = shared_loss_data["policy_td"]

        centered_rewards = minibatch["rewards"] - minibatch["reward_baseline"]
        advantages = compute_advantage(
            minibatch["values"],
            centered_rewards,
            minibatch["dones"],
            torch.ones_like(minibatch["values"]),
            torch.zeros_like(minibatch["values"], device=self.device),
            self.trainer_cfg.advantage.gamma,
            self.trainer_cfg.advantage.gae_lambda,
            self.device,
        )
        minibatch["advantages"] = advantages

        # Supervised Loss: Maximize log probability of the teacher's action -> L = - log(pi_student(a_teacher | s))
        policy_full_log_probs = policy_td["full_log_probs"].reshape(minibatch.shape[0], minibatch.shape[1], -1)
        teacher_actions = minibatch["teacher_actions"]
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0], minibatch.shape[1])

        loss = -student_log_probs.mean() * self.cfg.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(loss.item()))

        return loss, shared_loss_data, False
