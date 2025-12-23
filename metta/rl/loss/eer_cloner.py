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
    r_lambda: float = Field(default=0.05, ge=0)  # scale the teacher log likelihoods that are added to rewards

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

    def get_experience_spec(self) -> Composite:
        return Composite(teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long))

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
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

        teacher_actions = minibatch["teacher_actions"]
        student_actions = minibatch["actions"]
        teach_act_reward = teacher_actions == student_actions
        teach_act_reward_clamped = torch.clamp(teach_act_reward, min=1e-2)

        dones = minibatch["dones"] > 0.5
        truncateds = minibatch["truncateds"] > 0.5
        terminations = dones | truncateds
        teach_act_reward_clamped[terminations] = 0.0

        teach_act_reward_shift = torch.zeros_like(teach_act_reward)
        teach_act_reward_shift[:-1] = teach_act_reward_clamped[1:]

        rewards = minibatch["rewards"]
        rewards = rewards + self.cfg.r_lambda * teach_act_reward_shift
        minibatch["rewards"] = rewards

        centered_rewards = rewards - minibatch["reward_baseline"]
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

        policy_td = shared_loss_data["policy_td"]

        policy_full_log_probs = policy_td["full_log_probs"].reshape(minibatch.shape[0], minibatch.shape[1], -1)
        teacher_actions = minibatch["teacher_actions"]
        # get the student's logprob for the action that the teacher chose
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0], minibatch.shape[1])

        loss = -student_log_probs.mean() * self.cfg.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(loss.item()))

        return loss, shared_loss_data, False
