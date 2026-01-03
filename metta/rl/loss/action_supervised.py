from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig
from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext


class ActionSupervisedConfig(LossConfig):
    action_loss_coef: float = Field(default=1, ge=0)
    sample_enabled: bool = True  # True means sequentially sample from the buffer during train in this loss
    rollout_forward_enabled: bool = True  # Control the rollout. If true, ensure ppo_critic is not also running rollout
    train_forward_enabled: bool = True  # Forward policy during training. Same as above re PPO concurency collisions.
    teacher_led_proportion: float = Field(default=0.0, ge=0, le=1.0)  # at 0.0, it's purely student-led

    # Controls whether to add the imitation loss to the environment rewards.
    add_action_loss_to_rewards: bool = Field(default=False)
    action_reward_coef: float = Field(default=0.01, ge=0)  # value is awild ass guess

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "ActionSupervised":
        """Create ActionSupervised loss instance."""
        return ActionSupervised(policy, trainer_cfg, vec_env, device, instance_name, self)


class ActionSupervised(Loss):
    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "ActionSupervisedConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)

    def get_experience_spec(self) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        action_spec = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int32)

        return Composite(
            actions=action_spec,
            teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long),
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        env_slice = self._training_env_id(context)
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)

        if torch.rand(1) < self.cfg.teacher_led_proportion:
            # Save td["action"] into the td that goes to the replay buffer but then overwrite it with teacher actions
            # when sending to the environment. After it gets sent to env it is no longer used.
            # NOTE: teacher-leading means actions reported to wandb are teacher actions, not student actions
            td["actions"] = td["teacher_actions"].to(td["actions"].dtype)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"full_log_probs", "act_log_prob"} if self.cfg.add_action_loss_to_rewards else {"full_log_probs"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        policy_td = shared_loss_data["policy_td"]

        policy_full_log_probs = policy_td["full_log_probs"].reshape(minibatch.shape[0], minibatch.shape[1], -1)
        teacher_actions = minibatch["teacher_actions"]
        # get the student's logprob for the action that the teacher chose
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0], minibatch.shape[1])

        loss = -student_log_probs.mean() * self.cfg.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(loss.item()))

        # --------------------------Add action loss to rewards as per Matt's doc----------------------------------
        if self.cfg.add_action_loss_to_rewards:
            minibatch["rewards"] = (
                minibatch["rewards"] + self.cfg.action_reward_coef * policy_td["act_log_prob"].detach()
            )
            # NOTE: we should somehow normalize the policy loss before adding it to rewards, perhaps exponentiate then
            # softplus?

        return loss, shared_loss_data, False
