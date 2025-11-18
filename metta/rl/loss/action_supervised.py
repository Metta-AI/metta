from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig
from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.replay_samplers import sequential_sample
from metta.rl.training import ComponentContext


class ActionSupervisedConfig(LossConfig):
    action_loss_coef: float = Field(default=0.75, ge=0)
    sample_enabled: bool = True  # Does not use prioritized sampling
    rollout_forward_enabled: bool = True  # Update when including PPO as concurent loss
    train_forward_enabled: bool = True  # Update when including PPO as concurent loss
    teacher_lead_prob: float = Field(default=0.0, ge=0, le=1.0)  # at 0.0, it's purely student-led
    action_reward_coef: float = Field(default=0.01, ge=0)  # wild ass guess at this point

    # Controls whether to add the imitation loss to the environment rewards.
    add_action_loss_to_rewards: bool = Field(default=True)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "ActionSupervised":
        """Create ActionSupervised loss instance."""
        return ActionSupervised(
            policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config
        )


# --------------------------ActionSupervised Loss----------------------------------
class ActionSupervised(Loss):
    __slots__ = (
        "action_loss_coef",
        "add_action_loss_to_rewards",
        "rollout_forward_enabled",
        "train_forward_enabled",
        "sample_enabled",
        "teacher_lead_prob",
        "action_reward_coef",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any = None,
    ):
        # Get loss config from trainer_cfg if not provided
        if loss_config is None:
            loss_config = getattr(trainer_cfg.losses, instance_name, None)
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        # unpack config into slots
        self.action_loss_coef = self.cfg.action_loss_coef
        self.add_action_loss_to_rewards = self.cfg.add_action_loss_to_rewards
        self.rollout_forward_enabled = self.cfg.rollout_forward_enabled
        self.train_forward_enabled = self.cfg.train_forward_enabled
        self.sample_enabled = self.cfg.sample_enabled
        self.teacher_lead_prob: float = Field(default=0.0, ge=0, le=1.0)  # at 0.0, it's purely student-led
        self.action_reward_coef = self.cfg.action_reward_coef

    def get_experience_spec(self) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long),
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        if not self.rollout_forward_enabled:  # this can also be achived with loss run gates
            return

        if self.rollout_forward_enabled:  # flag offered for speed in cases where purely teacher led
            with torch.no_grad():
                self.policy.forward(td)

        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)

        if torch.rand(1) < self.cfg.teacher_lead_prob:
            # Save td["action"] into the td that goes to the replay buffer but then overwrite it with teacher actions
            # when sending to the environment. After it gets sent to env it is no longer used.
            # NOTE: teacher-leading means actions reported to wandb are teacher actions, not student actions
            td["actions"] = td["teacher_actions"]

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        if self.sample_enabled:
            minibatch, indices = sequential_sample(self.replay, mb_idx)
            shared_loss_data["sampled_mb"] = minibatch
            shared_loss_data["indices"] = NonTensorData(indices)
            # this writes to the same key that ppo uses, assuming we're using only one method of sampling at a time

        minibatch = shared_loss_data["sampled_mb"]

        # av update the below to use the sampler. forward then in the rollout. Should overtake prio sampler rollout.
        # use loss run gate accordingly
        # then clean up the below since the sampler runs the policy in training.
        # add the gather here, going of the policy's full logprobs
        if self.train_forward_enabled:
            policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
            B, TT = policy_td.batch_size
            policy_td = policy_td.reshape(B * TT)
            policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
            policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))
            flat_actions = minibatch["actions"].reshape(B * TT, -1)
            self.policy.reset_memory()
            policy_td = self.policy.forward(policy_td, action=flat_actions)
            policy_td = policy_td.reshape(B, TT)
            shared_loss_data["policy_td"] = policy_td

        # AV: the above runs a gather on the teacher actions against the student's logprobs which is the same as CE loss
        # so that's slick. But that means we shouldn't write this policy td to shared_loss_data when using PPO!
        # That means that when using PPO we need to write a separate gather here.
        else:
            policy_td = shared_loss_data["policy_td"]

        policy_full_log_probs = policy_td["full_log_probs"].reshape(minibatch.shape[0], minibatch.shape[1], -1)
        teacher_actions = minibatch["teacher_actions"]
        # run a gather to get the student's logprobs for the teacher actions
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0], minibatch.shape[1])

        loss = -student_log_probs.mean() * self.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(loss.item()))

        # --------------------------Add action loss to rewards as per Matt's doc----------------------------------
        if self.add_action_loss_to_rewards:
            minibatch["rewards"] = minibatch["rewards"] + self.action_reward_coef * policy_td["act_log_prob"].detach()
            # NOTE: we should somehow normalize the policy loss before adding it to rewards, perhaps exponentiate then
            # softplus?

        return loss, shared_loss_data, False
