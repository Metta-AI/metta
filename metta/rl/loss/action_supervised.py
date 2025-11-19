from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig
from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.replay_samplers import sample_minibatch_sequential
from metta.rl.training import ComponentContext


class ActionSupervisedConfig(LossConfig):
    action_loss_coef: float = Field(default=0.6, ge=0)
    value_loss_coef: float = Field(default=1.0, ge=0)
    gae_gamma: float = Field(default=0.977, ge=0, le=1.0)  # pulling from our PPO config
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)  # pulling from our PPO config
    vf_clip_coef: float = Field(default=0.1, ge=0)  # pulling from our PPO config
    use_own_sampling: bool = True  # Does not use prioritized sampling
    use_own_rollout: bool = True  # Update when including PPO as concurent loss
    student_led: bool = Field(
        default=False, description="Whether to use student-led training"
    )  # sigma as per Matt's document
    action_reward_coef: float = Field(default=0.01, ge=0)  # wild ass guess at this point

    # Controls whether to add the imitation loss to the environment rewards.
    add_action_loss_to_rewards: bool = Field(default=False)
    norm_adv: bool = Field(default=True)

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
        "norm_adv",
        "gae_gamma",
        "gae_lambda",
        "add_action_loss_to_rewards",
        "use_own_rollout",
        "use_own_sampling",
        "student_led",
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
        self.norm_adv = self.cfg.norm_adv
        self.gae_gamma = self.cfg.gae_gamma
        self.gae_lambda = self.cfg.gae_lambda
        self.add_action_loss_to_rewards = self.cfg.add_action_loss_to_rewards
        self.use_own_rollout = self.cfg.use_own_rollout
        self.use_own_sampling = self.cfg.use_own_sampling
        self.student_led = self.cfg.student_led
        self.action_reward_coef = self.cfg.action_reward_coef

    def get_experience_spec(self) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        spec = Composite(
            teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long),
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
        )

        if self.student_led:
            spec["values"] = scalar_f32

        return spec

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        if not self.use_own_rollout:
            # this will be replaced with loss run-gate scheduling
            return

        if self.student_led:
            with torch.no_grad():
                self.policy.forward(td)
                # if we were only using this loss (and not PPO) we could run the loss here and skip the train loop for
                # speed. However, not doing that in anticipation of the default being to use PPO.

        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)

        if not self.student_led:
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
        if self.use_own_sampling:
            minibatch, _ = sample_minibatch_sequential(self.replay, mb_idx)
            shared_loss_data["sampled_mb"] = minibatch
            # this writes to the same key that ppo uses, assuming we're using only one method of sampling at a time

        minibatch = shared_loss_data["sampled_mb"]

        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B, TT = policy_td.batch_size
        flat_policy_td = policy_td.reshape(B * TT)
        flat_policy_td.set("bptt", torch.full((B * TT,), TT, device=flat_policy_td.device, dtype=torch.long))
        flat_policy_td.set("batch", torch.full((B * TT,), B, device=flat_policy_td.device, dtype=torch.long))

        teacher_actions = minibatch["teacher_actions"].to(device=flat_policy_td.device, dtype=torch.long)
        flat_teacher_actions = teacher_actions.reshape(B * TT, -1)

        self.policy.reset_memory()
        policy_td = self.policy.forward(flat_policy_td, action=flat_teacher_actions).reshape(B, TT)
        # AV: the above runs a gather on the teacher actions against the student's logprobs which is the same as CE loss
        # so that's slick. But that means we shouldn't write this policy td to shared_loss_data when using PPO!
        # That means that when using PPO we need to write a separate gather here.

        actor_loss = -policy_td["act_log_prob"].mean() * self.cfg.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(actor_loss.item()))

        # --------------------------Now Value Loss----------------------------------
        if self.add_action_loss_to_rewards:
            minibatch["rewards"] = minibatch["rewards"] + self.action_reward_coef * policy_td["act_log_prob"].detach()
            # NOTE: we should somehow normalize the policy loss before adding it to rewards, perhaps exponentiate then
            # softplus?

        if self.student_led:
            values = minibatch["values"]
        else:
            values = policy_td["values"].detach()

        advantages = torch.zeros_like(values, device=self.device)
        advantages = compute_advantage(
            values,
            minibatch["rewards"],
            minibatch["dones"],
            torch.ones_like(values),  # effectively deactivates v-trace
            advantages,
            self.gae_gamma,
            self.gae_lambda,
            1.0,  # no v-trace
            1.0,  # no v-trace
            self.device,
        )

        returns = advantages + values
        if self.norm_adv:
            advantages = normalize_advantage_distributed(advantages)

        # compute value loss
        if self.student_led:
            old_values = minibatch["values"]
            newvalue = policy_td["values"].view(returns.shape)
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = old_values + torch.clamp(
                newvalue - old_values,
                -self.cfg.vf_clip_coef,
                self.cfg.vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean() * self.cfg.value_loss_coef
        else:
            new_values = policy_td["values"].view(returns.shape)
            value_loss = 0.5 * ((new_values - returns) ** 2).mean() * self.cfg.value_loss_coef

        self.loss_tracker["supervised_value_loss"].append(float(value_loss.item()))

        loss = actor_loss + value_loss

        return loss, shared_loss_data, False
