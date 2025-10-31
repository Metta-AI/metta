from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss import Loss
from metta.rl.loss.replay_sampler import sample_minibatch_sequential
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import ComponentContext
from mettagrid.base_config import Config


class ActionSupervisedConfig(Config):
    action_loss_coef: float = Field(default=0.75, ge=0)
    value_loss_coef: float = Field(default=1.5, ge=0)
    gae_gamma: float = Field(default=0.977, ge=0, le=1.0)  # pulling from our PPO config
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)  # pulling from our PPO config
    vf_clip_coef: float = Field(default=0.1, ge=0)  # pulling from our PPO config
    use_own_sampling: bool = True  # Does not use prioritized sampling
    use_own_rollout: bool = True  # Update when including PPO as concurent loss
    student_led: bool = True  # sigma as per Matt's document
    action_reward_coef: float = Field(default=0.01, ge=0)  # wild ass guess at this point

    # Below: branches for testing different approaches. Hopefully we can eliminate these hypers after testing.
    loss_type: str = Field(default="BCE")  # one of {"BCE", "MSE", "COSINE"}.
    add_action_loss_to_rewards: bool = Field(default=True)
    norm_adv: bool = Field(default=True)

    def create(
        self,
        policy: Policy,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create ActionSupervised loss instance."""
        return ActionSupervised(
            policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config
        )


# --------------------------Helper Functions----------------------------------
# helper to extract teacher actions from env obs. Then infer full logits and return.
def extract_teacher_logits_from_env_obs(env_obs: Tensor, action_space: Any) -> Tensor:
    # --> Run helper to extract teacher actions from env obs
    pass


# helper to delete teacher tokens from env obs
def delete_teacher_tokens_from_env_obs(env_obs: Tensor, teacher_tokens: Tensor) -> Tensor:
    pass


# helper to translate teacher logits (centering)
def center_teacher_logits(teacher_logits: Tensor) -> Tensor:
    pass


# helper to multinomial sample teacher actions
def multinomial_sample_teacher_actions(teacher_logits: Tensor) -> Tensor:
    pass


# --------------------------ActionSupervised Loss----------------------------------
class ActionSupervised(Loss):
    __slots__ = (
        "action_loss_coef",
        "value_loss_coef",
        "norm_adv",
        "vf_clip_coef",
        "gae_gamma",
        "gae_lambda",
        "add_action_loss_to_rewards",
        "use_own_rollout",
        "use_own_sampling",
        "loss_type",
        "student_led",
        "action_reward_coef",
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
        # unpack config into slots
        self.action_loss_coef = self.loss_cfg.action_loss_coef
        self.value_loss_coef = self.loss_cfg.value_loss_coef
        self.norm_adv = self.loss_cfg.norm_adv
        self.vf_clip_coef = self.loss_cfg.vf_clip_coef
        self.gae_gamma = self.loss_cfg.gae_gamma
        self.gae_lambda = self.loss_cfg.gae_lambda
        self.add_action_loss_to_rewards = self.loss_cfg.add_action_loss_to_rewards
        self.use_own_rollout = self.loss_cfg.use_own_rollout
        self.use_own_sampling = self.loss_cfg.use_own_sampling
        self.loss_type = self.loss_cfg.loss_type
        self.student_led = self.loss_cfg.student_led
        self.action_reward_coef = self.loss_cfg.action_reward_coef

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        logits_shape = torch.Size([int(act_space.n)])
        logits_f32 = UnboundedContinuous(shape=logits_shape, dtype=torch.float32)

        return Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            logits=logits_f32,  # student logits
            teacher_logits=logits_f32,
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            values=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        if not self.use_own_rollout:
            return

        # extract teacher output from env obs, construct teacher logits, and remove teacher tokens from env obs
        teacher_logits, env_obs_without_teacher_tokens = extract_teacher_logits_from_env_obs(
            td["env_obs"], self.env.single_action_space
        )
        td["teacher_logits"] = teacher_logits
        td["env_obs"] = env_obs_without_teacher_tokens

        # running with grad on in rollout since this is the only place we run student's forward. this is faster than
        # running it in training as per usual. we'll likely revert once including PPO. see note under
        # `if not student_led` below.
        self.policy.forward(td)

        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

        if not self.student_led:
            # we'll need to modify this logic when we move to including PPO by calling the student forward under
            # run_train(). For now, we save td["action"] into the td that goes to the replay buffer but then overwrite
            # it with teacher actions when sending to the environment. After it gets sent to env it is no longer used.

            # NOTE: teacher-leading means actions reported to wandb are teacher actions, not student actions

            teacher_actions = multinomial_sample_teacher_actions(teacher_logits)
            td["actions"] = teacher_actions

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

        # downselect to one of these after testing
        if self.loss_type == "BCE":
            student_logits = minibatch["logits"].to(dtype=torch.float32)
            teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).detach()
            # NOTE: we assume teacher logits are either 0 or 1.
            bce_per_class = F.binary_cross_entropy_with_logits(
                student_logits, teacher_logits.detach(), reduction="none"
            )  # not reduced yet, (B, T, A)
            imitation_per_step = bce_per_class.mean(dim=-1)  # reduced over class so we can pass it to reward if needed
            if self.add_action_loss_to_rewards:
                minibatch["rewards"] = minibatch["rewards"] + self.action_reward_coef * imitation_per_step.detach()
            actor_loss = bce_per_class.mean()  # now reduced over batch (same as batch mean)
        elif self.loss_type == "MSE":
            student_logits = minibatch["logits"].to(dtype=torch.float32)
            teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).detach()
            teacher_logits = center_teacher_logits(teacher_logits)
            actor_loss = F.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == "COSINE":
            student_logits = minibatch["logits"].to(dtype=torch.float32)
            teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).detach()
            teacher_logits = center_teacher_logits(teacher_logits)
            actor_loss = (1.0 - F.cosine_similarity(student_logits, teacher_logits, dim=-1)).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        actor_loss = actor_loss * self.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(actor_loss.item()))

        # --------------------------Now Value Loss----------------------------------
        advantages = torch.zeros_like(minibatch["values"], device=self.device)
        advantages = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            torch.ones_like(minibatch["values"]),  # effectively deactivates v-trace
            advantages,
            self.gae_gamma,
            self.gae_lambda,
            1.0,  # no v-trace
            1.0,  # no v-trace
            self.device,
        )

        returns = advantages + minibatch["values"]
        if self.norm_adv:
            advantages = normalize_advantage_distributed(advantages)

        # compute value loss
        old_values = minibatch["values"]
        newvalue_reshaped = minibatch["values"].view(returns.shape)
        v_loss_unclipped = (newvalue_reshaped - returns) ** 2
        v_clipped = old_values + torch.clamp(
            newvalue_reshaped - old_values,
            -self.vf_clip_coef,
            self.vf_clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean() * self.value_loss_coef

        self.loss_tracker["supervised_value_loss"].append(float(value_loss.item()))

        loss = actor_loss + value_loss

        return loss, shared_loss_data, False
