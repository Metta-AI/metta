from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.replay_sampler import sample_minibatch_sequential
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import ComponentContext
from mettagrid.base_config import Config


class ActionSupervisedConfig(Config):
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    use_own_sampling: bool = (True,)
    use_own_rollout: bool = (True,)
    student_led: bool = (True,)  # sigma as per Matt's document
    loss_type: str = Field(default="BCE")  # one of {"BCE", "MSE", "COSINE"}. Eliminate this hyper after testing

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


# helper to extract teacher actions from env obs. Then infer full logits and return.
def extract_teacher_logits_from_env_obs(env_obs: Tensor, action_space: Any) -> Tensor:
    # --> Run helper to extract teacher actions from env obs
    pass


# helper to delete teacher tokens from env obs
def delete_teacher_tokens_from_env_obs(env_obs: Tensor, teacher_tokens: Tensor) -> Tensor:
    pass


# helper to translate teacher logits (centering)
def translate_teacher_logits(teacher_logits: Tensor) -> Tensor:
    pass


# helper to multinomial sample teacher actions
def multinomial_sample_teacher_actions(teacher_logits: Tensor) -> Tensor:
    pass


class ActionSupervised(Loss):
    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "use_own_rollout",
        "use_own_sampling",
        "loss_type",
        "teacher_has_action_log_prob",
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
        self.use_own_rollout = self.loss_cfg.use_own_rollout
        self.use_own_sampling = self.loss_cfg.use_own_sampling
        self.loss_type = self.loss_cfg.loss_type

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            logits=scalar_f32,  # student logits
            full_log_probs=scalar_f32,  # student full log_probs
            teacher_logits=scalar_f32,
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

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

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
            targets = F.softmax(teacher_logits, dim=-1)
            ks_action_loss = F.binary_cross_entropy_with_logits(student_logits, targets)
        elif self.loss_type == "MSE":
            student_logits = minibatch["logits"].to(dtype=torch.float32)
            teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).detach()
            # --> Run helper to translate teacher logits
            teacher_logits = translate_teacher_logits(teacher_logits)
            ks_action_loss = F.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == "COSINE":
            # --> Run helper to translate teacher logits
            student_logits = minibatch["logits"].to(dtype=torch.float32)
            teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).detach()
            ks_action_loss = (1.0 - F.cosine_similarity(student_logits, teacher_logits, dim=-1)).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        loss = ks_action_loss * self.action_loss_coef

        self.loss_tracker["sl_ks_action_loss"].append(float(ks_action_loss.item()))

        return loss, shared_loss_data, False
