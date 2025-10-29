from typing import Any

import torch
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


class SLKickstarterConfig(Config):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    use_kl_div: bool = False,
    use_own_sampling: bool = True,
    use_own_rollout: bool = True,

    def create(
        self,
        policy: Policy,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create SLKickstarter loss instance."""
        return SLKickstarter(policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config)


class SLKickstarter(Loss):
    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "action_loss_coef",
        "value_loss_coef",
        "use_own_rollout",
        "use_own_sampling",
        "use_kl_div",
        "burn_in_steps",
        "burn_in_steps_iter",
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
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.action_loss_coef = self.loss_cfg.action_loss_coef
        self.value_loss_coef = self.loss_cfg.value_loss_coef
        self.use_own_rollout = self.loss_cfg.use_own_rollout
        self.use_own_sampling = self.loss_cfg.use_own_sampling
        self.use_kl_div = self.loss_cfg.use_kl_div
        self.burn_in_steps = 0
        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        self.burn_in_steps_iter = 0

        # load teacher policy
        from metta.rl.checkpoint_manager import CheckpointManager

        game_rules = getattr(self.env, "game_rules", getattr(self.env, "meta_data", None))
        if game_rules is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        self.teacher_policy = CheckpointManager.load_from_uri(self.loss_cfg.teacher_uri, game_rules, self.device)

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            act_log_prob=scalar_f32,
            full_log_probs=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        if not self.use_own_rollout:
            return

        with torch.no_grad():
            self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        # !!!!! Need to extract teacher output from env obs and store in td["teacher_actions"] and
        # td["teacher_full_log_probs"]!!!!

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

        return

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

        if self.use_kl_div:
            student_log_probs = minibatch["full_log_probs"].to(dtype=torch.float32)
            teacher_log_probs = minibatch["teacher_full_log_probs"].to(dtype=torch.float32).detach()
            student_probs = torch.exp(student_log_probs)
            ks_action_loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        else:
            student_action = minibatch["actions"].to(dtype=torch.int32)
            teacher_action = minibatch["teacher_actions"].to(dtype=torch.int32).detach()
            matching_actions = (teacher_action == student_action).float()
            ks_action_loss = (1.0 - matching_actions).mean()

        loss = ks_action_loss * self.action_loss_coef

        self.loss_tracker["sl_ks_action_loss"].append(float(ks_action_loss.item()))

        return loss, shared_loss_data, False
