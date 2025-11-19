from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.replay_samplers import sequential_sample
from metta.rl.training import ComponentContext
from metta.rl.utils import prepare_policy_forward_td
from mettagrid.policy.loader import initialize_or_load_policy

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SlicedKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)
    student_forward: bool = Field(default=True)  # probably always true for sliced_kickstarter
    teacher_lead_prob: float = Field(default=1.0, ge=0, le=1.0)  # set to 1 since we slice teacher lead separately

    # remainder of the sum below is left for the PPO loss to use
    student_led_proportion: float = Field(default=0.2, ge=0, le=1.0)
    teacher_led_proportion: float = Field(default=0.5, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "SlicedKickstarter":
        """Create Kickstarter loss instance."""
        return SlicedKickstarter(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class SlicedKickstarter(Loss):
    """This uses another policy that is forwarded during rollout, here, in the loss and then compares its logits and
    value against the student's using a KL divergence and MSE loss respectively.
    """

    __slots__ = (
        "teacher_policy",
        "student_forward",
        "rollout_batch_size",
        "stud_mask",
        "teacher_mask",
        "ppo_mask",
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
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.student_forward = self.cfg.student_forward

        # Load teacher. Lazy import to avoid circular dependency
        from metta.rl.checkpoint_manager import CheckpointManager

        policy_env_info = getattr(self.env, "policy_env_info", None)
        if policy_env_info is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")
        teacher_spec = CheckpointManager.policy_spec_from_uri(self.cfg.teacher_uri, device=self.device)
        self.teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec)

    def get_experience_spec(self) -> Composite:
        # Get action space size for logits shape
        act_space = self.env.single_action_space
        num_actions = act_space.n

        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        logits_f32 = UnboundedContinuous(shape=torch.Size([num_actions]), dtype=torch.float32)
        boolean = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool)

        return Composite(
            teacher_logits=logits_f32,
            teacher_values=scalar_f32,
            stud_mask=boolean,
            teacher_mask=boolean,
            ppo_mask=boolean,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            teacher_td = td.clone()
            self.teacher_policy.forward(teacher_td)
            teacher_actions = teacher_td["actions"]
            td["teacher_logits"] = teacher_td["logits"]
            td["teacher_values"] = teacher_td["values"]
            self.policy.forward(td)

        if not hasattr(self, "rollout_batch_size") or self.rollout_batch_size != td.batch_size.numel():
            self._create_slices(td.batch_size.numel())

        td["stud_mask"] = self.stud_mask
        td["teacher_mask"] = self.teacher_mask
        td["ppo_mask"] = self.ppo_mask

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

        if torch.rand(1) < self.cfg.teacher_lead_prob:
            # overwrite student actions w teacher actions with some probability. anneal this.
            td["actions"][self.teacher_mask] = teacher_actions[self.teacher_mask]  # slice - teacher led

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch, indices = sequential_sample(self.replay, mb_idx)
        # slice - minus teacher led minus student led
        train_stud_mask = minibatch["stud_mask"][:, 0]
        train_teacher_mask = minibatch["teacher_mask"][:, 0]
        train_ppo_mask = minibatch["ppo_mask"][:, 0]

        shared_loss_data["sampled_mb"] = minibatch

        # cut down all of shared_loss_data to just the ppo mask before passing out to PPO losses
        shared_loss_data = shared_loss_data[train_ppo_mask]

        # slice - minus teacher led minus student led
        shared_loss_data["indices"] = NonTensorData(indices[train_ppo_mask])
        # this writes to the same key that ppo uses, assuming we're using only one method of sampling at a time

        # Student forward pass
        # leave to false if also running PPO since it forwards student during train
        if self.student_forward:
            student_td, B, TT = prepare_policy_forward_td(minibatch, self.policy_experience_spec, clone=False)
            flat_actions = minibatch["actions"].reshape(B * TT, -1)
            self.policy.reset_memory()
            student_td = self.policy.forward(student_td, action=flat_actions)
            student_td = student_td.reshape(B, TT)
            shared_loss_data["policy_td"] = student_td[train_ppo_mask]  # this is for passing to PPO losses
        else:
            student_td = shared_loss_data["policy_td"]  # shared_loss_data is populated by PPO

        minibatch = minibatch[train_teacher_mask | train_stud_mask]
        student_td = student_td[train_teacher_mask | train_stud_mask]

        sliced_b, sliced_tt = minibatch.batch_size
        minibatch = minibatch.reshape(sliced_b * sliced_tt)
        student_td = student_td.reshape(sliced_b * sliced_tt)

        # action loss
        temperature = self.cfg.temperature
        teacher_logits = minibatch["teacher_logits"]
        student_logits = student_td["logits"]
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_probs = torch.exp(student_log_probs)
        ks_action_loss = (temperature**2) * (
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        )

        # value loss
        teacher_value = minibatch["teacher_values"].detach()
        student_value = student_td["values"]
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

        self.loss_tracker["ks_act_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["ks_val_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["ks_act_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["ks_val_loss_coef"].append(float(self.cfg.value_loss_coef))

        return loss, shared_loss_data, False

    def on_train_phase_end(self, context: ComponentContext | None = None) -> None:
        self._update_slices()
        super().on_train_phase_end(context)

    def _create_slices(self, B: int) -> None:
        self.rollout_batch_size = B
        stud_led_count = int(B * self.cfg.student_led_proportion)
        stud_slice = slice(0, stud_led_count)
        teacher_led_count = int(B * self.cfg.teacher_led_proportion)
        teacher_slice = slice(stud_led_count, stud_led_count + teacher_led_count)
        ppo_count = B - stud_led_count - teacher_led_count
        if ppo_count < 0:
            raise ValueError("PPO count error in sliced Kickstarter loss. Bad proportions.")
        ppo_slice = slice(stud_led_count + teacher_led_count, B)

        self.stud_mask = torch.zeros(B, dtype=torch.bool)
        self.stud_mask[stud_slice] = True
        self.teacher_mask = torch.zeros(B, dtype=torch.bool)
        self.teacher_mask[teacher_slice] = True
        self.ppo_mask = torch.zeros(B, dtype=torch.bool)
        self.ppo_mask[ppo_slice] = True

    def _update_slices(self) -> None:
        # we count on the hyperparmeter scheduler to update the cfg proportions
        self._create_slices(self.rollout_batch_size)
