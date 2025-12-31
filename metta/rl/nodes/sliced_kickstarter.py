from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.nodes.base import NodeBase, NodeConfig
from metta.rl.nodes.registry import NodeSpec
from metta.rl.nodes.teacher_policy import load_teacher_policy
from metta.rl.training import ComponentContext

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SlicedKickstarterConfig(NodeConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=10.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)

    # PPO consumes whatever portion of the batch isn't claimed by these slices
    student_led_proportion: float = Field(default=0.0, ge=0, le=1.0)
    teacher_led_proportion: float = Field(default=0.0, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "SlicedKickstarter":
        """Create Kickstarter loss instance."""
        return SlicedKickstarter(policy, trainer_cfg, vec_env, device, instance_name, self)


class SlicedKickstarter(NodeBase):
    """This uses another policy that is forwarded during rollout, here, in the loss and then compares its logits and
    value against the student's using a KL divergence and MSE loss respectively.
    """

    __slots__ = ("teacher_policy", "rollout_batch_size", "stud_mask", "teacher_mask", "ppo_mask")

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "SlicedKickstarterConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)
        self.teacher_policy = load_teacher_policy(self.env, policy_uri=self.cfg.teacher_uri, device=self.device)

    def get_experience_spec(self) -> Composite:
        # Get action space size for logits shape
        act_space = self.env.single_action_space
        num_actions = act_space.n

        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        logits_f32 = UnboundedContinuous(shape=torch.Size([num_actions]), dtype=torch.float32)
        boolean = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool)

        return Composite(
            rewards=scalar_f32,
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int32),
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

            if not hasattr(self, "rollout_batch_size") or self.rollout_batch_size != td.batch_size.numel():
                self._create_slices(td.batch_size.numel())

            self.policy.forward(td)

        td["stud_mask"] = self.stud_mask
        td["teacher_mask"] = self.teacher_mask
        td["ppo_mask"] = self.ppo_mask

        # Store experience
        env_slice = self._training_env_id(context)
        self.replay.store(data_td=td, env_id=env_slice)

        if self.teacher_mask.any():
            td["actions"][self.teacher_mask] = teacher_actions[self.teacher_mask]

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"logits", "values"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        student_td = shared_loss_data["policy_td"]

        # slice - minus teacher led minus student led
        train_stud_mask = minibatch["stud_mask"][:, 0]
        train_teacher_mask = minibatch["teacher_mask"][:, 0]
        train_ppo_mask = minibatch["ppo_mask"][:, 0]

        # cut down all of shared_loss_data to just the ppo mask before passing out to PPO losses
        shared_loss_data = shared_loss_data[train_ppo_mask]

        minibatch = minibatch[train_teacher_mask | train_stud_mask]
        student_td = student_td[train_teacher_mask | train_stud_mask]

        sliced_b, sliced_tt = minibatch.batch_size
        minibatch = minibatch.reshape(sliced_b * sliced_tt)
        student_td = student_td.reshape(sliced_b * sliced_tt)

        if minibatch.batch_size.numel() == 0 or student_td.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False

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
        ks_value_loss_vec = (teacher_value.detach() - student_value) ** 2
        ks_value_loss = ks_value_loss_vec.mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef
        shared_loss_data["ks_val_loss_vec"] = ks_value_loss_vec

        self.loss_tracker["ks_act_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["ks_val_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["ks_act_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["ks_val_loss_coef"].append(float(self.cfg.value_loss_coef))
        self.loss_tracker["ks_teacher_led_proportion"].append(float(self.cfg.teacher_led_proportion))
        self.loss_tracker["ks_student_led_proportion"].append(float(self.cfg.student_led_proportion))

        return loss, shared_loss_data, False

    def on_train_phase_end(self, context: ComponentContext | None = None) -> None:
        self._update_slices()
        super().on_train_phase_end(context)

    def _create_slices(self, B: int) -> None:
        self.rollout_batch_size = B

        rand_assignments = torch.rand(B, device=self.device)

        stud_threshold = self.cfg.student_led_proportion
        teacher_threshold = stud_threshold + self.cfg.teacher_led_proportion

        self.stud_mask = rand_assignments < stud_threshold
        self.teacher_mask = (rand_assignments >= stud_threshold) & (rand_assignments < teacher_threshold)
        self.ppo_mask = rand_assignments >= teacher_threshold

    def _update_slices(self) -> None:
        # we count on the hyperparmeter scheduler to update the cfg proportions
        self._create_slices(self.rollout_batch_size)


NODE_SPECS = [
    NodeSpec(
        key="sliced_kickstarter",
        config_cls=SlicedKickstarterConfig,
        default_enabled=False,
        has_rollout=True,
        has_train=True,
        writes_actions=True,
        produces_experience=True,
    )
]
