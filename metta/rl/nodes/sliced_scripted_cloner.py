from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.nodes.base import NodeBase, NodeConfig
from metta.rl.nodes.registry import NodeSpec
from metta.rl.training import ComponentContext

# Keep: heavy module + manages circular dependency (loss <-> trainer)
if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SlicedScriptedClonerConfig(NodeConfig):
    action_loss_coef: float = Field(default=1.0, ge=0, le=1.0)

    # PPO consumes whatever portion of the batch isn't claimed by these slices
    student_led_proportion: float = Field(default=0.0, ge=0, le=1.0)
    teacher_led_proportion: float = Field(default=0.0, ge=0, le=1.0)
    restrict_ppo_to_ppo_mask: bool = Field(
        default=True,
        description=(
            "If True (default), restrict downstream PPO losses to the PPO slice only. "
            "If False, downstream losses receive all slices so they can choose their own masking."
        ),
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "SlicedScriptedCloner":
        """Create SlicedScriptedCloner loss instance."""
        return SlicedScriptedCloner(policy, trainer_cfg, vec_env, device, instance_name, self)


class SlicedScriptedCloner(NodeBase):
    """This uses a scripted policy's actions (provided by the environment) to supervise the student
    on specific slices of the experience, similar to SlicedKickstarter but with Ground Truth actions.
    """

    __slots__ = ("rollout_batch_size", "stud_mask", "teacher_mask", "ppo_mask")

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "SlicedScriptedClonerConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)

    def get_experience_spec(self) -> Composite:
        teacher_actions = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long)
        actions = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int32)
        boolean = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool)
        rewards = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32)
        dones = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32)
        truncateds = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32)
        act_log_prob = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            teacher_actions=teacher_actions,
            actions=actions,
            act_log_prob=act_log_prob,
            stud_mask=boolean,
            teacher_mask=boolean,
            ppo_mask=boolean,
            rewards=rewards,
            dones=dones,
            truncateds=truncateds,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            if not hasattr(self, "rollout_batch_size") or self.rollout_batch_size != td.batch_size.numel():
                self._create_slices(td.batch_size.numel())

            self.policy.forward(td)

            if self.teacher_mask.any():
                # Align stored actions/logprobs with the teacher-led portion so PPO can learn from it.
                teacher_actions = td["teacher_actions"].to(dtype=torch.long)
                td["actions"][self.teacher_mask] = teacher_actions.to(td["actions"].dtype)[self.teacher_mask]
                # Teacher actions are produced by a scripted (deterministic) policy: treat behaviour prob as 1.0.
                td["act_log_prob"][self.teacher_mask] = 0.0

        td["stud_mask"] = self.stud_mask
        td["teacher_mask"] = self.teacher_mask
        td["ppo_mask"] = self.ppo_mask

        # Store experience
        env_slice = self._training_env_id(context)
        self.replay.store(data_td=td, env_id=env_slice)

        if self.teacher_mask.any():
            td["actions"][self.teacher_mask] = td["teacher_actions"].to(td["actions"].dtype)[self.teacher_mask]

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"full_log_probs"}

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
        shared_loss_data_for_downstream = shared_loss_data
        if self.cfg.restrict_ppo_to_ppo_mask:
            shared_loss_data_for_downstream = shared_loss_data[train_ppo_mask]

        train_slice_mask = train_teacher_mask | train_stud_mask
        slice_teacher_mask = train_teacher_mask[train_slice_mask]
        slice_student_mask = train_stud_mask[train_slice_mask]

        minibatch = minibatch[train_slice_mask]
        student_td = student_td[train_slice_mask]

        sliced_b, sliced_tt = minibatch.batch_size
        teacher_step_mask = slice_teacher_mask.unsqueeze(1).expand(sliced_b, sliced_tt).reshape(sliced_b * sliced_tt)
        student_step_mask = slice_student_mask.unsqueeze(1).expand(sliced_b, sliced_tt).reshape(sliced_b * sliced_tt)
        minibatch = minibatch.reshape(sliced_b * sliced_tt)
        student_td = student_td.reshape(sliced_b * sliced_tt)

        if minibatch.batch_size.numel() == 0 or student_td.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False

        # action loss
        policy_full_log_probs = student_td["full_log_probs"].reshape(sliced_b * sliced_tt, -1)
        teacher_actions = minibatch["teacher_actions"]

        # get the student's logprob for the action that the teacher chose
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0])

        loss = -student_log_probs.mean() * self.cfg.action_loss_coef
        if bool(teacher_step_mask.any()):
            teacher_loss = -student_log_probs[teacher_step_mask].mean() * self.cfg.action_loss_coef
            self.loss_tracker["supervised_action_loss_teacher_led"].append(float(teacher_loss.item()))
        if bool(student_step_mask.any()):
            student_loss = -student_log_probs[student_step_mask].mean() * self.cfg.action_loss_coef
            self.loss_tracker["supervised_action_loss_student_led"].append(float(student_loss.item()))

        self.loss_tracker["supervised_action_loss"].append(float(loss.item()))
        self.loss_tracker["supervised_action_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["teacher_led_proportion"].append(float(self.cfg.teacher_led_proportion))
        self.loss_tracker["student_led_proportion"].append(float(self.cfg.student_led_proportion))

        return loss, shared_loss_data_for_downstream, False

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
        key="sliced_scripted_cloner",
        config_cls=SlicedScriptedClonerConfig,
        default_enabled=False,
        has_rollout=True,
        has_train=True,
    )
]
