from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.nodes.base import NodeBase, NodeConfig
from metta.rl.nodes.registry import NodeSpec
from metta.rl.nodes.teacher_policy import load_teacher_policy
from metta.rl.training import ComponentContext
from metta.rl.utils import prepare_policy_forward_td
from mettagrid.util.uri_resolvers.schemes import (
    checkpoint_filename,
    parse_uri,
)

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SLCheckpointedKickstarterConfig(NodeConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)

    # Checkpoint reloading parameters
    checkpointed_interval: int = Field(default=1, gt=0, description="Interval at which teacher checkpoints are saved")
    epochs_per_checkpoint: int = Field(default=1, gt=0, description="Number of epochs to train with each checkpoint")
    terminating_epoch: int = Field(default=0, ge=0, description="Stop reloading checkpoints before this epoch")
    final_checkpoint: int = Field(default=0, ge=0, description="Final checkpoint to use (can be beyond terminating)")

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "SLCheckpointedKickstarter":
        """Create SLCheckpointedKickstarter loss instance."""
        return SLCheckpointedKickstarter(policy, trainer_cfg, vec_env, device, instance_name, self)


class SLCheckpointedKickstarter(NodeBase):
    """This is currently only student-led. No blockers to make it teacher-led, but not needed yet.
    It should be better at avoiding student-led curriculum hacking since we keep changing the teacher."""

    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "temperature",
        "teacher_uri",
        "_base_teacher_uri",
        "_checkpointed_interval",
        "_epochs_per_checkpoint",
        "_terminating_epoch",
        "_final_checkpoint",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "SLCheckpointedKickstarterConfig",
    ) -> None:
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)
        self.temperature = self.cfg.temperature
        self.teacher_uri = self.cfg.teacher_uri
        self._base_teacher_uri = self.cfg.teacher_uri  # Store original URI for checkpoint reloading
        self._checkpointed_interval = self.cfg.checkpointed_interval
        self._epochs_per_checkpoint = self.cfg.epochs_per_checkpoint
        self._terminating_epoch = self.cfg.terminating_epoch
        self._final_checkpoint = self.cfg.final_checkpoint
        self.teacher_policy = load_teacher_policy(self.env, policy_uri=self.cfg.teacher_uri, device=self.device)

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            if "actions" in td.keys():
                self.policy.forward(td, action=td["actions"])
            else:
                self.policy.forward(td)

        env_slice = self._training_env_id(
            context, error="ComponentContext.training_env_id is required for SLCheckpointedKickstarter rollout"
        )
        self.replay.store(data_td=td, env_id=env_slice)

    def get_experience_spec(self) -> Composite:
        return self.teacher_policy_spec

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"logits", "values"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        if context.epoch % self._epochs_per_checkpoint == 0:
            epoch = (context.epoch // self._epochs_per_checkpoint + 1) * self._checkpointed_interval
            self.load_teacher_policy(epoch)
        elif context.epoch == self._terminating_epoch:
            self.load_teacher_policy(self._final_checkpoint)

        minibatch = shared_loss_data["sampled_mb"]

        # Teacher forward pass
        teacher_td, B, TT = prepare_policy_forward_td(minibatch, self.teacher_policy_spec, clone=True)
        teacher_td = self.teacher_policy(teacher_td, action=None)

        student_td = shared_loss_data["policy_td"].reshape(B * TT)

        temperature = self.temperature
        teacher_logits = teacher_td["logits"].to(dtype=torch.float32)
        student_logits = student_td["logits"].to(dtype=torch.float32)

        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_probs = torch.exp(student_log_probs)

        ks_action_loss = (temperature**2) * (
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        )

        # Value loss
        student_value = student_td["values"].to(dtype=torch.float32)
        teacher_value = teacher_td["values"].to(dtype=torch.float32).detach()
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

        self.loss_tracker["sl_ks_action_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["sl_ks_value_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["sl_action_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["sl_value_loss_coef"].append(float(self.cfg.value_loss_coef))

        return loss, shared_loss_data, False

    def _construct_checkpoint_uri(self, epoch: int) -> str:
        """Construct a checkpoint URI from the base URI and epoch."""
        parsed = parse_uri(self._base_teacher_uri, allow_none=False)
        info = parsed.checkpoint_info
        if info is None:
            raise ValueError(f"Could not extract metadata from base URI: {self._base_teacher_uri}")
        run_name, _ = info
        filename = checkpoint_filename(run_name, epoch)

        if parsed.scheme == "file" and parsed.local_path:
            path = parsed.local_path.parent / filename
            return f"file://{path}"
        elif parsed.scheme == "s3" and parsed.bucket and parsed.key:
            if "/" in parsed.key:
                key_dir = parsed.key.rsplit("/", 1)[0]
                new_key = f"{key_dir}/{filename}"
            else:
                new_key = filename
            return f"s3://{parsed.bucket}/{new_key}"
        else:
            raise ValueError(f"Unsupported URI scheme for checkpoint reloading: {parsed.scheme}")

    def load_teacher_policy(self, checkpointed_epoch: int) -> None:
        """Load the teacher policy from a specific checkpoint."""
        new_uri = self._construct_checkpoint_uri(checkpointed_epoch)
        self.teacher_policy = load_teacher_policy(
            self.env,
            policy_uri=new_uri,
            device=self.device,
            error="Environment metadata is required to reload teacher policy",
        )

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False


NODE_SPECS = [
    NodeSpec(
        key="sl_checkpointed_kickstarter",
        config_cls=SLCheckpointedKickstarterConfig,
        default_enabled=False,
        has_rollout=True,
        has_train=True,
        writes_actions=True,
        produces_experience=True,
    )
]
