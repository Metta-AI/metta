from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.policy_artifact import policy_spec_from_uri
from metta.rl.training import ComponentContext
from metta.rl.utils import prepare_policy_forward_td
from mettagrid.policy.loader import initialize_or_load_policy

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SLCheckpointedKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)
    student_forward: bool = Field(default=False)

    # Checkpoint reloading parameters
    checkpointed_interval: int = Field(gt=0, description="Interval at which teacher checkpoints are saved")
    epochs_per_checkpoint: int = Field(gt=0, description="Number of epochs to train with each checkpoint")
    terminating_epoch: int = Field(ge=0, description="Stop reloading checkpoints before this epoch")
    final_checkpoint: int = Field(ge=0, description="Final checkpoint to use (can be beyond terminating)")

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "SLCheckpointedKickstarter":
        """Create SLCheckpointedKickstarter loss instance."""
        return SLCheckpointedKickstarter(
            policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config
        )


class SLCheckpointedKickstarter(Loss):
    __slots__ = (
        "teacher_policy",
        "teacher_policy_spec",
        "temperature",
        "student_forward",
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
        loss_config: Any = None,
    ) -> None:
        # Get loss config from trainer_cfg if not provided
        if loss_config is None:
            loss_config = getattr(trainer_cfg.losses, instance_name, None)
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.temperature = self.cfg.temperature
        self.student_forward = self.cfg.student_forward
        self.teacher_uri = self.cfg.teacher_uri
        self._base_teacher_uri = self.cfg.teacher_uri  # Store original URI for checkpoint reloading
        self._checkpointed_interval = self.cfg.checkpointed_interval
        self._epochs_per_checkpoint = self.cfg.epochs_per_checkpoint
        self._terminating_epoch = self.cfg.terminating_epoch
        self._final_checkpoint = self.cfg.final_checkpoint

        policy_env_info = getattr(self.env, "policy_env_info", None)
        if policy_env_info is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        teacher_spec = policy_spec_from_uri(self.cfg.teacher_uri, device=self.device)
        self.teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec)

        self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False

    def get_experience_spec(self) -> Composite:
        return self.teacher_policy_spec

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

        # Student forward pass
        if self.student_forward:
            student_td, B, TT = prepare_policy_forward_td(
                minibatch, self.policy.get_agent_experience_spec(), clone=True
            )
            student_td = self.policy(student_td, action=None)
        else:
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
        from metta.rl.checkpoint_manager import key_and_version
        from metta.utils.uri import ParsedURI

        # Parse the base URI
        parsed = ParsedURI.parse(self._base_teacher_uri)
        metadata = key_and_version(self._base_teacher_uri)
        if metadata is None:
            raise ValueError(f"Could not extract metadata from base URI: {self._base_teacher_uri}")
        run_name, _ = metadata

        # Construct new URI with the specified epoch
        if parsed.scheme == "file" and parsed.local_path:
            # For file URIs, replace the filename
            path = parsed.local_path.parent / f"{run_name}:v{epoch}.mpt"
            return f"file://{path}"
        elif parsed.scheme == "s3" and parsed.bucket and parsed.key:
            # For S3 URIs, replace the filename in the key
            # The key is the full path including the filename
            if "/" in parsed.key:
                key_dir = parsed.key.rsplit("/", 1)[0]
                new_key = f"{key_dir}/{run_name}:v{epoch}.mpt"
            else:
                # If key has no directory, just use the filename
                new_key = f"{run_name}:v{epoch}.mpt"
            return f"s3://{parsed.bucket}/{new_key}"
        else:
            raise ValueError(f"Unsupported URI scheme for checkpoint reloading: {parsed.scheme}")

    def load_teacher_policy(self, checkpointed_epoch: Optional[int] = None) -> Policy:
        """Load the teacher policy from a specific checkpoint."""
        new_uri = self._construct_checkpoint_uri(checkpointed_epoch)
        policy_env_info = getattr(self.env, "policy_env_info", None)
        if policy_env_info is None:
            raise RuntimeError("Environment metadata is required to reload teacher policy")

        teacher_spec = policy_spec_from_uri(new_uri, device=self.device)
        self.teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec)

        # Detach gradient
        for param in self.teacher_policy.parameters():
            param.requires_grad = False
