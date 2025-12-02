from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext
from metta.rl.utils import prepare_policy_forward_td
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.mpt_policy import MptPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class LogitKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)
    student_forward: bool = Field(default=False)  # use this if you need to forward student during train (eg if no PPO)
    teacher_lead_prob: float = Field(default=1.0, ge=0, le=1.0)  # at 0.0, it's purely student-led
    logit_noise_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    logit_noise_std: float = Field(default=1.0, ge=0.0)
    logit_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "LogitKickstarter":
        """Create LogitKickstarter loss instance."""
        return LogitKickstarter(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class LogitKickstarter(Loss):
    """This uses another policy that is forwarded during rollout, here, in the loss and then compares its logits and
    value against the student's using a KL divergence and MSE loss respectively.
    It also injects the teacher's logits into the student's observations.
    """

    __slots__ = (
        "teacher_policy",
        "student_forward",
        "extended_policy_env_info",
        "logit_feature_ids",
        "num_actions",
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

        base_policy_env_info = getattr(self.env, "policy_env_info", None)
        if base_policy_env_info is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        # Determine action space size
        act_space = self.env.single_action_space
        self.num_actions = int(act_space.n)

        # Extend the policy_env_info with extra observation features for logits
        self.extended_policy_env_info = self._build_extended_policy_env_info(base_policy_env_info)

        # Re-initialize the student policy to use the extended observation features.
        # This will, in particular, re-run ObsShimTokens.initialize_to_environment with
        # the updated feature list.
        if hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(self.extended_policy_env_info, self.device)

        # Initialize the teacher policy using the same extended env info so that its
        # obs encoder also understands the extra feature.
        teacher_spec = policy_spec_from_uri(self.cfg.teacher_uri, device=self.device)
        self.teacher_policy = initialize_or_load_policy(self.extended_policy_env_info, teacher_spec)
        if isinstance(self.teacher_policy, MptPolicy):
            self.teacher_policy = self.teacher_policy._policy

    def get_experience_spec(self) -> Composite:
        # Get action space size for logits shape
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        logits_f32 = UnboundedContinuous(shape=torch.Size([self.num_actions]), dtype=torch.float32)

        return Composite(
            teacher_logits=logits_f32,
            teacher_values=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            teacher_td = td.clone()
            self.teacher_policy.forward(teacher_td)
            teacher_actions = teacher_td["actions"]
            td["teacher_logits"] = teacher_td["logits"]
            td["teacher_values"] = teacher_td["values"]

            # Inject logits into obs
            self._inject_logit_tokens(td, teacher_td["logits"])

            self.policy.forward(td)

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

        if torch.rand(1) < self.cfg.teacher_lead_prob:
            # overwrite student actions w teacher actions with some probability. anneal this.
            td["actions"] = teacher_actions

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        B, TT = minibatch.batch_size

        # Student forward pass
        if self.student_forward:  # leave to false if also running PPO since it forwards student during train
            student_td, _, _ = prepare_policy_forward_td(minibatch, self.policy.get_agent_experience_spec(), clone=True)
            student_td = self.policy(student_td, action=None)
        else:
            student_td = shared_loss_data["policy_td"].reshape(B * TT)  # shared_loss_data is populated by PPO

        # action loss
        temperature = self.cfg.temperature
        teacher_logits = minibatch["teacher_logits"].to(dtype=torch.float32).reshape(B * TT, -1).detach()
        student_logits = student_td["logits"].to(dtype=torch.float32)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_probs = torch.exp(student_log_probs)
        ks_action_loss = (temperature**2) * (
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        )

        # value loss
        teacher_value = minibatch["teacher_values"].to(dtype=torch.float32).reshape(B * TT).detach()
        student_value = student_td["values"].to(dtype=torch.float32)
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

        self.loss_tracker["ks_act_loss"].append(float(ks_action_loss.item()))
        self.loss_tracker["ks_val_loss"].append(float(ks_value_loss.item()))
        self.loss_tracker["ks_act_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["ks_val_loss_coef"].append(float(self.cfg.value_loss_coef))

        return loss, shared_loss_data, False

    def _build_extended_policy_env_info(self, policy_env_info: PolicyEnvInterface) -> PolicyEnvInterface:
        """Create a PolicyEnvInterface that includes extra features for KS tokens."""
        existing_ids = {feat.id for feat in policy_env_info.obs_features}

        # Find enough free ids in the uint8 range [0, 255].
        free_ids: list[int] = []
        for candidate in range(256):
            if candidate not in existing_ids:
                free_ids.append(candidate)
            if len(free_ids) >= self.num_actions:
                break

        if len(free_ids) < self.num_actions:
            raise ValueError(
                f"Not enough free observation feature IDs for LogitKickstarter. "
                f"Needed {self.num_actions}, found {len(free_ids)}."
            )

        self.logit_feature_ids = free_ids

        new_features = []
        for i, fid in enumerate(free_ids):
            new_features.append(
                ObservationFeatureSpec(
                    id=fid,
                    name=f"teacher_logit_{i}",
                    normalization=1.0,
                )
            )

        extended_features = list(policy_env_info.obs_features) + new_features
        # Use model_copy to avoid mutating the original PolicyEnvInterface.
        return policy_env_info.model_copy(update={"obs_features": extended_features})

    def _inject_logit_tokens(self, td: TensorDict, logits: Tensor) -> None:
        """Inject teacher logits as tokens into the observation.

        Handles dropout and noise injection based on config.
        """
        if "env_obs" not in td.keys():
            return

        env_obs = td["env_obs"]

        # Expect token observations of shape [B, M, 3].
        if env_obs.dim() != 3 or env_obs.size(-1) != 3:
            raise ValueError(f"Expected env_obs of shape [B, M, 3], got {env_obs.shape}")

        batch_size, num_tokens, token_dim = env_obs.shape
        if num_tokens < self.num_actions:
            return

        device = env_obs.device
        dtype = env_obs.dtype

        # Check dropout
        if self.cfg.logit_dropout_prob > 0.0:
            dropout_mask = torch.rand(batch_size, device=device) < self.cfg.logit_dropout_prob
            # If all dropped out, return early (optimization)
            if dropout_mask.all():
                return
            active_mask = ~dropout_mask
        else:
            active_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        # If no active items (e.g. due to dropout), return
        if not active_mask.any():
            return

        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        num_active = active_indices.numel()

        # Create new tokens for active items: [num_active, num_actions, 3]
        new_tokens = torch.zeros((num_active, self.num_actions, token_dim), device=device, dtype=dtype)

        # Coord: 0
        new_tokens[..., 0] = 0

        # Attr IDs
        attr_ids_tensor = torch.tensor(self.logit_feature_ids, device=device, dtype=dtype)
        # Broadcast to active batch size
        new_tokens[..., 1] = attr_ids_tensor.unsqueeze(0).expand(num_active, -1)

        # Values: logits
        active_logits = logits[active_indices].to(dtype=dtype)

        # Apply noise if configured
        if self.cfg.logit_noise_prob > 0.0:
            noise_mask = torch.rand(num_active, device=device) < self.cfg.logit_noise_prob
            if noise_mask.any():
                noise = torch.randn_like(active_logits[noise_mask]) * self.cfg.logit_noise_std
                active_logits[noise_mask] += noise

        new_tokens[..., 2] = active_logits

        # Prepare new env_obs with injections for active rows only
        new_env_obs = env_obs.clone()

        trimmed_obs = env_obs[active_indices, : -self.num_actions, :]
        injected = torch.cat((new_tokens, trimmed_obs), dim=1)

        new_env_obs[active_indices] = injected

        td["env_obs"] = new_env_obs
