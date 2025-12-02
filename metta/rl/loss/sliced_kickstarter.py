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
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.mpt_policy import MptPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class SlicedKickstarterConfig(LossConfig):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.6, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    temperature: float = Field(default=2.0, gt=0)
    student_forward: bool = Field(default=True)  # probably always true for sliced_kickstarter

    # remainder of the sum below is left for the PPO loss to use
    student_led_proportion: float = Field(default=0.0, ge=0, le=1.0)
    teacher_led_proportion: float = Field(default=0.0, ge=0, le=1.0)

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
        "extended_policy_env_info",
        "student_feature_id",
        "teacher_feature_id",
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
        self.loss_profiles = None  # inherit default filtering (all)

        base_policy_env_info = getattr(self.env, "policy_env_info", None)
        if base_policy_env_info is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        # Extend the policy_env_info with an extra observation feature so the obs shim
        # can reserve a dedicated attribute index for the injected Kickstarter token.
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

            if not hasattr(self, "rollout_batch_size") or self.rollout_batch_size != td.batch_size.numel():
                self._create_slices(td.batch_size.numel())

            # Inject synthetic observation tokens for student-led and teacher-led
            # slices so the student policy can see which regime produced each step.
            self._inject_extra_obs_token(td)

            self.policy.forward(td)

        td["stud_mask"] = self.stud_mask
        td["teacher_mask"] = self.teacher_mask
        td["ppo_mask"] = self.ppo_mask

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

        if self.teacher_mask.any():
            td["actions"][self.teacher_mask] = teacher_actions[self.teacher_mask]

    def _build_extended_policy_env_info(self, policy_env_info: PolicyEnvInterface) -> PolicyEnvInterface:
        """Create a PolicyEnvInterface that includes extra features for KS tokens.

        The extra features use previously unused attribute indices so that
        ObsShimTokens can reserve clean slots for the injected observations.
        """
        existing_ids = {feat.id for feat in policy_env_info.obs_features}

        # Find two free ids in the uint8 range [0, 255].
        free_ids: list[int] = []
        for candidate in range(256):
            if candidate not in existing_ids:
                free_ids.append(candidate)
            if len(free_ids) >= 2:
                break

        if len(free_ids) < 2:
            raise ValueError("Not enough free observation feature IDs for SlicedKickstarter extra tokens.")

        self.student_feature_id, self.teacher_feature_id = free_ids[0], free_ids[1]

        student_feature = ObservationFeatureSpec(
            id=self.student_feature_id,
            name="student_led",
            normalization=1.0,
        )
        teacher_feature = ObservationFeatureSpec(
            id=self.teacher_feature_id,
            name="teacher_led",
            normalization=1.0,
        )

        extended_features = list(policy_env_info.obs_features) + [student_feature, teacher_feature]
        # Use model_copy to avoid mutating the original PolicyEnvInterface.
        return policy_env_info.model_copy(update={"obs_features": extended_features})

    def _inject_extra_obs_token(self, td: TensorDict) -> None:
        """Inject synthetic observation tokens at the start of selected sequences.

        For each batch element:
        - If stud_mask[b] is True, inject a token with attr_idx = student_feature_id.
        - If teacher_mask[b] is True, inject a token with attr_idx = teacher_feature_id.
        - Other batch elements are left unchanged.

        Injected tokens have:
        - coord byte = 0 (top-left tile)
        - attr_val   = 1

        To keep the observation shape consistent, we drop the last token in the
        sequence for modified rows and prepend the new token.
        """
        if "env_obs" not in td.keys():
            raise ValueError("TensorDict missing 'env_obs' key required for Kickstarter token injection.")

        env_obs = td["env_obs"]

        # Expect token observations of shape [B, M, 3] with uint8 dtype.
        if env_obs.dim() != 3 or env_obs.size(-1) != 3:
            raise ValueError(f"Expected 'env_obs' with shape [B, M, 3], got {env_obs.shape}")

        batch_size, num_tokens, token_dim = env_obs.shape
        if num_tokens == 0:
            raise ValueError("Encountered 'env_obs' with 0 tokens, cannot inject Kickstarter token.")

        # Ensure masks are available and match the current batch size.
        if not hasattr(self, "stud_mask") or not hasattr(self, "teacher_mask"):
            raise RuntimeError("Kickstarter masks not initialized. Has _create_slices been called?")

        if self.stud_mask.shape[0] != batch_size or self.teacher_mask.shape[0] != batch_size:
            raise RuntimeError(
                f"Batch size mismatch: env_obs={batch_size}, "
                f"stud_mask={self.stud_mask.shape[0]}, teacher_mask={self.teacher_mask.shape[0]}"
            )

        active_mask = self.stud_mask | self.teacher_mask
        if not torch.any(active_mask):
            return

        device = env_obs.device
        dtype = env_obs.dtype

        # Determine attribute IDs for active rows.
        attr_ids = torch.zeros(batch_size, device=device, dtype=dtype)
        attr_ids[self.stud_mask] = torch.as_tensor(self.student_feature_id, device=device, dtype=dtype)
        attr_ids[self.teacher_mask] = torch.as_tensor(self.teacher_feature_id, device=device, dtype=dtype)

        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        if active_indices.numel() == 0:
            return

        # Prepare new env_obs with injections for active rows only.
        new_env_obs = env_obs.clone()

        trimmed_env_obs = env_obs[active_indices, :-1, :]
        extra_token = torch.zeros((active_indices.numel(), 1, token_dim), device=device, dtype=dtype)
        extra_token[..., 0] = 0  # coord byte
        extra_token[..., 1] = attr_ids[active_indices].view(-1, 1).to(dtype=dtype)
        extra_token[..., 2] = 1  # raw attribute value

        injected = torch.cat((extra_token, trimmed_env_obs), dim=1)
        new_env_obs[active_indices] = injected

        td["env_obs"] = new_env_obs

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

        # sliced kickstarter MUST run first since it decides what to pass to PPO
        student_td, B, TT = prepare_policy_forward_td(minibatch, self.policy_experience_spec, clone=False)
        flat_actions = minibatch["actions"].reshape(B * TT, -1)
        self.policy.reset_memory()
        student_td = self.policy.forward(student_td, action=flat_actions)
        student_td = student_td.reshape(B, TT)
        shared_loss_data["policy_td"] = student_td[train_ppo_mask]  # this is for passing to PPO losses

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
        ks_value_loss = ((teacher_value.detach() - student_value) ** 2).mean()

        loss = ks_action_loss * self.cfg.action_loss_coef + ks_value_loss * self.cfg.value_loss_coef

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
        assert self.cfg.student_led_proportion + self.cfg.teacher_led_proportion <= 1.0, "Proportions must sum <= 1.0"
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
