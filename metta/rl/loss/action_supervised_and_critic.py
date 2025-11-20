"""Action supervised learning with critic training.

This loss supports supervised learning from a teacher policy (e.g., scripted agent)
while also training the critic with Bellman updates. The student policy is always
in the lead (student actions go to the environment).

Key features:
- Teacher policy inference on same states as student
- Cross-entropy loss for action supervision
- Critic training with GAE and Bellman updates
- Student-led rollouts (configurable via teacher_lead_prob)
- Smooth transition to pure RL (use with scheduler gates)
"""

from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.replay_samplers import sequential_sample
from metta.rl.training import ComponentContext
from metta.rl.utils import prepare_policy_forward_td
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.util.module import load_symbol


class ActionSupervisedAndCriticConfig(LossConfig):
    """Configuration for action supervised learning with critic training."""

    # Action supervision parameters
    action_loss_coef: float = Field(default=1.0, ge=0, description="Coefficient for action supervision loss")
    sample_enabled: bool = Field(default=True, description="Sequentially sample from buffer during train")
    rollout_forward_enabled: bool = Field(default=True, description="Forward policy during rollout")
    train_forward_enabled: bool = Field(default=True, description="Forward policy during training")
    teacher_lead_prob: float = Field(
        default=0.0, ge=0, le=1.0, description="Probability of using teacher actions (0.0 = student-led)"
    )

    # Optional reward shaping
    add_action_loss_to_rewards: bool = Field(default=False, description="Add imitation loss to rewards")
    action_reward_coef: float = Field(default=0.01, ge=0, description="Coefficient for action reward shaping")

    # Teacher policy configuration
    teacher_class_path: str = Field(
        default="cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
        description="Class path to teacher policy (e.g., Nim Thinky agent)",
    )
    teacher_uri: str | None = Field(default=None, description="Optional URI for loading trained teacher weights")

    # Critic training parameters (for Bellman updates)
    gamma: float = Field(default=0.977, ge=0, le=1.0, description="Discount factor")
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0, description="GAE lambda")
    vf_coef: float = Field(default=0.897619, ge=0, description="Value function loss coefficient")
    vf_clip_coef: float = Field(default=0.1, ge=0, description="Value clipping coefficient")
    clip_vloss: bool = Field(default=True, description="Whether to clip value loss")

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "ActionSupervisedAndCritic":
        """Create ActionSupervisedAndCritic loss instance."""
        return ActionSupervisedAndCritic(
            policy, trainer_cfg, vec_env, device, instance_name=instance_name, loss_config=loss_config
        )


class ActionSupervisedAndCritic(Loss):
    """Action supervised learning with critic training.

    This loss trains a student policy to imitate a teacher policy using cross-entropy loss
    on actions, while simultaneously training the critic using Bellman updates with actual
    environment rewards.
    """

    __slots__ = (
        "rollout_forward_enabled",
        "train_forward_enabled",
        "sample_enabled",
        "teacher_policy",
        "teacher_policy_spec",
        "advantages",
        "is_scripted_teacher",
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
        # Get loss config from trainer_cfg if not provided
        if loss_config is None:
            loss_config = getattr(trainer_cfg.losses, instance_name, None)
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)

        # Unpack config into slots
        self.rollout_forward_enabled = self.cfg.rollout_forward_enabled
        self.train_forward_enabled = self.cfg.train_forward_enabled
        self.sample_enabled = self.cfg.sample_enabled

        # Load teacher policy
        policy_env_info = getattr(self.env, "policy_env_info", None)
        if policy_env_info is None:
            raise RuntimeError("Environment metadata is required to instantiate teacher policy")

        # Load teacher policy
        self.teacher_policy = None
        self.teacher_policy_spec = None
        self.is_scripted_teacher = False

        if self.cfg.teacher_uri is not None:
            # Load trained policy from URI
            import logging

            from metta.rl.checkpoint_manager import CheckpointManager

            logging.info(f"Loading teacher policy from URI: {self.cfg.teacher_uri}")
            teacher_spec = CheckpointManager.policy_spec_from_uri(self.cfg.teacher_uri, device=self.device)
            self.teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec)

            # Get experience spec for trainable teacher
            if hasattr(self.teacher_policy, "get_agent_experience_spec"):
                self.teacher_policy_spec = self.teacher_policy.get_agent_experience_spec()

            # Detach gradient for teacher (teacher is frozen)
            if hasattr(self.teacher_policy, "parameters"):
                for param in self.teacher_policy.parameters():
                    param.requires_grad = False
        else:
            # Load scripted teacher from class path (e.g., Nim Thinky)
            import logging

            logging.info(f"Loading scripted teacher policy: {self.cfg.teacher_class_path}")
            TeacherClass = load_symbol(self.cfg.teacher_class_path)
            self.teacher_policy = TeacherClass(policy_env_info)
            self.is_scripted_teacher = True
            logging.info(f"Loaded scripted teacher: {self.cfg.teacher_class_path}")

        # Initialize advantages tensor for critic training
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Pre-allocate buffers to avoid memory leaks from repeated allocations
        self._teacher_obs_buffer = None
        self._teacher_actions_buffer = None
        self._rollout_count = 0

        # Register state attributes for checkpointing
        self.register_state_attr("advantages")

    def get_experience_spec(self) -> Composite:
        """Experience specification including teacher actions and values for critic."""
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long),
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            values=scalar_f32,  # Needed for critic training
            act_log_prob=scalar_f32,  # Needed for optional reward shaping
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Run rollout: student forward pass + teacher inference."""
        if not self.rollout_forward_enabled:
            return

        self._rollout_count += 1

        # Student forward pass
        with torch.no_grad():
            self.policy.forward(td)

        # Teacher forward pass
        with torch.no_grad():
            if self.is_scripted_teacher:
                # Scripted agent (e.g., Nim Thinky): use raw observations
                import numpy as np

                env_obs = td["env_obs"]  # Shape: (batch_size, num_tokens, token_dim)
                # batch_size = num_envs * num_agents (agents are flattened into batch)

                # Get the actual number of agents from the teacher policy
                num_agents = self.teacher_policy._num_agents

                # The batch dimension contains all agents from all environments flattened
                batch_size = env_obs.shape[0]
                num_envs = batch_size // num_agents

                if batch_size % num_agents != 0:
                    raise ValueError(
                        f"Batch size {batch_size} is not divisible by num_agents {num_agents}. "
                        f"Expected batch_size = num_envs * num_agents"
                    )

                # Reshape to (num_envs, num_agents, num_tokens, token_dim)
                num_tokens = env_obs.shape[1]
                token_dim = env_obs.shape[2]

                # MEMORY LEAK FIX: Lazy allocate buffers once, reuse them
                expected_shape = (num_envs, num_agents, num_tokens, token_dim)
                if self._teacher_obs_buffer is None or self._teacher_obs_buffer.shape != expected_shape:
                    self._teacher_obs_buffer = np.zeros(expected_shape, dtype=np.uint8)
                    self._teacher_actions_buffer = np.zeros((num_envs, num_agents), dtype=np.int32)

                # Reshape and copy to pre-allocated buffer
                env_obs_reshaped = env_obs.reshape(num_envs, num_agents, num_tokens, token_dim)

                # GPU -> CPU transfer - copy directly to buffer
                np.copyto(self._teacher_obs_buffer, env_obs_reshaped.cpu().numpy(), casting="unsafe")

                # Clear the reshaped view to free memory
                del env_obs_reshaped

                # Process each environment separately - reuse action buffer
                for env_idx in range(num_envs):
                    # Get view into pre-allocated buffer (no copy)
                    obs_for_env = self._teacher_obs_buffer[env_idx]

                    # Ensure contiguous (should already be)
                    if not obs_for_env.flags["C_CONTIGUOUS"]:
                        obs_for_env = np.ascontiguousarray(obs_for_env)
                        self._teacher_obs_buffer[env_idx] = obs_for_env

                    # Call directly into actions buffer (no append to list!)
                    self.teacher_policy.step_batch(obs_for_env, self._teacher_actions_buffer[env_idx])

                # Flatten and convert to torch - copy from buffer
                teacher_actions_flat = self._teacher_actions_buffer.reshape(batch_size)
                teacher_actions = torch.from_numpy(teacher_actions_flat.copy()).to(device=td.device, dtype=torch.long)

                td["teacher_actions"] = teacher_actions

            elif self.teacher_policy_spec is not None:
                # Trainable teacher policy: use tensordict interface
                teacher_td, B, TT = prepare_policy_forward_td(td, self.teacher_policy_spec, clone=True)
                teacher_td = self.teacher_policy(teacher_td, action=None)
                td["teacher_actions"] = teacher_td["actions"]
                # MEMORY LEAK FIX: Delete temporary tensordict
                del teacher_td
            else:
                # Fallback: use student actions (no supervision)
                td["teacher_actions"] = td["actions"].clone().to(torch.long)

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)

        # Student-led: student actions always go to environment (unless teacher_lead_prob > 0)
        if torch.rand(1) < self.cfg.teacher_lead_prob:
            # Save td["action"] into replay buffer but overwrite with teacher actions for environment
            # NOTE: teacher-leading means actions reported to wandb are teacher actions, not student actions
            td["actions"] = td["teacher_actions"]

        # MEMORY LEAK FIX: Periodically clear GPU cache
        if torch.cuda.is_available() and self._rollout_count % 100 == 0:
            torch.cuda.empty_cache()

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Training step: compute action supervision loss + critic loss."""

        # Sample from buffer
        if self.sample_enabled:
            minibatch, indices = sequential_sample(self.replay, mb_idx)
            shared_loss_data["sampled_mb"] = minibatch
            shared_loss_data["indices"] = NonTensorData(indices)

        minibatch = shared_loss_data["sampled_mb"]

        # Forward pass
        if self.train_forward_enabled:
            policy_td, B, TT = prepare_policy_forward_td(minibatch, self.policy_experience_spec, clone=False)
            flat_actions = minibatch["actions"].reshape(B * TT, -1)
            self.policy.reset_memory()
            policy_td = self.policy.forward(policy_td, action=flat_actions)
            policy_td = policy_td.reshape(B, TT)
            shared_loss_data["policy_td"] = policy_td
        else:
            policy_td = shared_loss_data["policy_td"]

        # ============ Action Supervision Loss ============
        policy_full_log_probs = policy_td["full_log_probs"].reshape(minibatch.shape[0], minibatch.shape[1], -1)
        teacher_actions = minibatch["teacher_actions"]
        # Get the student's log prob for the action that the teacher chose
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0], minibatch.shape[1])

        action_loss = -student_log_probs.mean() * self.cfg.action_loss_coef

        # ============ Critic Training with Bellman Updates ============
        # Compute advantages on first minibatch
        if mb_idx == 0:
            with torch.no_grad():
                advantages = torch.zeros_like(self.replay.buffer["values"], device=self.device)
                advantages = compute_advantage(
                    self.replay.buffer["values"],
                    self.replay.buffer["rewards"],
                    self.replay.buffer["dones"],
                    torch.ones_like(self.replay.buffer["values"]),  # importance ratio = 1 (on-policy)
                    advantages,
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                    1.0,  # rho_clip (no V-trace correction needed for supervised learning)
                    1.0,  # c_clip
                    self.device,
                )
                self.advantages = advantages

        # Compute value loss
        indices = shared_loss_data["indices"]
        newvalue = policy_td["values"].view(minibatch["values"].shape)
        returns = self.advantages[indices] + minibatch["values"]
        old_values = minibatch["values"]

        if self.cfg.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = old_values + torch.clamp(
                newvalue - old_values,
                -self.cfg.vf_clip_coef,
                self.cfg.vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        # ============ Combined Loss ============
        loss = action_loss + value_loss * self.cfg.vf_coef

        # Track metrics
        self.loss_tracker["supervised_action_loss"].append(float(action_loss.item()))
        self.loss_tracker["supervised_value_loss"].append(float(value_loss.item()))
        self.loss_tracker["action_loss_coef"].append(float(self.cfg.action_loss_coef))
        self.loss_tracker["value_coef"].append(float(self.cfg.vf_coef))

        # Optional: Add action loss to rewards (reward shaping)
        if self.cfg.add_action_loss_to_rewards:
            minibatch["rewards"] = (
                minibatch["rewards"] + self.cfg.action_reward_coef * policy_td["act_log_prob"].detach()
            )

        return loss, shared_loss_data, False
