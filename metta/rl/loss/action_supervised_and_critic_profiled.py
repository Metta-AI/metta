"""Profiled version of action_supervised_and_critic for performance analysis.

Use this by changing the import in your recipe from:
    from metta.rl.loss.action_supervised_and_critic import ActionSupervisedAndCriticConfig
to:
    from metta.rl.loss.action_supervised_and_critic_profiled import ActionSupervisedAndCriticConfig
"""

import time
from collections import defaultdict
from typing import Any

import torch
from tensordict import TensorDict

from metta.rl.loss.action_supervised_and_critic import (
    ActionSupervisedAndCritic as _BaseActionSupervisedAndCritic,
)
from metta.rl.loss.action_supervised_and_critic import ActionSupervisedAndCriticConfig
from metta.rl.training.context import ComponentContext


class ActionSupervisedAndCritic(_BaseActionSupervisedAndCritic):
    """Profiled version that tracks timing of each operation and fixes memory leaks."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.timing_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.log_interval = 10  # Log every N rollouts

        # Pre-allocate buffers to avoid repeated allocations (memory leak fix)
        self._teacher_obs_buffer = None
        self._teacher_actions_buffer = None
        self._rollout_count = 0

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Rollout with detailed timing and memory leak fixes."""
        rollout_start = time.perf_counter()
        self._rollout_count += 1

        # Track memory at start
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
            self.memory_stats["gpu_allocated_mb"].append(mem_allocated)
            self.memory_stats["gpu_reserved_mb"].append(mem_reserved)

        # Student forward pass
        t0 = time.perf_counter()
        self.policy.reset_memory()
        from metta.agent.util.torch_utils import prepare_policy_forward_td

        policy_td, B, TT = prepare_policy_forward_td(td, self.policy_experience_spec, clone=False)
        policy_td = self.policy(policy_td, action=None)
        td.update(policy_td)
        student_time = time.perf_counter() - t0

        # Teacher forward pass
        t1 = time.perf_counter()
        with torch.no_grad():
            if self.is_scripted_teacher:
                # Breakdown Nim agent timing
                t_reshape_start = time.perf_counter()
                import numpy as np

                env_obs = td["env_obs"]
                num_agents = self.teacher_policy._num_agents
                batch_size = env_obs.shape[0]
                num_envs = batch_size // num_agents

                if batch_size % num_agents != 0:
                    raise ValueError(
                        f"Batch size {batch_size} is not divisible by num_agents {num_agents}. "
                        f"Expected batch_size = num_envs * num_agents"
                    )

                num_tokens = env_obs.shape[1]
                token_dim = env_obs.shape[2]

                # MEMORY LEAK FIX: Lazy allocate buffers once, reuse them
                expected_shape = (num_envs, num_agents, num_tokens, token_dim)
                if self._teacher_obs_buffer is None or self._teacher_obs_buffer.shape != expected_shape:
                    self._teacher_obs_buffer = np.zeros(expected_shape, dtype=np.uint8)
                    self._teacher_actions_buffer = np.zeros((num_envs, num_agents), dtype=np.int32)

                # Reshape and copy to pre-allocated buffer
                env_obs_reshaped = env_obs.reshape(num_envs, num_agents, num_tokens, token_dim)
                t_reshape = time.perf_counter() - t_reshape_start

                # GPU -> CPU transfer - copy directly to buffer
                t_transfer_start = time.perf_counter()
                np.copyto(self._teacher_obs_buffer, env_obs_reshaped.cpu().numpy(), casting="unsafe")
                t_transfer = time.perf_counter() - t_transfer_start

                # Clear the reshaped view to free memory
                del env_obs_reshaped

                # Nim agent calls - reuse action buffer
                t_nim_start = time.perf_counter()
                for env_idx in range(num_envs):
                    # Get view into pre-allocated buffer (no copy)
                    obs_for_env = self._teacher_obs_buffer[env_idx]

                    # Ensure contiguous (should already be)
                    if not obs_for_env.flags["C_CONTIGUOUS"]:
                        obs_for_env = np.ascontiguousarray(obs_for_env)
                        self._teacher_obs_buffer[env_idx] = obs_for_env

                    # Call directly into actions buffer (no append to list!)
                    self.teacher_policy.step_batch(obs_for_env, self._teacher_actions_buffer[env_idx])
                t_nim = time.perf_counter() - t_nim_start

                # Stack and convert back - use pre-allocated buffer
                t_stack_start = time.perf_counter()
                teacher_actions_flat = self._teacher_actions_buffer.reshape(batch_size)
                teacher_actions = torch.from_numpy(teacher_actions_flat.copy()).to(device=td.device, dtype=torch.long)
                td["teacher_actions"] = teacher_actions
                t_stack = time.perf_counter() - t_stack_start

                # Store breakdown
                self.timing_stats["teacher_reshape"].append(t_reshape)
                self.timing_stats["teacher_transfer_gpu_to_cpu"].append(t_transfer)
                self.timing_stats["teacher_nim_calls"].append(t_nim)
                self.timing_stats["teacher_stack_and_transfer"].append(t_stack)

            elif self.teacher_policy_spec is not None:
                from metta.agent.util.torch_utils import prepare_policy_forward_td

                teacher_td, B, TT = prepare_policy_forward_td(td, self.teacher_policy_spec, clone=True)
                teacher_td = self.teacher_policy(teacher_td, action=None)
                td["teacher_actions"] = teacher_td["actions"]
                # MEMORY LEAK FIX: Delete temporary tensordict
                del teacher_td
            else:
                td["teacher_actions"] = td["actions"].clone().to(torch.long)

        teacher_time = time.perf_counter() - t1

        # Store experience
        t2 = time.perf_counter()
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)
        store_time = time.perf_counter() - t2

        # Student-led logic
        if torch.rand(1) < self.cfg.teacher_lead_prob:
            td["actions"] = td["teacher_actions"]

        total_rollout_time = time.perf_counter() - rollout_start

        # Track timings
        self.timing_stats["rollout_total"].append(total_rollout_time)
        self.timing_stats["student_forward"].append(student_time)
        self.timing_stats["teacher_total"].append(teacher_time)
        self.timing_stats["replay_store"].append(store_time)

        # MEMORY LEAK FIX: Periodically clear GPU cache
        if torch.cuda.is_available() and self._rollout_count % 100 == 0:
            torch.cuda.empty_cache()

        # Log periodically
        if len(self.timing_stats["rollout_total"]) % self.log_interval == 0:
            self._log_timing_stats(context)

    def _log_timing_stats(self, context: ComponentContext) -> None:
        """Log averaged timing and memory statistics."""
        import logging

        logger = logging.getLogger(__name__)

        msg_parts = [
            f"\n{'=' * 60}",
            f"Rollout Profile (agent_step={context.agent_step}, rollout={self._rollout_count})",
            f"{'=' * 60}",
        ]

        def avg_ms(key: str) -> float:
            values = self.timing_stats[key]
            return (sum(values) / len(values)) * 1000 if values else 0.0

        def avg_val(key: str) -> float:
            values = self.memory_stats[key]
            return sum(values) / len(values) if values else 0.0

        # Timing stats
        msg_parts.append(f"  Rollout Total:        {avg_ms('rollout_total'):7.2f} ms")
        msg_parts.append(f"    - Student Forward:  {avg_ms('student_forward'):7.2f} ms")
        msg_parts.append(f"    - Teacher Total:    {avg_ms('teacher_total'):7.2f} ms")

        if self.timing_stats["teacher_reshape"]:
            msg_parts.append(f"        • Reshape:      {avg_ms('teacher_reshape'):7.2f} ms")
            msg_parts.append(f"        • GPU->CPU:     {avg_ms('teacher_transfer_gpu_to_cpu'):7.2f} ms")
            msg_parts.append(f"        • Nim Calls:    {avg_ms('teacher_nim_calls'):7.2f} ms")
            msg_parts.append(f"        • Stack+Back:   {avg_ms('teacher_stack_and_transfer'):7.2f} ms")

        msg_parts.append(f"    - Replay Store:     {avg_ms('replay_store'):7.2f} ms")

        # Memory stats
        if self.memory_stats.get("gpu_allocated_mb"):
            msg_parts.append("  GPU Memory:")
            msg_parts.append(f"    - Allocated:        {avg_val('gpu_allocated_mb'):7.1f} MB")
            msg_parts.append(f"    - Reserved:         {avg_val('gpu_reserved_mb'):7.1f} MB")

            # Check for memory growth
            first_alloc = self.memory_stats["gpu_allocated_mb"][0]
            last_alloc = self.memory_stats["gpu_allocated_mb"][-1]
            growth = last_alloc - first_alloc
            if abs(growth) > 10:  # More than 10MB growth
                msg_parts.append(f"    - Growth (⚠️):      {growth:+7.1f} MB (potential leak!)")

        msg_parts.append(f"{'=' * 60}\n")

        logger.info("\n".join(msg_parts))

        # Clear stats for next interval
        self.timing_stats.clear()
        self.memory_stats.clear()


__all__ = ["ActionSupervisedAndCriticConfig", "ActionSupervisedAndCritic"]
