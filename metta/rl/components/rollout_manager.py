"""Manages the rollout phase of RL training."""

import logging
from typing import Any, List, Tuple

import torch

from metta.common.profiling.stopwatch import Stopwatch
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.experience import Experience
from metta.rl.util.rollout import get_observation, run_policy_inference

logger = logging.getLogger(__name__)


class RolloutManager:
    """Manages collection of experience through environment rollouts."""

    def __init__(
        self,
        env: Any,
        device: torch.device,
        timer: Stopwatch,
    ):
        """Initialize rollout manager.

        Args:
            env: The vectorized environment
            device: Device to run computations on
            timer: Stopwatch for timing operations
        """
        self.env = env
        self.device = device
        self.timer = timer

    def collect_rollouts(
        self,
        agent: Any,
        experience: Experience,
        agent_step: int,
    ) -> Tuple[List[Any], int]:
        """Collect experience by running rollouts in the environment.

        Args:
            agent: The policy/agent to collect experience with
            experience: Experience buffer to store trajectories
            agent_step: Current training step

        Returns:
            Tuple of (raw_infos from environment, updated agent_step)
        """
        raw_infos = []
        experience.reset_for_rollout()

        while not experience.ready_for_training:
            # Receive environment data
            o, r, d, t, info, training_env_id, mask, num_steps = get_observation(self.env, self.device, self.timer)
            agent_step += num_steps

            # Run policy inference
            actions, selected_action_log_probs, values, lstm_state_to_store = run_policy_inference(
                agent, o, experience, training_env_id.start, self.device
            )

            # Store experience
            experience.store(
                obs=o,
                actions=actions,
                logprobs=selected_action_log_probs,
                rewards=r,
                dones=d,
                truncations=t,
                values=values,
                env_id=training_env_id,
                mask=mask,
                lstm_state=lstm_state_to_store,
            )

            # Send actions back to environment
            with self.timer("_rollout.env"):
                self.env.send(actions.cpu().numpy().astype(dtype_actions))

            if info:
                raw_infos.extend(info)

        return raw_infos, agent_step
