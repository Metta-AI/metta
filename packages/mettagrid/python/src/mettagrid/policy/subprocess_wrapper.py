"""Subprocess wrapper for AgentPolicy to enable parallel execution."""

from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.policy.policy import AgentPolicy
from mettagrid.simulator.interface import Action, AgentObservation, ObservationToken

logger = logging.getLogger(__name__)


def _agent_step_worker(policy_data: bytes, obs_data: tuple) -> tuple[str, float]:
    """
    Worker function for subprocess execution of agent policy step.

    Args:
        policy_data: Pickled AgentPolicy instance
        obs_data: Tuple of (agent_id, observation_tokens_dict) for AgentObservation

    Returns:
        Tuple of (action_name, computation_time_ms)
    """

    from mettagrid.simulator.interface import AgentObservation

    agent_id, observation_tokens_dict = obs_data

    # Reconstruct observation tokens
    tokens = []
    for token_dict in observation_tokens_dict:
        feature = ObservationFeatureSpec(
            id=token_dict["feature_id"],
            name=token_dict["feature_name"],
            normalization=token_dict["feature_normalization"],
        )
        tokens.append(
            ObservationToken(
                feature=feature,
                location=token_dict["location"],
                value=token_dict["value"],
                raw_token=token_dict["raw_token"],
            )
        )
    obs = AgentObservation(agent_id=agent_id, tokens=tokens)

    # Reconstruct policy
    try:
        policy = pickle.loads(policy_data)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to reconstruct policy for agent {agent_id}: {e}")
        return ("noop", 0.0)

    # Execute policy step
    start_time = time.time()
    try:
        action = policy.step(obs)
        action_name = action.name if hasattr(action, "name") else str(action)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Policy step failed for agent {agent_id}: {e}")
        action_name = "noop"
    end_time = time.time()
    computation_time_ms = (end_time - start_time) * 1000.0

    return (action_name, computation_time_ms)


class PerAgentSubprocessWrapper(AgentPolicy):
    """Wrapper for AgentPolicy that executes policy steps in a subprocess.

    This wrapper enables parallel execution of agent policies by running
    each policy.step() call in a separate subprocess. The wrapper maintains
    a process pool that is shared across all step() calls.

    The wrapped policy must be pickleable for this to work.
    """

    def __init__(self, wrapped_policy: AgentPolicy, process_pool: ProcessPoolExecutor):
        """Initialize the wrapper.

        Args:
            wrapped_policy: The AgentPolicy instance to wrap
            process_pool: Shared ProcessPoolExecutor for subprocess execution
        """
        super().__init__(wrapped_policy.policy_env_info)
        self._wrapped_policy = wrapped_policy
        self._process_pool = process_pool
        self._policy_data: Optional[bytes] = None

        # Pre-pickle the policy for efficiency
        try:
            self._policy_data = pickle.dumps(wrapped_policy)
        except Exception as e:
            logger.warning(f"Failed to pickle policy, will fall back to sequential execution: {e}")
            self._policy_data = None

    def step(self, obs: AgentObservation) -> Action:
        """Execute policy step in subprocess.

        Args:
            obs: The observation for this agent

        Returns:
            The action to take
        """
        # Fall back to sequential if policy isn't pickleable
        if self._policy_data is None:
            return self._wrapped_policy.step(obs)

        # Serialize observation
        obs_data = (
            obs.agent_id,
            [
                {
                    "feature_id": token.feature.id,
                    "feature_name": token.feature.name,
                    "feature_normalization": token.feature.normalization,
                    "location": token.location,
                    "value": token.value,
                    "raw_token": token.raw_token,
                }
                for token in obs.tokens
            ],
        )

        # Submit to process pool
        try:
            future = self._process_pool.submit(_agent_step_worker, self._policy_data, obs_data)
            action_name, computation_time_ms = future.result()
        except Exception as e:
            logger.warning(f"Subprocess execution failed, falling back to sequential: {e}")
            return self._wrapped_policy.step(obs)

        # Reconstruct action from name
        actions_list = list(self.policy_env_info.actions.actions())
        action = next((a for a in actions_list if a.name == action_name), None)
        if action is None:
            logger.warning(f"Action '{action_name}' not found, falling back to wrapped policy")
            return self._wrapped_policy.step(obs)

        return action

    def reset(self, simulation: Optional[object] = None) -> None:
        """Reset the wrapped policy."""
        self._wrapped_policy.reset(simulation)
        # Re-pickle policy in case state changed
        try:
            self._policy_data = pickle.dumps(self._wrapped_policy)
        except Exception as e:
            logger.warning(f"Failed to re-pickle policy after reset: {e}")
