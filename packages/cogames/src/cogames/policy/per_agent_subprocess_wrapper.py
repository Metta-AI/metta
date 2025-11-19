"""Per-agent subprocess wrapper - each agent runs in its own persistent subprocess.

This architecture wraps AgentPolicy instances directly, giving each agent its own
dedicated subprocess. State is maintained within each agent's process, supporting
stateful policies (LSTM, scripted agents).

Architecture:
- Each AgentPolicy gets wrapped with its own subprocess
- State persists within the agent's process across steps
- Communication via Queue (request/response pattern)
"""

from __future__ import annotations

import pickle
from multiprocessing import Process, Queue
from typing import Any, Optional

from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import PolicySpec
from mettagrid.simulator import Action, AgentObservation


def _agent_worker_process(
    agent_id: int,
    policy_class_path: str,
    policy_data_path: Optional[str],
    env_info_dict: dict,
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    """Worker process for a single agent.

    This process maintains the agent's policy and state throughout the episode.
    It listens for observation requests and returns actions.
    """
    # Initialize policy in this process
    env_info = PolicyEnvInterface(**env_info_dict)
    policy_spec = PolicySpec(class_path=policy_class_path, data_path=policy_data_path)
    multi_agent_policy = initialize_or_load_policy(env_info, policy_spec)
    agent_policy = multi_agent_policy.agent_policy(agent_id)

    # Reset policy at start
    agent_policy.reset()

    # Main loop: wait for observations, compute actions
    while True:
        # Wait for observation
        request = request_queue.get()

        if request is None:  # Shutdown signal
            break

        if request == "reset":
            agent_policy.reset()
            response_queue.put("reset_ok")
            continue

        # request is (step_id, obs_data)
        step_id, obs_data = request
        obs = pickle.loads(obs_data)

        # Compute action
        action = agent_policy.step(obs)

        # Return action
        action_data = pickle.dumps(action)
        response_queue.put((step_id, action_data))


class PerAgentSubprocessPolicy(AgentPolicy):
    """AgentPolicy that runs in a dedicated subprocess.

    Each agent has its own persistent subprocess that maintains policy state.
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        policy_class_path: str,
        policy_data_path: Optional[str],
        env_info_dict: dict,
    ):
        super().__init__(policy_env_info)
        self._agent_id = agent_id
        self._policy_class_path = policy_class_path
        self._policy_data_path = policy_data_path
        self._env_info_dict = env_info_dict

        # Create queues for communication
        self._request_queue: Queue = Queue()
        self._response_queue: Queue = Queue()

        # Start the worker process
        self._process = Process(
            target=_agent_worker_process,
            args=(
                agent_id,
                policy_class_path,
                policy_data_path,
                env_info_dict,
                self._request_queue,
                self._response_queue,
            ),
        )
        self._process.start()

        # Track step IDs for ordering
        self._step_id = 0

    def step(self, obs: AgentObservation) -> Action:
        """Get action from the agent's subprocess."""
        # Serialize observation
        obs_data = pickle.dumps(obs)

        # Send request with step ID
        self._request_queue.put((self._step_id, obs_data))
        self._step_id += 1

        # Wait for response
        response = self._response_queue.get()
        step_id, action_data = response

        # Deserialize and return action
        action = pickle.loads(action_data)
        return action

    def reset(self, simulation: Optional[Any] = None) -> None:
        """Reset the agent's policy in its subprocess."""
        self._request_queue.put("reset")
        response = self._response_queue.get()
        assert response == "reset_ok"
        self._step_id = 0

    def shutdown(self) -> None:
        """Shutdown the agent's subprocess."""
        if self._process.is_alive():
            self._request_queue.put(None)  # Shutdown signal
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_process"):
            self.shutdown()


def wrap_agent_policy_in_subprocess(
    agent_policy: AgentPolicy,
    multi_agent_policy: MultiAgentPolicy,
    agent_id: int,
) -> PerAgentSubprocessPolicy:
    """Wrap an AgentPolicy to run in its own subprocess.

    This is the preferred way to wrap policies - wrap at the AgentPolicy level
    where they're actually used, rather than wrapping the MultiAgentPolicy.

    Args:
        agent_policy: The AgentPolicy instance to wrap (used for policy_env_info)
        multi_agent_policy: The parent MultiAgentPolicy (used to extract class/data path)
        agent_id: The agent ID

    Returns:
        A PerAgentSubprocessPolicy that runs the agent in a subprocess
    """
    # Get policy class path and data path from the MultiAgentPolicy
    policy_class = multi_agent_policy.__class__
    policy_class_path = f"{policy_class.__module__}.{policy_class.__name__}"

    policy_data_path: Optional[str] = None
    if hasattr(multi_agent_policy, "_policy_data_path"):
        policy_data_path = getattr(multi_agent_policy, "_policy_data_path", None)

    # Convert PolicyEnvInterface to dict for serialization
    env_info_dict = multi_agent_policy.policy_env_info.model_dump(mode="python")

    return PerAgentSubprocessPolicy(
        agent_policy.policy_env_info,
        agent_id,
        policy_class_path,
        policy_data_path,
        env_info_dict,
    )


class PerAgentSubprocessWrapper(MultiAgentPolicy):
    """Wrapper where each agent runs in its own persistent subprocess.

    This is a convenience wrapper that applies subprocess wrapping to all agents
    of a MultiAgentPolicy. For more control, use wrap_agent_policy_in_subprocess()
    directly on AgentPolicy instances.

    Architecture:
    - Each agent gets its own Process (not a pool worker)
    - Each process maintains its own policy instance and state
    - Communication via Queue (request/response pattern)
    - State persists across steps within each agent's process
    """

    def __init__(
        self,
        wrapped_policy: MultiAgentPolicy,
    ):
        """Initialize the per-agent subprocess wrapper.

        Args:
            wrapped_policy: The MultiAgentPolicy to wrap
        """
        super().__init__(wrapped_policy.policy_env_info)
        self._wrapped_policy = wrapped_policy
        self._num_agents = wrapped_policy.policy_env_info.num_agents

        # Get policy class path and data path
        policy_class = wrapped_policy.__class__
        self._policy_class_path = f"{policy_class.__module__}.{policy_class.__name__}"

        self._policy_data_path: Optional[str] = None
        if hasattr(wrapped_policy, "_policy_data_path"):
            self._policy_data_path = getattr(wrapped_policy, "_policy_data_path", None)

        # Convert PolicyEnvInterface to dict for serialization
        env_info_dict = wrapped_policy.policy_env_info.model_dump(mode="python")

        # Cache agent policies (each will create its own subprocess)
        self._agent_policies: dict[int, PerAgentSubprocessPolicy] = {}
        self._env_info_dict = env_info_dict

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy that runs in its own subprocess."""
        if agent_id not in self._agent_policies:
            # Use the wrapper function for consistency
            base_agent_policy = self._wrapped_policy.agent_policy(agent_id)
            self._agent_policies[agent_id] = wrap_agent_policy_in_subprocess(
                base_agent_policy,
                self._wrapped_policy,
                agent_id,
            )
        return self._agent_policies[agent_id]

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load policy data (requires restarting subprocesses)."""
        self._wrapped_policy.load_policy_data(policy_data_path)
        self._policy_data_path = policy_data_path
        # Note: Existing subprocesses won't reload - would need to restart them

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save policy data."""
        self._wrapped_policy.save_policy_data(policy_data_path)

    def reset(self) -> None:
        """Reset all agent policies in their subprocesses."""
        for agent_policy in self._agent_policies.values():
            agent_policy.reset()

    def shutdown_all(self) -> None:
        """Shutdown all agent subprocesses."""
        for agent_policy in self._agent_policies.values():
            agent_policy.shutdown()
        self._agent_policies.clear()

    def __del__(self):
        """Cleanup all subprocesses on deletion."""
        if hasattr(self, "_agent_policies"):
            self.shutdown_all()
