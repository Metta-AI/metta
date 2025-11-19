import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.policy.policy import AgentPolicy
from mettagrid.renderer.renderer import Renderer, RenderMode, create_renderer
from mettagrid.simulator import Simulator, SimulatorEventHandler
from mettagrid.util.stats_writer import StatsWriter

logger = logging.getLogger(__name__)


def _agent_step_worker(policy_data: tuple, obs_data: tuple) -> tuple[int, str, float]:
    """
    Worker function for subprocess execution of agent policy step.

    Args:
        policy_data: Tuple of (policy_class_path, policy_state) for reconstructing policy
        obs_data: Tuple of (agent_id, observation_tokens_dict) for AgentObservation

    Returns:
        Tuple of (agent_id, action_name, computation_time_ms)
    """
    import pickle
    import time

    from mettagrid.config.id_map import ObservationFeatureSpec
    from mettagrid.simulator.interface import AgentObservation, ObservationToken

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

    # Reconstruct policy (this may fail for non-pickleable policies)
    try:
        policy_class_path, policy_state = policy_data
        # For now, we'll need policies to be pickleable
        # This is a limitation that may need to be addressed per policy type
        policy = pickle.loads(policy_state) if policy_state else None
        if policy is None:
            raise ValueError("Policy reconstruction failed")
    except Exception as e:
        # Fallback: return noop action
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to reconstruct policy for agent {agent_id}: {e}")
        return (agent_id, "noop", 0.0)

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

    return (agent_id, action_name, computation_time_ms)


class Rollout:
    """Rollout class for running a multi-agent policy rollout."""

    def __init__(
        self,
        config: MettaGridConfig,
        policies: list[AgentPolicy],
        max_action_time_ms: int | None = 10000,
        render_mode: Optional[RenderMode] = None,
        seed: int = 0,
        pass_sim_to_policies: bool = False,
        event_handlers: Optional[list[SimulatorEventHandler]] = None,
        stats_writer: Optional[StatsWriter] = None,
        parallel_agents: bool = False,
    ):
        self._config = config
        self._policies = policies
        self._simulator = Simulator()
        self._max_action_time_ms: int = max_action_time_ms or 10000
        self._renderer: Optional[Renderer] = None
        self._timeout_counts: list[int] = [0] * len(policies)
        self._pass_sim_to_policies = pass_sim_to_policies  # Whether to pass the simulation to the policies
        self._parallel_agents = parallel_agents
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # Initialize process pool if parallel execution is enabled
        if self._parallel_agents and len(policies) > 1:
            max_workers = min(len(policies), os.cpu_count() or 1)
            try:
                self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
                # Test if policies are pickleable
                import pickle

                for i, policy in enumerate(policies):
                    try:
                        pickle.dumps(policy)
                    except Exception as e:
                        logger.warning(f"Policy {i} is not pickleable, falling back to sequential execution: {e}")
                        if self._process_pool:
                            self._process_pool.shutdown(wait=False)
                        self._process_pool = None
                        self._parallel_agents = False
                        break
            except Exception as e:
                logger.warning(f"Failed to initialize process pool, falling back to sequential execution: {e}")
                self._parallel_agents = False

        # Attach renderer if specified
        if render_mode is not None:
            self._renderer = create_renderer(render_mode)
            self._simulator.add_event_handler(self._renderer)
        # Attach stats tracker if provided
        if stats_writer is not None:
            self._simulator.add_event_handler(StatsTracker(stats_writer))
        # Attach additional event handlers
        for handler in event_handlers or []:
            self._simulator.add_event_handler(handler)
        self._sim = self._simulator.new_simulation(config, seed)
        self._agents = self._sim.agents()

        # Reset policies and create agent policies if needed
        for policy in self._policies:
            policy.reset()

    def __del__(self):
        """Clean up process pool on deletion."""
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=False)

    def step(self) -> None:
        """Execute one step of the rollout."""
        if self._parallel_agents and self._process_pool is not None and len(self._policies) > 1:
            # Parallel execution path
            import pickle

            # Prepare policy data and observations
            futures = {}
            for i in range(len(self._policies)):
                try:
                    policy_data = (None, pickle.dumps(self._policies[i]))
                    obs = self._agents[i].observation
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
                    future = self._process_pool.submit(_agent_step_worker, policy_data, obs_data)
                    futures[future] = i
                except Exception as e:
                    logger.warning(f"Failed to submit agent {i} for parallel execution: {e}, using sequential")
                    # Fall back to sequential for this agent
                    start_time = time.time()
                    action = self._policies[i].step(self._agents[i].observation)
                    end_time = time.time()
                    if (end_time - start_time) * 1000.0 > self._max_action_time_ms:
                        elapsed_ms = (end_time - start_time) * 1000.0
                        logger.warning(f"Action took {elapsed_ms}ms, exceeding max of {self._max_action_time_ms}ms")
                        action = self._config.game.actions.noop.Noop()
                        self._timeout_counts[i] += 1
                    self._agents[i].set_action(action)

            # Collect results from parallel execution
            for future in as_completed(futures):
                i = futures[future]
                try:
                    agent_id, action_name, computation_time_ms = future.result()
                    # Find action by name
                    actions_list = list(self._config.game.actions.actions())
                    action = next((a for a in actions_list if a.name == action_name), None)
                    if action is None:
                        logger.warning(f"Action '{action_name}' not found, using noop")
                        action = self._config.game.actions.noop.Noop()
                    if computation_time_ms > self._max_action_time_ms:
                        logger.warning(
                            f"Action took {computation_time_ms}ms, exceeding max of {self._max_action_time_ms}ms"
                        )
                        action = self._config.game.actions.noop.Noop()
                        self._timeout_counts[i] += 1
                    self._agents[i].set_action(action)
                except Exception as e:
                    logger.warning(f"Failed to get result for agent {i}: {e}, using noop")
                    self._agents[i].set_action(self._config.game.actions.noop.Noop())
        else:
            # Sequential execution path (original implementation)
            for i in range(len(self._policies)):
                start_time = time.time()
                action = self._policies[i].step(self._agents[i].observation)
                end_time = time.time()
                if (end_time - start_time) * 1000.0 > self._max_action_time_ms:
                    logger.warning(
                        f"Action took {end_time - start_time} seconds, exceeding max of {self._max_action_time_ms}ms"
                    )
                    action = self._config.game.actions.noop.Noop()
                    self._timeout_counts[i] += 1
                self._agents[i].set_action(action)

        if self._renderer is not None:
            self._renderer.render()

        self._sim.step()

    def run_until_done(self) -> None:
        """Run the rollout until completion or early exit."""
        while not self.is_done():
            self.step()

    def is_done(self) -> bool:
        return self._sim.is_done()

    @property
    def timeout_counts(self) -> list[int]:
        """Return the timeout counts for each agent."""
        return self._timeout_counts
