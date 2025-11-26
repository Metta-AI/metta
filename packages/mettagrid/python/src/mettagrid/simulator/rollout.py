import logging
import time
from typing import Optional

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.policy.policy import AgentPolicy
from mettagrid.renderer.renderer import Renderer, RenderMode, create_renderer
from mettagrid.simulator import Simulator, SimulatorEventHandler
from mettagrid.util.stats_writer import StatsWriter

logger = logging.getLogger(__name__)


class Rollout:
    """Rollout class for running a multi-agent policy rollout."""

    def __init__(
        self,
        config: MettaGridConfig,
        policies: list[AgentPolicy],
        max_action_time_ms: int | None = 10000,
        render_mode: Optional[RenderMode] = None,
        seed: int = 0,
        event_handlers: Optional[list[SimulatorEventHandler]] = None,
        stats_writer: Optional[StatsWriter] = None,
    ):
        self._config = config
        self._policies = policies
        self._simulator = Simulator()
        self._max_action_time_ms: int = max_action_time_ms or 10000
        self._renderer: Optional[Renderer] = None
        self._timeout_counts: list[int] = [0] * len(policies)
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

    def step(self) -> None:
        """Execute one step of the rollout."""
        logger.info(f"[FLOW-14] Rollout.step() called - querying {len(self._policies)} agent policies")
        for i in range(len(self._policies)):
            start_time = time.time()
            logger.info(f"[FLOW-15] Querying policy for agent {i}")
            action = self._policies[i].step(self._agents[i].observation)
            logger.info(f"[FLOW-16] Agent {i} returned action: {action.name}")
            end_time = time.time()
            if (end_time - start_time) > self._max_action_time_ms:
                logger.warning(
                    f"Action took {end_time - start_time} seconds, exceeding max of {self._max_action_time_ms}ms"
                )
                action = self._config.game.actions.noop.Noop()
                self._timeout_counts[i] += 1
            self._agents[i].set_action(action)

        if self._renderer is not None:
            self._renderer.render()

        logger.info("[FLOW-17] Calling Simulation.step() to execute C++ simulation step")
        self._sim.step()
        logger.info(f"[FLOW-18] Simulation step complete. Current step: {self._sim.current_step}")

    def run_until_done(self) -> None:
        """Run the rollout until completion or early exit."""
        step_count = 0
        logger.info("[FLOW-19] Starting run_until_done loop")
        while not self.is_done():
            step_count += 1
            logger.info(f"[FLOW-20] Starting step {step_count}")
            self.step()
        logger.info(f"[FLOW-21] Run completed after {step_count} steps")

    def is_done(self) -> bool:
        return self._sim.is_done()

    @property
    def timeout_counts(self) -> list[int]:
        """Return the timeout counts for each agent."""
        return self._timeout_counts
