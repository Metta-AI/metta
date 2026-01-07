import logging
import time
from typing import Optional

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.policy.policy import AgentPolicy
from mettagrid.renderer.renderer import Renderer, RenderMode, create_renderer
from mettagrid.simulator.interface import SimulatorEventHandler
from mettagrid.simulator.simulator import Simulator
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

        self._step_count = 0
        self._failed_agents: list[bool] = [False] * len(policies)
        self._agent_failure_steps: list[int | None] = [None] * len(policies)

    def step(self) -> None:
        """Execute one step of the rollout."""
        if self._step_count % 100 == 0:
            logger.debug(f"Step {self._step_count}")

        for i, policy in enumerate(self._policies):
            if self._failed_agents[i]:
                action = self._config.game.actions.noop.Noop()
                self._agents[i].set_action(action)
                continue

            start_time = time.time()
            try:
                action = policy.step(self._agents[i].observation)
            except Exception:
                logger.exception(
                    "Policy %s failed at step %s",
                    policy.__class__.__name__,
                    self._step_count,
                )
                self._record_failure(i)
                action = self._config.game.actions.noop.Noop()
            else:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self._max_action_time_ms:
                    logger.warning(
                        "Action took %.0fms, exceeding max of %sms",
                        elapsed_ms,
                        self._max_action_time_ms,
                    )
                    action = self._config.game.actions.noop.Noop()
                    self._timeout_counts[i] += 1

            self._agents[i].set_action(action)

        if self._renderer is not None:
            self._renderer.render()

        self._sim.step()
        self._step_count += 1

    def _record_failure(self, agent_index: int) -> None:
        self._failed_agents[agent_index] = True
        self._agent_failure_steps[agent_index] = self._step_count

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

    @property
    def agent_failure_steps(self) -> list[int | None]:
        """Return the step indices where agents first failed (None if never failed)."""
        return self._agent_failure_steps
