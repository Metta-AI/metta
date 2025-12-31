"""Log-based renderer that outputs simulation events to the logger."""

import logging

from typing_extensions import override

from mettagrid.renderer.renderer import Renderer

logger = logging.getLogger(__name__)


class LogRenderer(Renderer):
    """Renderer that logs simulation events for debugging and analysis."""

    def __init__(self):
        super().__init__()

    @override
    def on_episode_start(self) -> None:
        """Log episode start."""
        logger.info("=== Episode Start ===")
        logger.info(f"Num agents: {self._sim.num_agents}")
        logger.info(f"Max steps: {self._sim.config.game.max_steps}")

    @override
    def on_step(self) -> None:
        """Log each step."""
        current_step = self._sim.current_step
        logger.info("--------------------------------")
        logger.info(f"Step {current_step}")
        logger.info(f"Episode rewards: {self._sim.episode_rewards}")
        logger.info(f"Done: {self._sim.is_done()}")

    @override
    def on_episode_end(self) -> None:
        """Log episode end."""
        logger.info("================================")
        logger.info("=== Episode End ===")
        logger.info(f"Total steps: {self._sim.current_step}")
        logger.info(f"Total rewards: {self._sim.episode_rewards}")
        logger.info(f"Total stats: {self._sim.episode_stats}")
        logger.info(f"Done: {self._sim.is_done()}")
