"""Log-based renderer that outputs simulation events to the logger."""

import logging

import typing_extensions

import mettagrid.renderer.renderer

logger = logging.getLogger(__name__)


class LogRenderer(mettagrid.renderer.renderer.Renderer):
    """Renderer that logs simulation events for debugging and analysis."""

    def __init__(self):
        super().__init__()

    @typing_extensions.override
    def on_episode_start(self) -> None:
        """Log episode start."""
        assert self._sim is not None

        logger.info("=== Episode Start ===")
        logger.info(f"Num agents: {self._sim.num_agents}")
        logger.info(f"Max steps: {self._sim.config.game.max_steps}")

    @typing_extensions.override
    def on_step(self) -> None:
        """Log each step."""
        assert self._sim is not None

        current_step = self._sim.current_step
        logger.info("--------------------------------")
        logger.info(f"Step {current_step}")
        logger.info(f"Episode rewards: {self._sim.episode_rewards}")
        logger.info(f"Done: {self._sim.is_done()}")

    @typing_extensions.override
    def on_episode_end(self) -> None:
        """Log episode end."""
        assert self._sim is not None

        logger.info("================================")
        logger.info("=== Episode End ===")
        logger.info(f"Total steps: {self._sim.current_step}")
        logger.info(f"Total rewards: {self._sim.episode_rewards}")
        logger.info(f"Total stats: {self._sim.episode_stats}")
        logger.info(f"Done: {self._sim.is_done()}")
