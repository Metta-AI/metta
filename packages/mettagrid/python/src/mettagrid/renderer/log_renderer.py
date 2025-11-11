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
        assert self._sim is not None

        logger.info("=== Episode Start ===")
        logger.info(f"Num agents: {self._sim.num_agents}")
        logger.info(f"Max steps: {self._sim.config.game.max_steps}")

    @override
    def on_step(self) -> None:
        """Log each step."""
        assert self._sim is not None

        current_step = self._sim.current_step
        raw_actions = self._sim.raw_actions()
        action_names = self._sim.action_names
        named_actions = []
        for action_idx in raw_actions:
            idx = int(action_idx)
            if 0 <= idx < len(action_names):
                named_actions.append(action_names[idx])
            else:
                named_actions.append(f"invalid:{idx}")
        logger.info("--------------------------------")
        logger.info(f"Step {current_step}")
        logger.info(f"Actions (idx): {raw_actions.tolist()}")
        logger.info(f"Actions (name): {named_actions}")
        logger.info(f"Episode rewards: {self._sim.episode_rewards}")
        logger.info(f"Done: {self._sim.is_done()}")

    @override
    def on_episode_end(self) -> None:
        """Log episode end."""
        assert self._sim is not None

        logger.info("================================")
        logger.info("=== Episode End ===")
        logger.info(f"Total steps: {self._sim.current_step}")
        logger.info(f"Total rewards: {self._sim.episode_rewards}")
        logger.info(f"Total stats: {self._sim.episode_stats}")
        logger.info(f"Done: {self._sim.is_done()}")
