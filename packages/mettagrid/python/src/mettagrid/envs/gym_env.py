"""
MettaGridGymEnv - Gymnasium adapter for MettaGrid.

This class implements the Gymnasium environment interface using the base MettaGridEnv.
Supports both single-agent and multi-agent modes.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from gymnasium import Env as GymEnv
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.renderer.renderer import NoRenderer, Renderer


class MettaGridGymEnv(MettaGridCore, GymEnv):
    """
    Gymnasium adapter for MettaGrid environments.

    This class provides a Gymnasium-compatible interface for MettaGrid environments,
    supporting both single-agent and multi-agent scenarios.
    No training features are included - this is purely for Gymnasium compatibility.

    Inherits from:
    - MettaGridCore: Core C++ environment wrapper functionality
    - gymnasium.Env: Standard Gymnasium environment interface
      https://github.com/Farama-Foundation/Gymnasium/blob/ad23dfbbe29f83107404f9f6a56131f6b498d0d7/gymnasium/core.py#L23
    """

    def __init__(
        self,
        mg_config: MettaGridConfig,
        render_mode: Optional[
            Literal["text", "unicode", "miniscope", "gui", "human", "replay", "none", "explicit"]
        ] = None,
        renderer: Optional[Renderer] = None,
    ):
        """
        Initialize Gymnasium environment.

        Args:
            mg_config: Environment configuration
            render_mode: Rendering mode (same options as MettaGridEnv)
            renderer: Optional explicit renderer to use (requires render_mode="explicit")
        """
        assert mg_config.game.num_agents == 1, "Gymnasium environments must be single-agent"

        # Initialize core functionality
        MettaGridCore.__init__(
            self,
            mg_config,
        )

        # Initialize Gym environment
        GymEnv.__init__(self)

        # Create or use renderer
        if renderer is not None:
            self._renderer: Renderer = renderer
        else:
            self._renderer: Renderer = self._create_renderer(render_mode)

    def _create_renderer(self, render_mode: Optional[str]) -> Renderer:
        """Create the appropriate renderer based on render_mode."""
        if render_mode in ("text", "unicode", "miniscope"):
            from mettagrid.renderer.miniscope import MiniscopeRenderer

            # All text modes default to full interactive (can be disabled if needed)
            return MiniscopeRenderer(enable_full_interactive=True)
        elif render_mode in ("gui", "human"):
            from mettagrid.renderer.mettascope import MettascopeRenderer

            return MettascopeRenderer()
        elif render_mode == "replay":
            # Note: replay mode is not supported in GymEnv without explicit renderer
            # Use an explicit ReplayLogRenderer via the renderer parameter instead
            raise ValueError(
                "render_mode='replay' requires passing an explicit ReplayLogRenderer "
                "via the renderer parameter with replay_dir configured"
            )
        else:  # None or "none"
            return NoRenderer()

    @override  # gymnasium.Env.reset
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observations, info)
        """
        # Call the base reset method
        obs, info = super().reset(seed)

        return obs[0], info

    @override  # gymnasium.Env.step
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            action: Action array. For single-agent: shape (2,). For multi-agent: shape (num_agents, 2)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Handle single-agent action format
        if action.ndim == 1:
            # Convert single action to multi-agent format
            actions = action[np.newaxis, ...]  # Add batch dimension
        else:
            actions = action

        # Ensure correct dtype
        actions = actions.astype(dtype_actions)

        # Call base step implementation
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Handle single-agent return format
        return observations[0], rewards[0].item(), terminals[0].item(), truncations[0].item(), infos

    # Gymnasium space properties
    @property
    @override  # gymnasium.Env.observation_space
    def observation_space(self):
        """Get observation space."""
        return self._observation_space

    @property
    @override  # gymnasium.Env.action_space
    def action_space(self):
        """Get action space."""
        return self._action_space

    # PufferLib compatibility properties
    @property
    def single_observation_space(self):
        """Single agent observation space (PufferLib compatibility)."""
        return self._observation_space

    @property
    def single_action_space(self):
        """Single agent action space (PufferLib compatibility)."""
        return self._action_space
