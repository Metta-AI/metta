"""
Trajectory collector for GAMMA metrics.

Collects agent positions, velocities, and task directions during episodes.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt


class TrajectoryCollector:
    """
    Collects agent trajectories during episodes for alignment metric computation.

    Usage:
        collector = TrajectoryCollector(num_agents=16)

        # During episode
        collector.reset()
        for step in range(episode_length):
            positions = get_agent_positions(env)  # shape (num_agents, 2)
            collector.record_step(positions, dt=0.1)

        # After episode
        trajectories = collector.get_trajectories()
    """

    def __init__(self, num_agents: int):
        """
        Initialize trajectory collector.

        Args:
            num_agents: Number of agents to track
        """
        self.num_agents = num_agents
        self.reset()

    def reset(self) -> None:
        """Reset collector for a new episode."""
        self._positions: dict[int, list[npt.NDArray[np.floating[Any]]]] = defaultdict(list)
        self._velocities: dict[int, list[npt.NDArray[np.floating[Any]]]] = defaultdict(list)
        self._task_directions: dict[int, list[npt.NDArray[np.floating[Any]]]] = defaultdict(list)
        self._power: dict[int, list[float]] = defaultdict(list)
        self._prev_positions: dict[int, npt.NDArray[np.floating[Any]] | None] = {}
        self._dt_history: list[float] = []

    def record_step(
        self,
        positions: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]] | None = None,
        power: npt.NDArray[np.floating[Any]] | None = None,
        dt: float = 0.1,
    ) -> None:
        """
        Record one timestep of agent data.

        Args:
            positions: Agent positions, shape (num_agents, d)
            task_directions: Optional task directions, shape (num_agents, d)
            power: Optional power measurements, shape (num_agents,)
            dt: Time step size
        """
        self._dt_history.append(dt)

        for agent_id in range(len(positions)):
            pos = positions[agent_id]

            # Store position
            self._positions[agent_id].append(pos.copy())

            # Compute velocity from position difference
            if self._prev_positions.get(agent_id) is not None:
                prev_pos = self._prev_positions[agent_id]
                velocity = (pos - prev_pos) / dt
            else:
                velocity = np.zeros_like(pos)

            self._velocities[agent_id].append(velocity)
            self._prev_positions[agent_id] = pos.copy()

            # Store task direction
            if task_directions is not None and agent_id < len(task_directions):
                self._task_directions[agent_id].append(task_directions[agent_id].copy())
            else:
                # Default: zero task direction (will need to be computed later)
                self._task_directions[agent_id].append(np.zeros_like(pos))

            # Store power
            if power is not None and agent_id < len(power):
                self._power[agent_id].append(float(power[agent_id]))

    def get_trajectories(self) -> list[dict[str, npt.NDArray[np.floating[Any]]]]:
        """
        Get collected trajectories for all agents.

        Returns:
            List of trajectory dictionaries, one per agent, with keys:
                - 'positions': shape (T, d)
                - 'velocities': shape (T, d)
                - 'task_directions': shape (T, d)
                - 'power': shape (T,) if power was recorded
        """
        trajectories = []

        for agent_id in range(self.num_agents):
            if agent_id not in self._positions or len(self._positions[agent_id]) == 0:
                # Agent has no data, create empty trajectory
                traj = {
                    "positions": np.zeros((0, 2)),
                    "velocities": np.zeros((0, 2)),
                    "task_directions": np.zeros((0, 2)),
                }
            else:
                traj = {
                    "positions": np.array(self._positions[agent_id]),
                    "velocities": np.array(self._velocities[agent_id]),
                    "task_directions": np.array(self._task_directions[agent_id]),
                }

                if agent_id in self._power and len(self._power[agent_id]) > 0:
                    traj["power"] = np.array(self._power[agent_id])

            trajectories.append(traj)

        return trajectories

    def get_mean_dt(self) -> float:
        """Get mean time step size."""
        if len(self._dt_history) == 0:
            return 0.1  # Default
        return float(np.mean(self._dt_history))

    def get_agent_trajectory(self, agent_id: int) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Get trajectory for a specific agent."""
        trajectories = self.get_trajectories()
        if agent_id < len(trajectories):
            return trajectories[agent_id]
        return {
            "positions": np.zeros((0, 2)),
            "velocities": np.zeros((0, 2)),
            "task_directions": np.zeros((0, 2)),
        }
