#!/usr/bin/env -S uv run
# Runs policies with ASCII rendering to visualize agent behavior in real-time.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Protocol, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid import (
    MettaGridEnv,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.util.actions import generate_valid_random_actions
from metta.mettagrid.util.hydra import get_cfg
from metta.util.metta_script import metta_script
from tools.utils import get_policy_store_from_cfg


class Policy(Protocol):
    """Protocol for policy classes."""

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict actions given observations."""
        ...


class BasePolicy(ABC):
    """Base class for all policies."""

    def __init__(self, env: MettaGridEnv) -> None:
        self.env = env
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space

    @abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict actions given observations."""
        pass


class RandomPolicy(BasePolicy):
    """Simple random policy using valid action generation."""

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return valid random actions for all agents."""
        assert obs.dtype == dtype_observations
        return generate_valid_random_actions(self.env, self.num_agents)


class SimplePolicy(BasePolicy):
    """A simple policy that tries to move towards objectives with valid actions."""

    def __init__(self, env: MettaGridEnv) -> None:
        super().__init__(env)

        # Movement options
        self.cardinal_directions: List[int] = [1, 3, 5, 7]  # up, left, right, down
        self.move_directions: List[int] = [1, 2, 3, 5, 7, 8]  # Cardinal + diagonal
        self.rotation_orientations: List[int] = [0, 1, 2, 3]  # up, down, left, right

        # Get action indices
        self._initialize_action_indices()

    def _initialize_action_indices(self) -> None:
        """Initialize action type indices from environment."""
        try:
            action_names: List[str] = self.env.action_names
            self.move_idx: int = action_names.index("move") if "move" in action_names else 0
            self.rotate_idx: int = action_names.index("rotate") if "rotate" in action_names else 1
            self.pickup_idx: int = action_names.index("pickup") if "pickup" in action_names else 2
        except (AttributeError, ValueError):
            # Fallback to default indices
            self.move_idx = 0
            self.rotate_idx = 1
            self.pickup_idx = 2

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return simple movement actions with proper validation."""
        assert obs.dtype == dtype_observations

        action_type: int
        direction: int

        # Decide on action type based on probability
        rand_val: float = np.random.random()

        if rand_val < 0.6:
            # Move action with cardinal direction
            direction = int(np.random.choice(self.cardinal_directions))
            action_type = self.move_idx
        elif rand_val < 0.8:
            # Move action with any direction
            direction = int(np.random.choice(self.move_directions))
            action_type = self.move_idx
        elif rand_val < 0.9:
            # Rotate action
            direction = int(np.random.choice(self.rotation_orientations))
            action_type = self.rotate_idx
        else:
            # Stay in place or try pickup
            direction = 4 if np.random.random() < 0.5 else 0
            action_type = self.pickup_idx if np.random.random() < 0.3 else self.move_idx

        # Generate valid actions for all agents
        actions = generate_valid_random_actions(
            self.env, self.num_agents, force_action_type=action_type, force_action_arg=direction
        )

        # For multi-agent, only force the action for the first agent
        if self.num_agents > 1:
            # Let other agents act randomly
            other_actions = generate_valid_random_actions(self.env, self.num_agents - 1)
            actions[1:] = other_actions[: self.num_agents - 1]

        assert actions.dtype == dtype_actions
        return actions


class PickupAlwaysPolicy(BasePolicy):
    """A policy that moves randomly but always picks up resources when available."""

    def __init__(self, env: MettaGridEnv) -> None:
        super().__init__(env)

        # Movement options
        self.cardinal_directions: List[int] = [1, 3, 5, 7]  # up, left, right, down
        self.move_directions: List[int] = [1, 2, 3, 5, 7, 8]  # Cardinal + diagonal
        self.rotation_orientations: List[int] = [0, 1, 2, 3]  # up, down, left, right

        # Get action indices
        self._initialize_action_indices()

    def _initialize_action_indices(self) -> None:
        """Initialize action type indices from environment."""
        try:
            action_names: List[str] = self.env.action_names
            self.move_idx: int = action_names.index("move") if "move" in action_names else 0
            self.rotate_idx: int = action_names.index("rotate") if "rotate" in action_names else 1
            self.pickup_idx: int = action_names.index("pickup") if "pickup" in action_names else 2
        except (AttributeError, ValueError):
            # Fallback to default indices
            self.move_idx = 0
            self.rotate_idx = 1
            self.pickup_idx = 2

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return actions that always pick up resources when available, otherwise move randomly."""
        assert obs.dtype == dtype_observations

        # For now, we'll use a simple approach: always try pickup first, then move randomly
        # In a more sophisticated implementation, we'd check the observation for nearby resources

        action_type: int
        direction: int

        # Always try pickup first (50% chance)
        if np.random.random() < 0.5:
            action_type = self.pickup_idx
            direction = 0  # No direction needed for pickup
        else:
            # Move randomly
            rand_val: float = np.random.random()
            if rand_val < 0.6:
                # Move action with cardinal direction
                direction = int(np.random.choice(self.cardinal_directions))
                action_type = self.move_idx
            elif rand_val < 0.8:
                # Move action with any direction
                direction = int(np.random.choice(self.move_directions))
                action_type = self.move_idx
            else:
                # Rotate action
                direction = int(np.random.choice(self.rotation_orientations))
                action_type = self.rotate_idx

        # Generate valid actions for all agents
        actions = generate_valid_random_actions(
            self.env, self.num_agents, force_action_type=action_type, force_action_arg=direction
        )

        # For multi-agent, only force the action for the first agent
        if self.num_agents > 1:
            # Let other agents act randomly
            other_actions = generate_valid_random_actions(self.env, self.num_agents - 1)
            actions[1:] = other_actions[: self.num_agents - 1]

        assert actions.dtype == dtype_actions
        return actions


class OpportunisticPolicy(BasePolicy):
    """A policy that wanders randomly but will reliably harvest adjacent resources.

    Logic per step (single-agent focus, but works multi-agent):
    1. If a harvestable object (mine, generator, converter) is in the tile the
       agent is *facing*, issue a pickup (``get_items``) action.
    2. Otherwise, if such an object exists in any *adjacent* cardinal tile,
       issue a rotate so the agent will face it on the next step.
    3. Otherwise take a random valid move (same distribution as before).
    """

    ORIENT_TO_DELTA = {
        0: (-1, 0),  # up
        1: (1, 0),  # down
        2: (0, -1),  # left
        3: (0, 1),  # right
    }
    DELTA_TO_ORIENT = {v: k for k, v in ORIENT_TO_DELTA.items()}

    def __init__(
        self,
        env: MettaGridEnv,
        pickup_prob: float = 0.5,
        cardinal_prob: float = 0.6,
        any_prob: float = 0.2,
        rotate_prob: float = 0.1,
    ) -> None:
        super().__init__(env)

        # Movement options
        self.cardinal_directions: List[int] = [1, 3, 5, 7]  # up, left, right, down (MettaGrid arg encoding)
        self.move_directions: List[int] = [1, 2, 3, 5, 7, 8]  # Cardinal + diagonal
        self.rotation_orientations: List[int] = [0, 1, 2, 3]  # up, down, left, right

        # Probabilities for different action types when wandering
        self.pickup_prob = pickup_prob
        self.cardinal_prob = cardinal_prob
        self.any_prob = any_prob
        self.rotate_prob = rotate_prob

        # Normalize probabilities to sum to 1.0
        total = pickup_prob + cardinal_prob + any_prob + rotate_prob
        self.pickup_prob, self.cardinal_prob, self.any_prob, self.rotate_prob = [
            p / total for p in [pickup_prob, cardinal_prob, any_prob, rotate_prob]
        ]

        # Get action indices from environment
        self._initialize_action_indices()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _initialize_action_indices(self) -> None:
        """Determine indices of move/rotate/pickup actions for this env."""
        try:
            action_names: List[str] = self.env.action_names
            self.move_idx: int = action_names.index("move") if "move" in action_names else 0
            self.rotate_idx: int = action_names.index("rotate") if "rotate" in action_names else 1
            # Prefer explicit get_items action for converters
            if "get_items" in action_names:
                self.pickup_idx = action_names.index("get_items")
            elif "pickup" in action_names:
                self.pickup_idx = action_names.index("pickup")
            else:
                self.pickup_idx = 2
        except (AttributeError, ValueError):
            # Fallback defaults
            self.move_idx = 0
            self.rotate_idx = 1
            self.pickup_idx = 2

    # ------------------------------------------------------------------
    # Core policy logic
    # ------------------------------------------------------------------
    def _adjacent_resources(self) -> tuple[bool, int]:
        """Return (is_front, desired_orientation) if a resource is adjacent.

        is_front: True when the resource is already in the tile the agent is
        facing. desired_orientation is the orientation that *would* face the
        resource (equal to current orientation when is_front is True).
        Returns (False, -1) when no adjacent resource found.
        """
        grid_objects = self.env.grid_objects
        agent_row = agent_col = agent_ori = None

        # 1. Locate the controlled agent (assuming id==0 for simplicity)
        for obj in grid_objects.values():
            if obj.get("agent_id") == 0:  # primary agent
                agent_row, agent_col = obj["r"], obj["c"]
                agent_ori = int(obj.get("agent:orientation", 0))
                break
        if agent_row is None:
            return (False, -1)
        # Narrow types: ensure agent_row and agent_col are ints for type checker
        assert isinstance(agent_row, int) and isinstance(agent_col, int)

        # 2. Build lookup of resource positions (cardinal neighbours only)
        for delta, orient in self.DELTA_TO_ORIENT.items():
            dr, dc = delta
            target_pos = (agent_row + dr, agent_col + dc)
            for obj in grid_objects.values():
                if obj.get("r") == target_pos[0] and obj.get("c") == target_pos[1]:
                    type_name = self.env.object_type_names[obj["type"]]
                    base_name = type_name.split(".")[0]
                    if base_name.startswith(("mine", "generator", "converter")):
                        # Harvestable resource found adjacent
                        is_front = orient == agent_ori
                        return (is_front, orient)
        return (False, -1)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Decide on an action for each step."""
        assert obs.dtype == dtype_observations

        is_front, desired_orientation = self._adjacent_resources()

        if desired_orientation != -1:
            if is_front:
                # Resource straight ahead – pick it up
                action_type = self.pickup_idx
                action_arg = 0
            else:
                # Need to rotate toward the resource
                action_type = self.rotate_idx
                action_arg = desired_orientation
        else:
            # No nearby resource – fallback to random wandering (existing probabilities)
            rand_val: float = np.random.random()
            if rand_val < self.cardinal_prob:
                action_type = self.move_idx
                action_arg = int(np.random.choice(self.cardinal_directions))
            elif rand_val < self.cardinal_prob + self.any_prob:
                action_type = self.move_idx
                action_arg = int(np.random.choice(self.move_directions))
            elif rand_val < self.cardinal_prob + self.any_prob + self.rotate_prob:
                action_type = self.rotate_idx
                action_arg = int(np.random.choice(self.rotation_orientations))
            else:
                action_type = self.pickup_idx
                action_arg = 0

        # Generate valid array for all agents; first agent gets forced action
        actions = generate_valid_random_actions(
            self.env, self.num_agents, force_action_type=action_type, force_action_arg=action_arg
        )
        return actions


class TrainedPolicyWrapper(BasePolicy):
    """Wrapper for trained policies with action validation."""

    def __init__(self, policy: Any, env: MettaGridEnv) -> None:
        super().__init__(env)
        self.policy = policy
        self._max_args: List[int] = env.max_action_args
        self._num_action_types: int = env.single_action_space.nvec[0]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict actions using trained policy with validation."""
        assert obs.dtype == dtype_observations

        import torch

        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).float()

            # Get actions from policy
            actions, _ = self.policy.forward(obs_tensor)
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()

            # Ensure proper shape
            actions = self._reshape_actions(actions)

            # Validate and clamp actions
            actions = self._validate_actions(actions)

            assert actions.dtype == dtype_actions
            return actions

    def _reshape_actions(self, actions: np.ndarray) -> np.ndarray:
        """Ensure actions have proper shape."""
        if len(actions.shape) == 1:
            return actions.reshape(1, -1).astype(dtype_actions)
        elif len(actions.shape) > 2:
            return actions.reshape(self.num_agents, -1).astype(dtype_actions)
        return actions.astype(dtype_actions)

    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        """Validate and clamp actions to valid ranges."""
        assert actions.dtype == dtype_actions, f"Actions must have dtype {dtype_actions}, got {actions.dtype}"

        for i in range(actions.shape[0]):
            # Clamp action type
            actions[i, 0] = np.clip(actions[i, 0], 0, self._num_action_types - 1)

            # Clamp action argument
            action_type: int = int(actions[i, 0])
            max_arg: int = self._max_args[action_type] if action_type < len(self._max_args) else 0
            actions[i, 1] = np.clip(actions[i, 1], 0, max_arg)

        return actions


def get_policy(policy_type: str, env: MettaGridEnv, cfg: DictConfig) -> Policy:
    """
    Get a policy based on the specified type.

    Args:
        policy_type: Type of policy ("random", "simple", "opportunistic", or "trained")
        env: MettaGrid environment
        cfg: Hydra configuration

    Returns:
        Policy instance
    """
    if policy_type == "random":
        return RandomPolicy(env)
    elif policy_type == "simple":
        return SimplePolicy(env)
    elif policy_type == "opportunistic":
        return OpportunisticPolicy(env)
    elif policy_type == "trained":
        return _load_trained_policy(env, cfg)
    else:
        print(f"Unknown policy type '{policy_type}', using simple policy")
        return SimplePolicy(env)


def _load_trained_policy(env: MettaGridEnv, cfg: DictConfig) -> Policy:
    """Attempt to load a trained policy, falling back to simple policy on failure."""
    try:
        policy_store = get_policy_store_from_cfg(cfg)
        policy_pr = policy_store.policy_record(cfg.policy_uri)

        # Check if this is a custom simple agent based on metadata
        if hasattr(policy_pr, "metadata") and policy_pr.metadata:
            agent_type = getattr(policy_pr.metadata, "agent_type", None)
            if agent_type == "pickup_always":
                print("🤖 Using pickup_always policy (always picks up resources)")
                return PickupAlwaysPolicy(env)
            elif agent_type == "pickup_sometimes":
                print("🤖 Using pickup_sometimes policy (sometimes picks up resources)")
                return SimplePolicy(env)  # Use regular simple policy for this case
            elif agent_type == "random":
                print("🤖 Using random policy")
                return RandomPolicy(env)

        # Default to trained policy wrapper
        return TrainedPolicyWrapper(policy_pr.policy, env)
    except Exception as e:
        print(f"Failed to load trained policy: {e}")
        print("Falling back to simple policy")
        return SimplePolicy(env)


def setup_environment(cfg: DictConfig) -> Tuple[MettaGridEnv, str]:
    """
    Set up the MettaGrid environment based on configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (environment, render_mode)
    """
    # Determine render mode
    render_mode: str = cfg.renderer_job.get("renderer_type", "human")
    if render_mode not in ["human", "nethack", "miniscope", "raylib"]:
        print(f"⚠️  Unknown renderer type '{render_mode}', using 'human' (nethack)")
        render_mode = "human"

    # Create curriculum
    if hasattr(cfg, "env") and cfg.env is not None:
        # Use the env configuration directly
        curriculum = SingleTaskCurriculum("renderer", cfg.env)
        print(f"📊 Using environment config: {cfg.env.game.num_agents} agents")
    else:
        # Fall back to the legacy renderer_job.environment approach
        env_cfg = get_cfg("benchmark")
        env_cfg.game.num_agents = cfg.renderer_job.num_agents
        env_cfg.game.max_steps = cfg.renderer_job.max_steps

        if hasattr(cfg.renderer_job, "environment") and cfg.renderer_job.environment:
            env_cfg.game.map_builder = OmegaConf.create(cfg.renderer_job.environment)
            print(f"📊 Using legacy environment config: {cfg.renderer_job.num_agents} agents")

        curriculum = SingleTaskCurriculum("renderer", env_cfg)

    env = MettaGridEnv(curriculum, render_mode=render_mode)

    return env, render_mode


def main(cfg: DictConfig) -> None:
    """
    Run policy visualization with ASCII or Miniscope rendering.

    Args:
        cfg: Hydra configuration
    """
    # Set up environment
    env, render_mode = setup_environment(cfg)

    # Get policy
    policy: Policy = get_policy(cfg.renderer_job.policy_type, env, cfg)
    print(f"🤖 Using {cfg.renderer_job.policy_type} policy")
    print(f"🎨 Using {render_mode} renderer")

    # Reset environment
    obs, info = env.reset()
    assert obs.dtype == dtype_observations, f"Observations must have dtype {dtype_observations}, got {obs.dtype}"

    print(f"🎮 Starting visualization for {cfg.renderer_job.num_steps} steps...")

    total_reward = 0.0
    step_count = 0

    try:
        for _step in range(cfg.renderer_job.num_steps):
            # Get action from policy
            actions = policy.predict(obs)

            try:
                # Step environment
                obs, rewards, terminals, truncations, info = env.step(actions)

                # Assert dtypes match expected
                assert obs.dtype == dtype_observations, "Observations dtype mismatch"
                assert rewards.dtype == dtype_rewards, "Rewards dtype mismatch"
                assert terminals.dtype == dtype_terminals, "Terminals dtype mismatch"
                assert truncations.dtype == dtype_truncations, "Truncations dtype mismatch"
            except AssertionError as e:
                if "Task is already complete" in str(e):
                    # Handle the case where task completion is called multiple times
                    print("🔄 Episode completed, resetting environment...")
                    obs, info = env.reset()
                    continue
                else:
                    raise  # Re-raise if it's a different assertion error

            # Track rewards
            step_reward = np.sum(rewards)
            total_reward += step_reward
            step_count += 1

            # Get resources from info if available
            resources_str = "None: 0"
            if "inventory" in info:
                inventory = info["inventory"]
                resources = [f"{k}: {v}" for k, v in inventory.items() if v > 0]
                if resources:
                    resources_str = ", ".join(resources)

            # Display compact score header
            print(
                f"\rScore: {total_reward:.1f} | Steps: {step_count}/{cfg.renderer_job.num_steps} | Resources: {resources_str}",
                end="",
                flush=True,
            )

            # Render with ASCII renderer
            env.render()

            # Reset if episode done
            if terminals.any():
                print("🏁 Episode finished, resetting...")
                obs, _info = env.reset()

            # Optional sleep for visualization
            if cfg.renderer_job.sleep_time > 0:
                import time

                time.sleep(cfg.renderer_job.sleep_time)

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    finally:
        env.close()

    print(f"\n🎯 Final Results: {total_reward:.3f} reward over {step_count:,} steps")


metta_script(main, "renderer_job")
