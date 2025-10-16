"""Interactive play tool for Metta simulations."""

import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from rich.console import Console

from metta.agent.utils import obs_to_td
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.training_environment import EnvironmentMetaData
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config
from mettagrid import MettaGridEnv, RenderMode, dtype_actions

logger = logging.getLogger(__name__)

DIRECTION_ACTION_NAMES: dict[int, str] = {
    0: "move_north",
    1: "move_south",
    2: "move_west",
    3: "move_east",
    4: "move_northwest",
    5: "move_northeast",
    6: "move_southwest",
    7: "move_southeast",
}


def _flatten_action_request(
    action_request: SimpleNamespace,
    *,
    total_actions: int,
    noop_action_id: int,
    move_action_lookup: dict[int, int],
) -> int:
    """Translate a MettaScope ActionRequest into a flattened action index."""

    raw_action_id = int(getattr(action_request, "action_id", -1))
    if 0 <= raw_action_id < total_actions:
        return raw_action_id

    argument_value = getattr(action_request, "argument", None)
    if argument_value is not None:
        orientation_idx = int(argument_value)
        flattened_move = move_action_lookup.get(orientation_idx)
        if flattened_move is not None:
            return flattened_move

    logger.debug(
        "Received unrecognized manual action; defaulting to noop",
        extra={
            "action_id": raw_action_id,
            "argument": getattr(action_request, "argument", None),
        },
    )
    return noop_action_id


class PlayTool(Tool):
    """Interactive play tool for Metta simulations using MettaScope2."""

    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    max_steps: Optional[int] = None
    seed: int = 42
    render: RenderMode = "gui"

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run an interactive play session with the configured simulation."""

        console = Console()
        device = torch.device(self.system.device)

        # Create environment directly with render mode
        env = MettaGridEnv(env_cfg=self.sim.env, render_mode=self.render)

        # Load policy if provided, otherwise use mock agent (random actions)
        if self.policy_uri:
            logger.info(f"Loading policy from {self.policy_uri}")
            policy = CheckpointManager.load_from_uri(self.policy_uri)

            # Create environment metadata for policy initialization
            env_metadata = EnvironmentMetaData(
                obs_width=env.obs_width,
                obs_height=env.obs_height,
                obs_features=env.observation_features,
                action_names=env.action_names,
                num_agents=env.num_agents,
                observation_space=env.observation_space,
                action_space=env.single_action_space,
                feature_normalizations=env.feature_normalizations,
            )

            # Initialize policy to environment
            policy.eval()
            policy.initialize_to_environment(env_metadata, device)
        else:
            logger.info("No policy specified, using random actions")
            policy = None

        # Build action lookup tables
        action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
        noop_action_id = action_lookup.get("noop", 0)
        move_action_lookup = {
            orientation: action_lookup[name]
            for orientation, name in DIRECTION_ACTION_NAMES.items()
            if name in action_lookup
        }

        # Reset environment
        obs, _ = env.reset(seed=self.seed)

        # Initialize game state
        step_count = 0
        num_agents = env.num_agents
        actions = np.zeros(num_agents, dtype=dtype_actions)
        total_rewards = np.zeros(num_agents)

        # Main game loop
        while self.max_steps is None or step_count < self.max_steps:
            # Check if renderer wants to continue (e.g., user quit or interactive loop finished)
            if not env._renderer.should_continue():
                break

            # Render the environment (handles display and user input)
            env.render()

            # Get user actions from renderer (if any)
            user_actions = env._renderer.get_user_actions()

            # Get actions - use user input if available, otherwise use policy
            for agent_id in range(num_agents):
                if agent_id in user_actions:
                    # User provided action for this agent
                    action_id, action_param = user_actions[agent_id]
                    # Flatten the action using the helper function
                    actions[agent_id] = _flatten_action_request(
                        SimpleNamespace(action_id=action_id, argument=action_param),
                        total_actions=len(env.action_names),
                        noop_action_id=noop_action_id,
                        move_action_lookup=move_action_lookup,
                    )
                else:
                    # Use policy action
                    if policy is not None:
                        # Convert single agent observation to TensorDict and get action
                        agent_obs = obs[agent_id : agent_id + 1]  # Keep dimension
                        td = obs_to_td(agent_obs, device)
                        policy(td)
                        actions[agent_id] = td["actions"][0].item()
                    else:
                        # Random action if no policy
                        actions[agent_id] = np.random.randint(0, len(env.action_names))

            # Step the environment
            obs, rewards, dones, truncated, _ = env.step(actions)

            # Update total rewards
            total_rewards += rewards
            step_count += 1

            if all(dones) or all(truncated):
                break

        # Print summary
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {step_count}")
        console.print(f"Total Rewards: {total_rewards}")
        console.print(f"Final Reward Sum: {float(sum(total_rewards)):.2f}")

        return None
