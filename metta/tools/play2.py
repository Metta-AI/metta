# Starts mettascope2 and allows you to play as a metta agent.

import json
import logging
import sys

import numpy as np
import torch as torch

from metta.common.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.mettagrid.grid_object_formatter import format_grid_object
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)

# Add Mettascope2 bindings to the path
sys.path.append("mettascope2/bindings/generated")
# cd into mettascope2 directory
# os.chdir("mettascope2")
import mettascope2


class PlayTool(Tool):
    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    replay_dir: str | None = None
    stats_dir: str | None = None
    open_browser_on_start: bool = True

    @property
    def effective_replay_dir(self) -> str:
        """Get the replay directory, defaulting to system.data_dir/replays if not specified."""
        return self.replay_dir if self.replay_dir is not None else f"{self.system.data_dir}/replays"

    @property
    def effective_stats_dir(self) -> str:
        """Get the stats directory, defaulting to system.data_dir/stats if not specified."""
        return self.stats_dir if self.stats_dir is not None else f"{self.system.data_dir}/stats"

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Invokes the PlayTool and starts the Mettascope2 renderer."""

        # Create a simulation that we are going to play.
        from metta.tools.play import create_simulation

        sim = create_simulation(self)
        sim.start_simulation()
        env = sim.get_env()
        initial_replay = sim.get_replay()

        current_step = 0
        actions = np.zeros((env.num_agents, 2))
        total_rewards = np.zeros(env.num_agents)

        mettascope2.init(data_dir="mettascope2/data", replay=json.dumps(initial_replay))

        def send_replay_step():
            grid_objects = []
            for i, grid_object in enumerate(env.grid_objects.values()):
                if len(grid_objects) <= i:
                    grid_objects.append({})

                if "agent_id" in grid_object:
                    agent_id = grid_object["agent_id"]
                    total_rewards[agent_id] += env.rewards[agent_id]

                update_object = format_grid_object(grid_object, actions, env.action_success, env.rewards, total_rewards)

                grid_objects[i] = update_object

            step_replay = {"step": current_step, "objects": grid_objects}
            return json.dumps(step_replay)

        while True:
            replay_step = send_replay_step()
            should_close = mettascope2.render(current_step, replay_step)
            if should_close:
                break
            actions = sim.generate_actions()
            # TODO: Get actions from mettascope2.
            # Just do random actions for now.
            # Randomize actions in-place
            actions[:, 0] = np.random.randint(0, 5, size=len(actions))  # Random action types
            actions[:, 1] = np.random.randint(0, 4, size=len(actions))  # Random action args
            sim.step_simulation(actions)
            current_step += 1

        sim.end_simulation()


def create_simulation(cfg: PlayTool) -> Simulation:
    """Create a simulation for playing/replaying."""
    logger.info(f"Creating simulation: {cfg.sim.name}")

    # Create simulation using CheckpointManager integration
    sim = Simulation.create(
        sim_config=cfg.sim,
        device=cfg.system.device,
        vectorization=cfg.system.vectorization,
        stats_dir=cfg.effective_stats_dir,
        replay_dir=cfg.effective_replay_dir,
        policy_uri=cfg.policy_uri,
    )
    return sim
