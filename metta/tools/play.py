# Starts a websocket server that allows you to play as a metta agent.

import json
import logging
from typing import ClassVar

import numpy as np
import torch as torch

from metta.common.tool import Tool
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.wandb.context import WandbConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config
from mettagrid.util.grid_object_formatter import format_grid_object

logger = logging.getLogger(__name__)


class PlayTool(Tool):
    tool_name: ClassVar[str] = "play"
    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    replay_dir: str | None = None
    stats_dir: str | None = None
    open_browser_on_start: bool = True
    mettascope2: bool = False

    @property
    def effective_replay_dir(self) -> str:
        """Get the replay directory, defaulting to system.data_dir/replays if not specified."""
        return self.replay_dir if self.replay_dir is not None else f"{self.system.data_dir}/replays"

    @property
    def effective_stats_dir(self) -> str:
        """Get the stats directory, defaulting to system.data_dir/stats if not specified."""
        return self.stats_dir if self.stats_dir is not None else f"{self.system.data_dir}/stats"

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.mettascope2:
            import mettagrid.mettascope as mettascope2

            sim = Simulation.create(
                sim_config=self.sim,
                device=self.system.device,
                vectorization=self.system.vectorization,
                stats_dir=self.effective_stats_dir,
                replay_dir=self.effective_replay_dir,
                policy_uri=self.policy_uri,
            )
            sim.start_simulation()
            env = sim.get_env()
            initial_replay = sim.get_replay()

            current_step = 0
            actions = np.zeros((env.num_agents, 2))
            total_rewards = np.zeros(env.num_agents)

            response = mettascope2.init(replay=json.dumps(initial_replay))
            if response.should_close:
                return

            def send_replay_step():
                grid_objects = []
                for i, grid_object in enumerate(env.grid_objects().values()):
                    if len(grid_objects) <= i:
                        grid_objects.append({})

                    if "agent_id" in grid_object:
                        agent_id = grid_object["agent_id"]
                        total_rewards[agent_id] += env.rewards[agent_id]

                    update_object = format_grid_object(
                        grid_object, actions, env.action_success, env.rewards, total_rewards
                    )

                    grid_objects[i] = update_object

                step_replay = {"step": current_step, "objects": grid_objects}
                return json.dumps(step_replay)

            while True:
                replay_step = send_replay_step()
                response = mettascope2.render(current_step, replay_step)
                if response.should_close:
                    break

                actions = sim.generate_actions()
                # Just do random actions for now.
                actions[:, 0] = np.random.randint(0, 5, size=len(actions))  # Random action types
                actions[:, 1] = np.random.randint(0, 4, size=len(actions))  # Random action args
                for action in response.actions:
                    actions[action.agent_id, 0] = action.action_id
                    actions[action.agent_id, 1] = action.argument

                sim.step_simulation(actions)
                current_step += 1

            sim.end_simulation()

        else:
            import mettascope.server as server

            ws_url = "%2Fws"

            if self.open_browser_on_start:
                server.run(self, open_url=f"?wsUrl={ws_url}")
            else:
                logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
                server.run(self)
