from metta.common.util.config import Config
from metta.map.scene import Scene


class RemoveAgentsParams(Config):
    pass


class RemoveAgents(Scene[RemoveAgentsParams]):
    """
    This class solves a frequent problem: `game.num_agents` must match the
    number of agents in the map.

    You can use this scene to remove agents from the map. Then apply `Random`
    scene to place as many agents as you want.

    (TODO - it might be better to remove `game.num_agents` from the config
    entirely, and just use the number of agents in the map.)
    """

    def render(self):
        for i in range(self.height):
            for j in range(self.width):
                value = self.grid[i, j]
                if value.startswith("agent.") or value == "agent":
                    self.grid[i, j] = "empty"
