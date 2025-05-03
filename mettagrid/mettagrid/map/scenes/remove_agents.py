from mettagrid.map.node import Node
from mettagrid.map.scene import Scene, TypedChild


class RemoveAgents(Scene):
    """
    This class solves a frequent problem: `game.num_agents` must match the
    number of agents in the map.

    You can use this scene to remove agents from the map. Then apply `Random`
    scene to place as many agents as you want.

    (TODO - it might be better to remove `game.num_agents` from the config
    entirely, and just use the number of agents in the map.)
    """

    def __init__(self, children: list[TypedChild] | None = None):
        super().__init__(children=children)

    def _render(self, node: Node):
        for i in range(node.height):
            for j in range(node.width):
                value = node.grid[i, j]
                if value.startswith("agent.") or value == "agent":
                    node.grid[i, j] = "empty"
