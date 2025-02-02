from typing import Optional
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room.room import Room, GameObject

class Random(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 0,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall"
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._objects = objects
        self._agents = agents

    def _build(self):
        symbols = []
        area = self._width * self._height

        if isinstance(self._agents, int):
            agents = ["agent.agent"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = ["agent." + agent for agent, na in self._agents.items() for _ in range(na)]

        # Check if total objects exceed room size and halve counts if needed
        total_objects = sum(count for count in self._objects.values()) + len(agents)
        while total_objects > 2*area / 3:
            for obj_name in self._objects:
                self._objects[obj_name] = max(1, self._objects[obj_name] // 2)
                total_objects = sum(count for count in self._objects.values()) + len(agents)

        # Add all objects in the proper amounts to a single large array.
        for obj_name, count in self._objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)

        assert(len(symbols) <= area), f"Too many objects in room: {len(symbols)} > {area}"
        symbols.extend(["empty"] * (area - len(symbols)))

        # Shuffle and reshape the array into a room.
        symbols = np.array(symbols).astype(str)
        self._rng.shuffle(symbols)
        return symbols.reshape(self._height, self._width)
