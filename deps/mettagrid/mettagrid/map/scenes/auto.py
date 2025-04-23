from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import Node

from mettagrid.map.scene import Scene, TypedChild
from mettagrid.map.scenes.bsp import BSPLayout
from mettagrid.map.scenes.make_connected import MakeConnected
from mettagrid.map.scenes.mirror import Mirror
from mettagrid.map.scenes.random import Random
from mettagrid.map.scenes.random_objects import RandomObjects
from mettagrid.map.scenes.random_scene import RandomScene
from mettagrid.map.scenes.room_grid import RoomGrid
from mettagrid.map.utils.random import MaybeSeed, sample_int_distribution


# Global config for convenience.
# Never instantiated, used as a duck type for OmegaConf configs.
@dataclass
class AutoConfig:
    num_agents: int = 0
    grid: Any = field(default_factory=dict)
    room_symmetry: Any = field(default_factory=dict)
    content: Any = field(default_factory=list)
    bsp: Any = field(default_factory=dict)
    layout: Any = field(default_factory=dict)
    objects: Any = field(default_factory=dict)
    room_objects: Any = field(default_factory=dict)
    # TODO - stricter types


class BaseAuto(Scene):
    def __init__(self, config: AutoConfig, rng: np.random.Generator):
        super().__init__(children=[])
        self._config = config
        self._rng = np.random.default_rng(rng)

    def _render(self, node: Node):
        pass


class Auto(BaseAuto):
    def __init__(self, config: AutoConfig, seed: MaybeSeed = None):
        rng = np.random.default_rng(seed)
        super().__init__(config, rng=rng)

    def get_children(self, node) -> list[TypedChild]:
        return [
            {"scene": AutoLayout(config=self._config, rng=self._rng), "where": "full"},
            {
                "scene": Random(objects=self._config.objects, seed=self._rng),
                "where": "full",
            },
            {"scene": MakeConnected(seed=self._rng), "where": "full"},
            {
                "scene": Random(agents=self._config.num_agents, seed=self._rng),
                "where": "full",
            },
        ]


class AutoLayout(BaseAuto):
    def get_children(self, node) -> list[TypedChild]:
        weights = np.array([self._config.layout.grid, self._config.layout.bsp], dtype=np.float32)
        weights /= weights.sum()
        layout = self._rng.choice(["grid", "bsp"], p=weights)

        def children_for_tag(tag: str) -> list[TypedChild]:
            return [
                {
                    "scene": AutoSymmetry(config=self._config, rng=self._rng),
                    "where": {"tags": [tag]},
                },
                {
                    "scene": RandomObjects(object_ranges=self._config.room_objects, seed=self._rng),
                    "where": {"tags": [tag]},
                },
            ]

        if layout == "grid":
            rows = sample_int_distribution(self._config.grid.rows, self._rng)
            columns = sample_int_distribution(self._config.grid.columns, self._rng)

            return [
                {
                    "scene": RoomGrid(
                        rows=rows,
                        columns=columns,
                        border_width=0,  # randomize? probably not very useful
                        children=children_for_tag("room"),
                    ),
                    "where": "full",
                },
            ]
        elif layout == "bsp":
            area_count = sample_int_distribution(self._config.bsp.area_count, self._rng)

            return [
                {
                    "scene": BSPLayout(
                        area_count=area_count,
                        children=children_for_tag("zone"),
                        seed=self._rng,
                    ),
                    "where": "full",
                }
            ]
        else:
            raise ValueError(f"Invalid layout: {layout}")


class AutoSymmetry(BaseAuto):
    def get_children(self, node) -> list[TypedChild]:
        weights = np.array(
            [
                self._config.room_symmetry.none,
                self._config.room_symmetry.horizontal,
                self._config.room_symmetry.vertical,
                self._config.room_symmetry.x4,
            ],
            dtype=np.float32,
        )
        weights /= weights.sum()
        symmetry = self._rng.choice(["none", "horizontal", "vertical", "x4"], p=weights)
        scene = RandomScene(candidates=self._config.content, seed=self._rng)
        if symmetry != "none":
            scene = Mirror(scene, symmetry)
        return [{"scene": scene, "where": "full"}]
