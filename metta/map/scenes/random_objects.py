from metta.map.node import Node
from metta.map.scene import TypedChild
from metta.map.scenes.random import Random
from metta.map.utils.random import sample_float_distribution
from metta.util.config import Config


class RandomObjectsParams(Config):
    object_ranges: dict = {}


class RandomObjects(Node):
    """
    Fill the grid with random objects. Unlike Random, this scene takes the percentage ranges of objects,
    not the absolute count.

    It's rarely useful to pick the random number of agents, so this scene doesn't have that parameter.
    """

    params_type = RandomObjectsParams

    def get_children(self) -> list[TypedChild]:
        size = self.height * self.width
        objects = {}
        for obj_name, distribution in self.params.object_ranges.items():
            percentage = sample_float_distribution(distribution, self.rng)
            objects[obj_name] = int(size * percentage)

        return [
            {
                "scene": lambda grid: Random(grid=grid, params={"objects": objects}, seed=self.rng),
                "where": "full",
            },
            *self.children,
        ]

    def render(self):
        pass
