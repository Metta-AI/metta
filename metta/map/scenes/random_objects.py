from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.scenes.random import Random
from metta.map.types import ChildrenAction
from metta.map.utils.random import FloatDistribution, sample_float_distribution


class RandomObjectsParams(Config):
    object_ranges: dict[str, FloatDistribution] = {}


class RandomObjects(Scene[RandomObjectsParams]):
    """
    Fill the grid with random objects. Unlike Random, this scene takes the percentage ranges of objects,
    not the absolute count.

    It's rarely useful to pick the random number of agents, so this scene doesn't have that parameter.
    """

    def get_children(self) -> list[ChildrenAction]:
        size = self.height * self.width
        objects = {}
        for obj_name, distribution in self.params.object_ranges.items():
            percentage = sample_float_distribution(distribution, self.rng)
            objects[obj_name] = int(size * percentage)

        return [
            ChildrenAction(
                scene=lambda grid: Random(grid=grid, params={"objects": objects}, seed=self.rng),
                where="full",
            ),
            *self.children,
        ]

    def render(self):
        pass
