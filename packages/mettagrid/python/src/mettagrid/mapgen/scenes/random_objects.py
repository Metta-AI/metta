from mettagrid.config.config import Config
from mettagrid.mapgen.random.float import FloatDistribution
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.random import Random


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
            percentage = distribution.sample(self.rng)
            objects[obj_name] = int(size * percentage)

        return [
            ChildrenAction(
                scene=Random.factory(Random.Params(objects=objects)),
                where="full",
            ),
            *self.children_actions,
        ]

    def render(self):
        pass
