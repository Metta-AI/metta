import mettagrid.mapgen.random.float
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.random


class RandomObjectsConfig(mettagrid.mapgen.scene.SceneConfig):
    object_ranges: dict[str, mettagrid.mapgen.random.float.FloatDistribution] = {}


class RandomObjects(mettagrid.mapgen.scene.Scene[RandomObjectsConfig]):
    """
    Fill the grid with random objects. Unlike Random, this scene takes the percentage ranges of objects,
    not the absolute count.

    It's rarely useful to pick the random number of agents, so this scene doesn't have that parameter.
    """

    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        size = self.height * self.width
        objects = {}
        for obj_name, distribution in self.config.object_ranges.items():
            percentage = distribution.sample(self.rng)
            objects[obj_name] = int(size * percentage)

        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.random.Random.Config(objects=objects),
                where="full",
            ),
        ]

    def render(self):
        pass
