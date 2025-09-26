from typing import List

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.radial_objects import RadialObjects, RadialObjectsParams
from mettagrid.mapgen.types import AreaWhere


class QuadrantResourcesParams(Config):
    resource_types: List[str] = ["generator_red", "generator_blue", "generator_green", "lab"]
    count_per_quadrant: int = 5
    k: float = 3.0
    min_radius: int = 6
    clearance: int = 1


class QuadrantResources(Scene[QuadrantResourcesParams]):
    """
    Randomly assign distinct resource types to the four quadrants and place them
    with edge-biased distribution in each quadrant.
    """

    def get_children(self) -> list[ChildrenAction]:
        types = list(self.params.resource_types)
        if len(types) >= 4:
            chosen = list(self.rng.choice(types, size=4, replace=False))
        else:
            # If fewer than 4 provided, allow repetition
            chosen = list(self.rng.choice(types, size=4, replace=True))

        actions: list[ChildrenAction] = []
        for i, t in enumerate(chosen):
            actions.append(
                ChildrenAction(
                    scene=RadialObjects.factory(
                        RadialObjectsParams(
                            objects={t: self.params.count_per_quadrant},
                            k=self.params.k,
                            min_radius=self.params.min_radius,
                            clearance=self.params.clearance,
                        )
                    ),
                    where=AreaWhere(tags=[f"q.{i}"]),
                    lock="resources",
                    limit=1,
                    order_by="first",
                )
            )

        return [*actions, *self.children_actions]

    def render(self):
        # Create local quadrant areas (2x2 split of current area)
        h, w = self.height, self.width
        mx, my = w // 2, h // 2
        # top-left
        self.make_area(0, 0, mx, my, tags=["q", "q.0"])
        # top-right
        self.make_area(mx, 0, w - mx, my, tags=["q", "q.1"])
        # bottom-left
        self.make_area(0, my, mx, h - my, tags=["q", "q.2"])
        # bottom-right
        self.make_area(mx, my, w - mx, h - my, tags=["q", "q.3"])
