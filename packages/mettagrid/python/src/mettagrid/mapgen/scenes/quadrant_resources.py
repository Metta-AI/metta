from typing import List

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.radial_objects import RadialObjects, RadialObjectsParams


class QuadrantResourcesParams(Config):
    resource_types: List[str] = ["generator_red", "generator_blue", "generator_green", "lab"]
    count_per_quadrant: int = 5
    k: float = 3.0
    min_radius: int = 6
    clearance: int = 1
    forced_type: str | None = None


class QuadrantResources(Scene[QuadrantResourcesParams]):
    """
    Place a single resource type in THIS area (intended to be one quadrant), edge-biased.
    Use with where=AreaWhere(tags=["quadrant"]) so it is instantiated once per quadrant.
    """

    def get_children(self) -> list[ChildrenAction]:
        # Choose one type for this instance
        if self.params.forced_type is not None:
            t = self.params.forced_type
        else:
            idx = int(self.rng.integers(0, len(self.params.resource_types)))
            t = self.params.resource_types[idx]

        return [
            ChildrenAction(
                scene=RadialObjects.factory(
                    RadialObjectsParams(
                        objects={t: self.params.count_per_quadrant},
                        k=self.params.k,
                        min_radius=self.params.min_radius,
                        clearance=self.params.clearance,
                        carve=True,
                    )
                ),
                where="full",
                lock="resources",
                limit=1,
                order_by="first",
            )
        ]

    def render(self):
        pass
