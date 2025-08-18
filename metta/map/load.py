import numpy as np

from metta.map.scene import SceneConfigOrFile, make_scene
from metta.map.utils.storable_map import StorableMap
from metta.mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig

from .types import Area


# Note that this class can't be a scene, because the width and height come from the stored data.
class Load(MapBuilder):
    """
    Load a pregenerated map from a URI (file or S3 object).

    See also: `FromS3Dir` for picking a random map from a directory of pregenerated maps.
    """

    class Config(MapBuilderConfig["Load"]):
        uri: str
        extra_root: SceneConfigOrFile | None = None

    def __init__(self, config: Config):
        self.config = config

    def build(self):
        _storable_map = StorableMap.from_uri(self.config.uri)
        grid = _storable_map.grid

        area = Area.root_area_from_grid(grid)

        if self.config.extra_root is not None:
            root_scene = make_scene(self.config.extra_root, area, rng=np.random.default_rng())
            root_scene.render_with_children()

        return GameMap(grid=grid)
