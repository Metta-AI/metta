from mettagrid.config.room.room import Room
from mettagrid.map.utils.storable_map import StorableMap

from .scene import SceneCfg, make_scene


# Note that this class can't be a scene, because the width and height come from the stored data.
class Load(Room):
    """
    Load a pregenerated map from a URI (file or S3 object).

    See also: `FromS3Dir` for picking a random map from a directory of pregenerated maps.
    """

    def __init__(self, uri: str, extra_root: SceneCfg | None = None):
        super().__init__()
        self._uri = uri
        self._storable_map = StorableMap.from_uri(uri)

        if extra_root is not None:
            self._root = make_scene(extra_root)
        else:
            self._root = None

    def build(self):
        grid = self._storable_map.grid

        if self._root is not None:
            root_node = self._root.make_node(grid)
            root_node.render()

        return grid
