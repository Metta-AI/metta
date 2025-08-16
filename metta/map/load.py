from typing import cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.map.scene import SceneCfg, make_scene
from metta.map.utils.storable_map import StorableMap
from metta.mettagrid.map_builder.map_builder import GameMap

from .types import Area


# Note that this class can't be a scene, because the width and height come from the stored data.
class Load(GameMap):
    """
    Load a pregenerated map from a URI (file or S3 object).

    See also: `FromS3Dir` for picking a random map from a directory of pregenerated maps.
    """

    _extra_root: dict | None = None

    def __init__(self, uri: str, extra_root: SceneCfg | DictConfig | None = None):
        super().__init__(grid=None)
        self._uri = uri
        self._storable_map = StorableMap.from_uri(uri)

        if isinstance(extra_root, DictConfig):
            extra_root = cast(dict, OmegaConf.to_container(extra_root))

        if isinstance(extra_root, dict):
            self._extra_root = extra_root

    def build(self):
        grid = self._storable_map.grid

        area = Area.root_area_from_grid(grid)

        if self._extra_root is not None:
            root_scene = make_scene(self._extra_root, area, rng=np.random.default_rng())
            root_scene.render_with_children()

        return GameMap(grid=grid)
