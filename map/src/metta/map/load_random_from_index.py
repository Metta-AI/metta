import random

from metta.map.load import Load
from metta.mettagrid.util import file as file_utils

from .types import SceneCfg


class LoadRandomFromIndex(Load):
    """
    Load a random map from a list of pregenerated maps.

    The index file can be produced with the following command:
        python -m tools.index_s3_maps --dir=s3://...

    See also: `LoadRandom` for a version that loads a random map from an S3 directory.
    """

    def __init__(self, index_uri: str, extra_root: SceneCfg | None = None):
        self._index_uri = index_uri

        # For 10k maps in a directory we'd have to fetch 100Kb of index data.
        # (Can we optimize this further by caching?)
        index = file_utils.read(self._index_uri).decode()
        index = index.split("\n")
        random_map_uri = random.choice(index)

        super().__init__(random_map_uri, extra_root)
