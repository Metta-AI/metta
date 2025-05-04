import os
import random
from pathlib import Path

from mettagrid.map.load import Load
from mettagrid.map.utils import s3utils
from mettagrid.map.utils.storage import parse_file_uri

from .scene import SceneCfg


def get_random_map_uri(dir_uri: str) -> str:
    if dir_uri.startswith("s3://"):
        filenames = s3utils.list_objects(dir_uri)
        filenames = [uri for uri in filenames if uri.endswith(".yaml")]
        return random.choice(filenames)
    else:
        dirname = parse_file_uri(dir_uri)
        if not os.path.isdir(dirname):
            raise ValueError(f"Directory {dirname} does not exist")

        filenames = os.listdir(dirname)
        filenames = [Path(dirname) / Path(filename) for filename in filenames if filename.endswith(".yaml")]
        return str(random.choice(filenames))


class LoadRandom(Load):
    """
    Load a random map from a directory, local or S3.

    See also: `LoadRandomFromIndex` for a version that loads a random map from a pre-generated index.
    """

    def __init__(self, dir: str, extra_root: SceneCfg | None = None):
        self._dir_uri = dir

        random_map_uri = get_random_map_uri(self._dir_uri)

        super().__init__(random_map_uri, extra_root)
