import os
import random
from pathlib import Path

from metta.map.load import Load
from metta.map.utils import s3utils

from .types import SceneCfg


def parse_file_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri.split("file://")[1]

    # we don't support any other schemes
    if "://" in uri:
        raise ValueError(f"Invalid URI: {uri}")

    # probably a local file name
    return uri


def get_random_map_uri(dir_uri: str) -> str:
    if dir_uri.startswith("s3://"):
        filenames = s3utils.list_objects(dir_uri)
        filenames = [uri for uri in filenames if uri.endswith(".yaml")]
        if not filenames:
            raise ValueError(f"No maps found in {dir_uri}")
        return random.choice(filenames)
    else:
        dirname = parse_file_uri(dir_uri)
        if not os.path.isdir(dirname):
            raise ValueError(f"Directory {dirname} does not exist")

        filenames = os.listdir(dirname)
        filenames = [Path(dirname) / Path(filename) for filename in filenames if filename.endswith(".yaml")]
        if not filenames:
            raise ValueError(f"No maps found in {dirname}")
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
