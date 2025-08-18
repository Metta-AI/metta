import os
import pathlib


def get_asset_path():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "assets")


def get_config_path():
    path = pathlib.Path.home() / ".config/mettagrid"
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())
