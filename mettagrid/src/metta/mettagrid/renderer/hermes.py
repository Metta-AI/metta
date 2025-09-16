import pathlib


def get_config_path():
    path = pathlib.Path.home() / ".config/mettagrid"
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())
