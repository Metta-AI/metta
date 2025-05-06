import os

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf


def find_repository_root(start_path=None):
    """
    Find the repository root by traversing up directories looking for common repository markers.

    Args:
        start_path: Path to start searching from. Defaults to the current file's directory.

    Returns:
        String path to the repository root.

    Raises:
        ValueError: If repository root cannot be found.
    """
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))

    current_path = start_path

    # Common repository root markers
    repo_markers = [".git"]

    # Traverse up until we find a repo marker or hit the filesystem root
    while current_path != os.path.dirname(current_path):  # Stop at filesystem root
        for marker in repo_markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path

        # Move up one directory
        current_path = os.path.dirname(current_path)

    # If we reach here, we've hit the filesystem root without finding a repo marker
    raise ValueError("Repository root not found. Make sure you're within a valid repository.")


repo_root = find_repository_root()
mettagrid_configs_root = os.path.join(repo_root, "mettagrid", "configs")
scenes_root = os.path.join(mettagrid_configs_root, "scenes")

print(f"Repository root: {repo_root}")
print(f"Repository root: {repo_root}")
print(f"Repository root: {repo_root}")


# proxy to hydra.utils.instantiate
# mettagrid doesn't load configs through hydra anymore, but it still needs this function
def simple_instantiate(cfg: DictConfig, recursive: bool = False):
    return hydra.utils.instantiate(cfg, _recursive_=recursive)


def get_test_basic_cfg():
    return get_cfg("test_basic")


def get_cfg(config_name: str):
    cfg = OmegaConf.load(f"{mettagrid_configs_root}/{config_name}.yaml")
    assert isinstance(cfg, DictConfig)
    return cfg
