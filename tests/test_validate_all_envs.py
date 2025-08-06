import os
from typing import Any, cast

import pytest
from botocore.exceptions import NoCredentialsError, ProfileNotFound
from omegaconf import OmegaConf

from metta.common.util.mettagrid_cfgs import MettagridCfgFileMetadata
from metta.common.util.resolvers import register_resolvers
from metta.map.utils.storable_map import StorableMap
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

register_resolvers()

# Define which specific configs are known to be slow
ADD_PYTEST_MARK_SLOW = {
    "game/map_builder/auto.yaml",
    "game/map_builder/random_scene.yaml",
}

# Extremely slow configs that should be excluded in CI but run locally
EXCLUDE_PATTERNS_IN_CI = {"game/map_builder/wfc_demo.yaml", "game/map_builder/convchain.yaml"}


def map_or_env_configs() -> list[MettagridCfgFileMetadata]:
    metadata_by_kind = MettagridCfgFileMetadata.get_all()
    result = metadata_by_kind["map"] + metadata_by_kind["env"]

    # If this test is failing and you have configs that are too hard to fix
    # properly, you can add them to this list.
    exclude_patterns = [
        # uses a class that doesn't exist
        "multiagent/experiments/cylinder_world.yaml",
        # usually incomplete
        "defaults.yaml",
        # partials
        "mettagrid.yaml",
        "game/agent",
        "game/groups",
        "game/objects",
        "game/reward_sharing",
        # have unset params
        "multiagent/experiments/varied_terrain.yaml",
        "game/map_builder/load.yaml",
        "game/map_builder/load_random.yaml",
        "navigation/training/sparse.yaml",
        "object_use/training/easy_all_objects.yaml",
        # These are curricula, not envs
        "navigation/training/",
        # These are broken into different files to work around curriculum needs. They don't load right in this test.
        "cooperation/experimental/",
        "navigation_sequence/experiments",
        "multiagent/experiments/",
        "multiagent/multiagent/",
    ]

    # In CI, also exclude extremely slow configs
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        exclude_patterns.extend(EXCLUDE_PATTERNS_IN_CI)

    # exclude some configs that won't work
    result = [cfg for cfg in result if not any(pattern in cfg.path for pattern in exclude_patterns)]

    return result


def get_all_configs_with_marks():
    """Generate test parameters with appropriate marks."""
    configs = map_or_env_configs()
    params = []

    for cfg in configs:
        marks = []

        # Mark slow configs
        if cfg.path in ADD_PYTEST_MARK_SLOW:
            marks.append(pytest.mark.slow)

        # Mark extremely slow configs (these will be excluded in CI by map_or_env_configs)
        if cfg.path in EXCLUDE_PATTERNS_IN_CI:
            marks.append(pytest.mark.slow)

        if marks:
            params.append(pytest.param(cfg, marks=marks, id=cfg.path))
        else:
            params.append(pytest.param(cfg, id=cfg.path))

    return params


# TODO: This should probably be switched to "is this config either an env or a curriculum" or something, so we need
# fewer exceptions. We could also standardize naming to help with this.
@pytest.mark.parametrize("cfg_metadata", get_all_configs_with_marks())
class TestValidateAllEnvs:
    def test_map(self, cfg_metadata: MettagridCfgFileMetadata):
        try:
            map_cfg = cfg_metadata.get_cfg().get_map_cfg()
            StorableMap.from_cfg(map_cfg)
        except (NoCredentialsError, ProfileNotFound) as e:
            pytest.skip(f"Skipping {cfg_metadata.path} because it requires AWS credentials: {e}")
        except Exception as e:
            pytest.fail(f"Failed to validate map config {cfg_metadata.path}: {e}")

    def test_mettagrid_config(self, cfg_metadata: MettagridCfgFileMetadata):
        if cfg_metadata.kind != "env":
            return

        cfg = cfg_metadata.get_cfg().cfg.game
        OmegaConf.resolve(cfg)
        game_config_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(game_config_dict, dict)

        # uncomment for debugging
        print(OmegaConf.to_yaml(OmegaConf.create(game_config_dict)))

        from_mettagrid_config(cast(dict[str, Any], game_config_dict))
