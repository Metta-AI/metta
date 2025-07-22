from typing import Any, cast

import pytest
from botocore.exceptions import NoCredentialsError, ProfileNotFound
from omegaconf import OmegaConf

from metta.common.util.mettagrid_cfgs import MettagridCfgFileMetadata
from metta.common.util.resolvers import register_resolvers
from metta.map.utils.storable_map import StorableMap
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

register_resolvers()


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
        # Is a curriculum, not an env
        "navigation/training/sparse_bucketed",
        # These are broken into different files to work around curriculum needs. They don't load right in this test.
        "cooperation/experimental/",
    ]

    # exclude some configs that won't work
    result = [cfg for cfg in result if not any(pattern in cfg.path for pattern in exclude_patterns)]

    return result


@pytest.mark.parametrize("cfg_metadata", map_or_env_configs(), ids=[cfg.path for cfg in map_or_env_configs()])
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
