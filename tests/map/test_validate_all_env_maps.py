import pytest
from botocore.exceptions import NoCredentialsError, ProfileNotFound

from metta.common.util.mettagrid_cfgs import MettagridCfgFileMetadata
from metta.common.util.resolvers import register_resolvers
from metta.map.utils.storable_map import map_builder_cfg_to_storable_map

register_resolvers()


def map_or_env_configs() -> list[MettagridCfgFileMetadata]:
    metadata_by_kind = MettagridCfgFileMetadata.get_all()
    result = metadata_by_kind["map"] + metadata_by_kind["env"]

    # If this test is failing and you have configs that are too hard to fix
    # properly, you can add them to this list.
    exclude_patterns = [
        "multiagent/experiments",
        "multiagent/multiagent",
        # "memory/evals",
        "mettagrid.yaml",
        # usually incomplete
        "defaults",
        # have unset params
        "game/",
    ]

    # exclude some configs that won't work
    result = [cfg for cfg in result if not any(pattern in cfg.path for pattern in exclude_patterns)]

    return result


@pytest.mark.parametrize("cfg_metadata", map_or_env_configs(), ids=[cfg.path for cfg in map_or_env_configs()])
def test_validate_cfg(cfg_metadata):
    try:
        map_cfg = cfg_metadata.get_cfg().get_map_cfg()
        map_builder_cfg_to_storable_map(map_cfg)
    except (NoCredentialsError, ProfileNotFound) as e:
        pytest.skip(f"Skipping {cfg_metadata.path} because it requires AWS credentials: {e}")
    except Exception as e:
        pytest.fail(f"Failed to validate map config {cfg_metadata.path}: {e}")
