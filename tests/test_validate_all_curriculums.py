import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from metta.common.util.mettagrid_cfgs import METTAGRID_CFG_ROOT, MettagridCfgFileMetadata
from metta.common.util.resolvers import register_resolvers
from metta.mettagrid.curriculum import Curriculum, curriculum_from_config_path

register_resolvers()


@pytest.fixture(scope="module", autouse=True)
def hydra_init():
    """Initialize Hydra once for all tests in this module."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base=None)
    yield
    GlobalHydra.instance().clear()


def curriculum_configs() -> list[MettagridCfgFileMetadata]:
    metadata_by_kind = MettagridCfgFileMetadata.get_all()
    result = metadata_by_kind["curriculum"]

    # If this test is failing and you have configs that are too hard to fix
    # properly, you can add them to this list.
    exclude_patterns = [
        "curriculum/progressive.yaml",  # partial
    ]

    for p in (cfg.path for cfg in result):
        print(p)

    # exclude some configs that won't work
    result = [cfg for cfg in result if not any(pattern in cfg.path for pattern in exclude_patterns)]

    return result


@pytest.mark.parametrize("cfg_metadata", curriculum_configs(), ids=[cfg.path for cfg in curriculum_configs()])
class TestValidateAllCurricula:
    def test_curriculum(self, cfg_metadata: MettagridCfgFileMetadata):
        # Test the curriculum_from_config_path function as it will be used by the trainer
        cfg_path = METTAGRID_CFG_ROOT + "/" + cfg_metadata.path
        env_overrides = OmegaConf.create({})  # Empty overrides for testing

        print(f"\nTesting curriculum: {cfg_metadata.path}")

        # This is exactly how the trainer will load curricula
        tree = curriculum_from_config_path(cfg_path, env_overrides)

        assert isinstance(tree, Curriculum)

        # Print some debug info
        print(f"  - Created Curriculum with {tree.num_tasks} children")

        # Test that we can sample a task from the tree
        task = tree.sample()
        assert hasattr(task, "env_config")
        assert hasattr(task, "name")
        print(f"  - Sampled task: {task.name}")
