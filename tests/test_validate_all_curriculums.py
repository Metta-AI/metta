import hydra
import pytest

from metta.common.util.mettagrid_cfgs import MettagridCfgFileMetadata
from metta.common.util.resolvers import register_resolvers
from metta.mettagrid.curriculum.core import Curriculum

register_resolvers()


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
class TestValidateAllCurriculums:
    @pytest.mark.slow
    def test_curriculum(self, cfg_metadata: MettagridCfgFileMetadata):
        cfg = cfg_metadata.get_cfg().cfg
        with hydra.initialize(version_base=None, config_path="../configs"):
            curriculum = hydra.utils.instantiate(cfg, _recursive_=True)
        assert isinstance(curriculum, Curriculum)
