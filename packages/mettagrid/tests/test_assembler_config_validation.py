import pytest

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import AssemblerConfig, GameConfig, RecipeConfig
from mettagrid.mettagrid_c import GameConfig as CppGameConfig


def make_env_cfg_from_assembler_config(assembler_config: AssemblerConfig) -> CppGameConfig:
    return convert_to_cpp_game_config(
        GameConfig(
            num_agents=2,
            objects={
                "assembler": assembler_config,
            },
        )
    )


def test_assembler_config_allows_disjoint_patterns() -> None:
    make_env_cfg_from_assembler_config(
        AssemblerConfig(
            type_id=7,
            recipes=[
                (["N"], RecipeConfig()),
                (["E"], RecipeConfig()),
            ],
        )
    )


def test_assembler_config_allows_partially_overlapping_patterns() -> None:
    make_env_cfg_from_assembler_config(
        AssemblerConfig(
            type_id=7,
            recipes=[
                (["Any", "Any"], RecipeConfig()),
                (["N", "Any"], RecipeConfig()),
            ],
        )
    )


def test_assembler_config_rejects_fully_overlapping_patterns_unless_explicitly_allowed() -> None:
    with pytest.raises(ValueError, match="has no valid cog patterns"):
        make_env_cfg_from_assembler_config(
            AssemblerConfig(
                type_id=7,
                recipes=[
                    (["N", "Any"], RecipeConfig()),
                    (["Any", "Any"], RecipeConfig()),
                ],
            )
        )
    make_env_cfg_from_assembler_config(
        AssemblerConfig(
            type_id=7,
            recipes=[
                (["N", "Any"], RecipeConfig()),
                (["Any", "Any"], RecipeConfig()),
            ],
            fully_overlapping_recipes_allowed=True,
        )
    )
