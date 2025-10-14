import pytest

from mettagrid.config.mettagrid_config import AssemblerConfig, RecipeConfig


def test_assembler_config_allows_disjoint_patterns() -> None:
    AssemblerConfig(
        type_id=7,
        recipes=[
            (["N"], RecipeConfig()),
            (["E"], RecipeConfig()),
        ],
    )


def test_assembler_config_allows_partially_overlapping_patterns() -> None:
    AssemblerConfig(
        type_id=7,
        recipes=[
            (["N", "Any"], RecipeConfig()),
            (["Any", "Any"], RecipeConfig()),
        ],
    )


def test_assembler_config_rejects_fully_overlapping_patterns() -> None:
    with pytest.raises(ValueError, match="has no valid cog patterns"):
        AssemblerConfig(
            type_id=7,
            recipes=[
                (["Any", "Any"], RecipeConfig()),
                (["N", "Any"], RecipeConfig()),
            ],
        )
