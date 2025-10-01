"""Tests for automatic percolation-based length scale in ClipperConfig."""

import numpy as np

from mettagrid.config.mettagrid_config import AssemblerConfig, ClipperConfig, GameConfig, RecipeConfig
from mettagrid.map_builder.random import RandomMapBuilder


class TestPercolationLengthScale:
    """Test automatic percolation-based length scale calculations."""

    def test_auto_length_scale_enabled_by_default(self):
        """Test that auto length scale is enabled by default and calculates correctly."""
        game_config = GameConfig(
            map_builder=RandomMapBuilder.Config(agents=10, width=50, height=50),
            objects={
                f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[]) for i in range(25)
            },
            clipper=ClipperConfig(
                recipe=RecipeConfig(input_resources={"ore_red": 5}),
                clip_rate=0.1,
            ),
        )

        # Verify auto_length_scale is True by default
        assert game_config.clipper.auto_length_scale is True

        # Verify length scale is calculated automatically
        # Expected: (50 / sqrt(25)) * sqrt(4.51 / (4 * pi)) = 10 * 0.5991 ≈ 5.991
        expected = (50 / np.sqrt(25)) * np.sqrt(4.51 / (4 * np.pi))
        assert np.isclose(game_config.clipper.length_scale, expected, rtol=1e-5)
        assert np.isclose(game_config.clipper.length_scale, 5.991, atol=0.01)

    def test_auto_length_scale_with_fudge_factor(self):
        """Test that length_scale_factor acts as a fudge factor for auto calculation."""
        for fudge_factor in [0.5, 1.0, 1.5, 2.0]:
            game_config = GameConfig(
                map_builder=RandomMapBuilder.Config(agents=10, width=50, height=50),
                objects={
                    f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[])
                    for i in range(25)
                },
                clipper=ClipperConfig(
                    recipe=RecipeConfig(input_resources={"ore_red": 5}),
                    clip_rate=0.1,
                    length_scale_factor=fudge_factor,
                ),
            )

            # Expected: base_scale * fudge_factor
            base_scale = (50 / np.sqrt(25)) * np.sqrt(4.51 / (4 * np.pi))
            expected = base_scale * fudge_factor
            assert np.isclose(game_config.clipper.length_scale, expected, rtol=1e-5)

    def test_manual_length_scale_when_auto_disabled(self):
        """Test that manual length_scale is used when auto_length_scale=False."""
        manual_value = 3.14
        game_config = GameConfig(
            map_builder=RandomMapBuilder.Config(agents=10, width=50, height=50),
            objects={
                f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[]) for i in range(25)
            },
            clipper=ClipperConfig(
                recipe=RecipeConfig(input_resources={"ore_red": 5}),
                clip_rate=0.1,
                auto_length_scale=False,
                length_scale=manual_value,
            ),
        )

        # Verify manual value is preserved
        assert game_config.clipper.length_scale == manual_value

    def test_auto_length_scale_different_grid_sizes(self):
        """Test that auto length scale adapts to different grid sizes."""
        num_buildings = 25

        for width, height in [(25, 25), (50, 50), (100, 100), (50, 100)]:
            game_config = GameConfig(
                map_builder=RandomMapBuilder.Config(agents=10, width=width, height=height),
                objects={
                    f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[])
                    for i in range(num_buildings)
                },
                clipper=ClipperConfig(recipe=RecipeConfig(), clip_rate=0.1),
            )

            grid_size = max(width, height)
            expected = (grid_size / np.sqrt(num_buildings)) * np.sqrt(4.51 / (4 * np.pi))
            assert np.isclose(game_config.clipper.length_scale, expected, rtol=1e-5)

    def test_auto_length_scale_different_building_counts(self):
        """Test that auto length scale adapts to different building densities."""
        width, height = 50, 50

        for num_buildings in [10, 25, 50, 100]:
            game_config = GameConfig(
                map_builder=RandomMapBuilder.Config(agents=10, width=width, height=height),
                objects={
                    f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[])
                    for i in range(num_buildings)
                },
                clipper=ClipperConfig(recipe=RecipeConfig(), clip_rate=0.1),
            )

            grid_size = max(width, height)
            expected = (grid_size / np.sqrt(num_buildings)) * np.sqrt(4.51 / (4 * np.pi))
            assert np.isclose(game_config.clipper.length_scale, expected, rtol=1e-5)

    def test_auto_length_scale_no_buildings(self):
        """Test that auto length scale gracefully handles zero buildings."""
        game_config = GameConfig(
            map_builder=RandomMapBuilder.Config(agents=10, width=50, height=50),
            objects={},  # No buildings
            clipper=ClipperConfig(recipe=RecipeConfig(), clip_rate=0.1),
        )

        # Should fall back to default value
        assert game_config.clipper.length_scale == 1.0

    def test_auto_length_scale_no_clipper(self):
        """Test that no clipper doesn't cause errors."""
        game_config = GameConfig(
            map_builder=RandomMapBuilder.Config(agents=10, width=50, height=50),
            objects={
                f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[]) for i in range(25)
            },
            clipper=None,
        )

        # Should not raise any errors
        assert game_config.clipper is None

    def test_percolation_formula_correctness(self):
        """Verify the percolation formula produces expected values."""
        # Test case: 50x50 grid with 25 buildings
        game_config = GameConfig(
            map_builder=RandomMapBuilder.Config(agents=10, width=50, height=50),
            objects={
                f"assembler_{i}": AssemblerConfig(name=f"assembler_{i}", type_id=i + 1, recipes=[]) for i in range(25)
            },
            clipper=ClipperConfig(recipe=RecipeConfig(), clip_rate=0.1, length_scale_factor=1.0),
        )

        # Manual calculation: (50 / sqrt(25)) * sqrt(4.51 / (4 * pi))
        # = 10 * sqrt(4.51 / 12.566370614...)
        # = 10 * sqrt(0.35887...)
        # = 10 * 0.599058...
        # ≈ 5.991
        assert np.isclose(game_config.clipper.length_scale, 5.991, atol=0.001)
