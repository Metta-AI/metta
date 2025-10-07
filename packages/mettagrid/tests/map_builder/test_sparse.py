import textwrap
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.map_builder.sparse import SparseMapBuilder


def test_sparse_builder_builds_grid():
    config = SparseMapBuilder.Config(
        width=3,
        height=2,
        objects=[
            {"row": 0, "column": 1, "name": "wall"},
            {"row": 1, "column": 2, "name": "agent.agent"},
        ],
    )

    builder = SparseMapBuilder(config)
    game_map = builder.build()

    assert game_map.grid.shape == (2, 3)
    assert game_map.grid[0, 1] == "wall"
    assert game_map.grid[1, 2] == "agent.agent"
    assert game_map.grid[0, 0] == "empty"


def test_map_builder_config_from_uri_detects_sparse_builder():
    yaml_content = textwrap.dedent(
        """
        type: mettagrid.map_builder.sparse.SparseMapBuilder
        width: 2
        height: 2
        objects:
          - row: 0
            column: 0
            name: wall
        """
    )

    with NamedTemporaryFile(mode="w", suffix=".map", delete=False, encoding="utf-8") as tmp:
        tmp.write(yaml_content)
        path = Path(tmp.name)

    try:
        config = MapBuilderConfig.from_uri(path)
        assert isinstance(config, SparseMapBuilder.Config)

        builder = config.create()
        assert isinstance(builder, SparseMapBuilder)

        game_map = builder.build()
        expected = np.array([["wall", "empty"], ["empty", "empty"]], dtype=game_map.grid.dtype)
        assert np.array_equal(game_map.grid, expected)
    finally:
        path.unlink(missing_ok=True)
