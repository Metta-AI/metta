import pytest

from mettagrid.map.mapgen import MapGen

# Define test parameters for different scenes
SCENE_PARAMS = [
    "wfc/blob.yaml",
    "wfc/simple.yaml",
    "convchain/blob.yaml",
    "convchain/c_shape.yaml",
    "convchain/diagonal.yaml",
]


# Parametrized test that will run for each scene
@pytest.mark.parametrize("scene", SCENE_PARAMS)
def test_scene_gen(benchmark, scene):
    """Benchmark scene generation for different scene types."""
    size = 20

    # This is where the actual benchmarking happens
    result = benchmark.pedantic(
        lambda: MapGen(size, size, root=scene).build(),
        iterations=3,  # Number of iterations within each round
        rounds=1,  # Number of rounds to perform
        warmup_rounds=0,  # No warmup
    )

    # Optional validation
    assert result is not None, f"Failed to generate scene {scene}"
