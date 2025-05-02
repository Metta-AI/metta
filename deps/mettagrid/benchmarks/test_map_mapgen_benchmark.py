import pytest

from mettagrid.map.mapgen import MapGen

# Define test parameters for different scenes
SCENE_PARAMS = [
    "/scenes/wfc/blob",
    "/scenes/wfc/simple",
    "/scenes/convchain/blob",
    "/scenes/convchain/c_shape",
    "/scenes/convchain/diagonal",
]


# Parametrized test that will run for each scene
@pytest.mark.parametrize("scene", SCENE_PARAMS)
def test_scene_gen(benchmark, hydra_setup, scene):
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
