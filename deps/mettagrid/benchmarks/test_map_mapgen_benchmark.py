import pytest
from hydra import compose, initialize

from mettagrid.map.mapgen import MapGen

# Define test parameters for different scenes
SCENE_PARAMS = [
    "/scenes/wfc/blob",
    "/scenes/wfc/simple",
    "/scenes/convchain/blob",
    "/scenes/convchain/c_shape",
    "/scenes/convchain/diagonal",
]


# Fixture to initialize Hydra before tests
@pytest.fixture(scope="module")
def hydra_setup():
    # Initialize Hydra with the correct relative path
    with initialize(version_base=None, config_path="../configs"):
        # Load the default config
        cfg = compose(config_name="test_basic")
        yield cfg


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
