import numpy as np
import numpy.testing as npt

from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.scenes.uniform_extractors import UniformExtractorParams, UniformExtractorScene
from mettagrid.mapgen.types import Area


def _make_scene(params: UniformExtractorParams) -> UniformExtractorScene:
    grid = create_grid(9, 9)
    area = Area.root_area_from_grid(grid)
    return UniformExtractorScene(area=area, params=params, seed=123)


def test_default_weights_distribution() -> None:
    scene = _make_scene(UniformExtractorParams())

    names, probabilities = scene._resolve_extractor_distribution()

    expected_names = [
        "carbon_extractor",
        "oxygen_extractor",
        "germanium_extractor",
        "silicon_extractor",
        "charger",
    ]
    expected_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.3], dtype=float)
    expected_probabilities = expected_weights / expected_weights.sum()

    assert names == expected_names
    npt.assert_allclose(probabilities, expected_probabilities)


def test_custom_weights_override_defaults() -> None:
    params = UniformExtractorParams(
        extractor_names=["charger", "oxygen_extractor", "silicon_extractor"],
        extractor_weights={"charger": 1.0, "oxygen_extractor": 2.0},
    )
    scene = _make_scene(params)

    names, probabilities = scene._resolve_extractor_distribution()

    assert names == ["charger", "oxygen_extractor"]
    npt.assert_allclose(probabilities, np.array([1.0 / 3.0, 2.0 / 3.0], dtype=float))
