from metta.map.scenes.random import Random
from metta.map.scenes.scene_cache import SceneCache
from tests.map.scenes.utils import render_scene


def test_scene_cache(tmp_path):
    grids = []
    for _ in range(50):
        scene = render_scene(
            SceneCache,
            SceneCache.Params(
                cache_dir=str(tmp_path),
                cache_size=10,
                scene=lambda grid: Random(grid=grid, params={"agents": 10}),
            ),
            (30, 30),
        )
        grids.append(scene.grid)

    unique_grids = set(",".join(grid.flatten()) for grid in grids)

    # the chance that we'll generate less than 5 unique maps when we asked for 50 maps is negligible
    assert 5 <= len(unique_grids) <= 10
