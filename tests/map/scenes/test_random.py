import numpy as np

from metta.map.scenes.random import Random
from tests.map.scenes.utils import render_scene


def make_grid(height: int, width: int) -> np.ndarray:
    return np.full((height, width), "empty", dtype="<U50")


def test_objects():
    scene = render_scene(Random, {"objects": {"altar": 3, "temple": 2}}, (3, 3))

    assert (scene.grid == "altar").sum() == 3
    assert (scene.grid == "temple").sum() == 2


def test_agents():
    scene = render_scene(Random, {"agents": 2}, (3, 3))

    assert (scene.grid == "agent.agent").sum() == 2


def test_agents_dict():
    scene = render_scene(Random, {"agents": {"prey": 2, "predator": 1}}, (3, 3))

    assert (scene.grid == "agent.prey").sum() == 2
    assert (scene.grid == "agent.predator").sum() == 1
