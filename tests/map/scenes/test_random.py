import numpy as np

from metta.map.scenes.random import Random
from tests.map.scenes.utils import render_node


def make_grid(height: int, width: int) -> np.ndarray:
    return np.full((height, width), "empty", dtype="<U50")


def test_objects():
    node = render_node(Random, {"objects": {"mine": 3, "generator": 2}}, (3, 3))

    assert (node.grid == "mine").sum() == 3
    assert (node.grid == "generator").sum() == 2


def test_agents():
    node = render_node(Random, {"agents": 2}, (3, 3))

    assert (node.grid == "agent.agent").sum() == 2


def test_agents_dict():
    node = render_node(Random, {"agents": {"prey": 2, "predator": 1}}, (3, 3))

    assert (node.grid == "agent.prey").sum() == 2
    assert (node.grid == "agent.predator").sum() == 1
