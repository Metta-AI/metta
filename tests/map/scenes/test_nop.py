from metta.map.scenes.nop import Nop
from tests.map.scenes.utils import render_node


def test_basic():
    node = render_node(Nop, {}, (3, 3))

    assert (node.grid == "empty").sum() == 9
