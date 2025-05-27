import numpy as np

from metta.map.utils.storable_map import ascii_to_grid


def test_load_unicode_map():
    lines = [
        "🧱🧱🧱",
        "⛩A ",
        "🧱🧱🧱",
    ]
    grid = ascii_to_grid(lines)
    assert grid.shape == (3, 3)
    assert np.array_equal(
        grid,
        np.array(
            [
                ["wall", "wall", "wall"],
                ["altar", "agent.agent", "empty"],
                ["wall", "wall", "wall"],
            ],
            dtype="<U50",
        ),
    )
