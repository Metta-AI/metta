from typing import Literal

from mettagrid import GameMap
from mettagrid.mapgen.utils.ascii_grid import grid_to_lines

ShowMode = Literal["ascii", "ascii_border"]


def show_game_map(game_map: GameMap, mode: ShowMode):
    if mode == "ascii":
        ascii_lines = grid_to_lines(game_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = grid_to_lines(game_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
