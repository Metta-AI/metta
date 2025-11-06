import typing

import mettagrid
import mettagrid.mapgen.utils.ascii_grid

ShowMode = typing.Literal["ascii", "ascii_border"]


def show_game_map(game_map: mettagrid.GameMap, mode: ShowMode):
    if mode == "ascii":
        ascii_lines = mettagrid.mapgen.utils.ascii_grid.grid_to_lines(game_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = mettagrid.mapgen.utils.ascii_grid.grid_to_lines(game_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
