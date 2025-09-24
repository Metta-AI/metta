from typing import Literal

from mettagrid.mapgen.utils.storable_map import StorableMap, grid_to_lines

ShowMode = Literal["ascii", "ascii_border", "none"]


def show_map(storable_map: StorableMap, mode: ShowMode | None):
    if not mode or mode == "none":
        return

    if mode == "ascii":
        ascii_lines = grid_to_lines(storable_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = grid_to_lines(storable_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
