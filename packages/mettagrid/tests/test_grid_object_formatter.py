import warnings

import mettagrid.util.grid_object_formatter as grid_object_formatter


def test_format_grid_object_base_falls_back_to_anchor_cell() -> None:
    previous_count = grid_object_formatter._malformed_cells_count
    grid_object_formatter._malformed_cells_count = 0

    grid_object = {
        "id": 1,
        "type_id": 2,
        "location": (4, 5, 6),
        "cells": [("invalid", "cells", "data")],
    }

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        formatted = grid_object_formatter.format_grid_object_base(grid_object)

    try:
        assert formatted["cells"] == [(4, 5, 6)]
        assert len(caught_warnings) >= 1
        message = str(caught_warnings[0].message)
        assert "anchor" in message.lower()
    finally:
        grid_object_formatter._malformed_cells_count = previous_count
