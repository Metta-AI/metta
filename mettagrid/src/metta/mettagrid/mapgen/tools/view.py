import logging

import typer

from metta.mettagrid.mapgen.utils.show import ShowMode, show_map
from metta.mettagrid.mapgen.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


def main(
    uri: str = typer.Argument(..., help="Path or URI to the map file"),
    mode: ShowMode = typer.Option("ascii_border", help="Show mode: ascii, ascii_border, or none"),
):
    """
    View a map from a file or URI using the specified display mode.
    """
    logger.info(f"Loading map from {uri}")
    storable_map = StorableMap.from_uri(uri)
    show_map(storable_map, mode)


if __name__ == "__main__":
    typer.run(main)
