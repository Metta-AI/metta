import logging
from typing import Annotated

import typer

from mettagrid.mapgen.utils.show import ShowMode, show_map
from mettagrid.mapgen.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


def main(
    uri: Annotated[str, typer.Argument(help="Path or URI to the map file")],
    mode: Annotated[ShowMode, typer.Option(help="Show mode: ascii, ascii_border, or none")] = "ascii_border",
):
    """
    View a map from a file or URI using the specified display mode.
    """
    logger.info(f"Loading map from {uri}")
    storable_map = StorableMap.from_uri(uri)
    show_map(storable_map, mode)


if __name__ == "__main__":
    typer.run(main)
