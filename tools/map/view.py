#!/usr/bin/env -S uv run
import argparse
import logging
from typing import get_args

from metta.map.load_random import get_random_map_uri
from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap
from metta.util.metta_script import hydraless_metta_script
from tools.map.gen import uri_is_file

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-mode", choices=get_args(ShowMode), help="Show the map in the specified mode", default="mettascope"
    )
    parser.add_argument("uri", type=str, help="URI of the map to view")
    args = parser.parse_args()

    uri = args.uri

    if not uri_is_file(uri):
        # probably a directory
        while uri.endswith("/"):
            uri = uri[:-1]
        logger.info(f"Loading random map from directory {uri}")
        uri = get_random_map_uri(uri)

    logger.info(f"Loading map from {uri}")
    storable_map = StorableMap.from_uri(uri)

    show_map(storable_map, args.show_mode)


hydraless_metta_script(main)
