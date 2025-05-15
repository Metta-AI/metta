import argparse
import logging
import os
import random
import signal
import string
import time
from datetime import datetime
from typing import cast, get_args

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap
from metta.util.resolvers import register_resolvers

# Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_map(cfg_path: str, overrides: DictConfig | None = None):
    # since we are not using hydra here we can't rely on the callback that normally sets up our custom resolvers
    register_resolvers()

    cfg: DictConfig = cast(DictConfig, OmegaConf.merge(OmegaConf.load(cfg_path), overrides))
    if not OmegaConf.is_dict(cfg):
        raise ValueError(f"Invalid config type: {type(cfg)}")

    recursive = False
    if cfg.get("game") and cfg.game.get("map_builder"):
        # safe default for old room maps
        recursive = cfg.game.get("recursive_map_builder", True)
        # Looks like env config, unwrapping the map config from it. This won't
        # work for all configs because they could rely on Hydra-specific
        # features, but it has a decent chance of working.
        cfg = cfg.game.map_builder

    if "mapgen" in cfg.get("_target_"):
        # mapgen works better with recursive=False
        recursive = False

    # Generate and measure time taken
    start = time.time()
    map_builder = hydra.utils.instantiate(cfg, _recursive_=recursive)
    grid = map_builder.build()
    gen_time = time.time() - start
    logger.info(f"Time taken to build map: {gen_time}s")

    storable_map = StorableMap(
        grid=grid,
        metadata={
            "gen_time": gen_time,
            "timestamp": datetime.now().isoformat(),
        },
        config=cfg,
    )
    return storable_map


# Based on heuristics, see https://github.com/Metta-AI/mettagrid/pull/108#discussion_r2054699842
def uri_is_file(uri: str) -> bool:
    last_part = uri.split("/")[-1]
    return "." in last_part and len(last_part.split(".")[-1]) <= 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-uri", type=str, help="Output URI")
    parser.add_argument("--show-mode", choices=get_args(ShowMode), help="Show the map in the specified mode")
    parser.add_argument("--count", type=int, default=1, help="Number of maps to generate")
    parser.add_argument("--overrides", type=str, default="", help="OmniConf overrides for the map config")
    parser.add_argument("cfg_path", type=str, help="Path to the map config file")
    args = parser.parse_args()

    show_mode = args.show_mode
    output_uri = args.output_uri
    count = args.count
    cfg_path = args.cfg_path
    overrides = args.overrides

    overrides_cfg = OmegaConf.from_cli([override for override in overrides.split(" ") if override])

    if count > 1 and not output_uri:
        # requested multiple maps, let's check that output_uri is a directory
        if not output_uri:
            raise ValueError("Cannot generate more than one map without providing output_uri")

    # s3 can store things at s3://.../foo////file, so we need to remove trailing slashes
    while output_uri and output_uri.endswith("/"):
        output_uri = output_uri[:-1]

    output_is_file = output_uri and uri_is_file(output_uri)

    if count > 1 and output_is_file:
        raise ValueError(f"{output_uri} looks like a file, cannot generate multiple maps in a single file")

    def make_output_uri() -> str | None:
        if not output_uri:
            return None  # the map won't be saved

        if output_is_file:
            return output_uri

        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{output_uri}/map_{random_suffix}.yaml"

    for i in range(count):
        if count > 1:
            logger.info(f"Generating map {i + 1} of {count}")

        # Generate and measure time taken
        storable_map = make_map(cfg_path, overrides_cfg)

        # Save the map if requested
        target_uri = make_output_uri()
        if target_uri:
            storable_map.save(target_uri)

        # Show the map if requested
        show_map(storable_map, show_mode)


if __name__ == "__main__":
    main()
