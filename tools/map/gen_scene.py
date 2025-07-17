#!/usr/bin/env -S uv run
import argparse
import logging
import os
import signal
from typing import cast, get_args

from omegaconf import DictConfig, OmegaConf

from metta.common.util.resolvers import register_resolvers
from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap

# Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_map(cfg_path: str, width: int, height: int, overrides: DictConfig | None = None):
    register_resolvers()

    cfg: DictConfig = cast(DictConfig, OmegaConf.merge(OmegaConf.load(cfg_path), overrides))

    if not OmegaConf.is_dict(cfg):
        raise ValueError(f"Invalid config type: {type(cfg)}")

    cfg = cast(DictConfig, cfg)

    mapgen_cfg = OmegaConf.create(
        {
            "_target_": "metta.map.mapgen.MapGen",
            "width": width,
            "height": height,
            "root": cfg,
        }
    )

    return StorableMap.from_cfg(mapgen_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-mode", choices=get_args(ShowMode), default="mettascope", help="Show the map in the specified mode"
    )
    parser.add_argument("--overrides", type=str, default="", help="OmegaConf overrides for the scene config")
    parser.add_argument("cfg_path", type=str, help="Path to the scene config file")
    parser.add_argument("width", type=int, help="Width of the map")
    parser.add_argument("height", type=int, help="Height of the map")
    args = parser.parse_args()

    show_mode = args.show_mode
    cfg_path = args.cfg_path
    overrides = args.overrides

    overrides_cfg = OmegaConf.from_cli([override for override in overrides.split(" ") if override])

    storable_map = make_map(cfg_path, args.width, args.height, overrides_cfg)

    show_map(storable_map, show_mode)


if __name__ == "__main__":
    main()
