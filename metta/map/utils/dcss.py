import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from multiprocessing import Pool

from omegaconf import OmegaConf

from metta.map.config import scenes_root
from metta.map.utils.make_scene_config import (
    make_convchain_config_from_pattern,
    make_wfc_config_from_pattern,
)
from metta.util.metta_script import hydraless_metta_script

# This script extracts maps from Dungeon Crawl Stone Soup into individual yaml scene files.

logger = logging.getLogger(__name__)


def fetch_simple():
    url = "https://raw.githubusercontent.com/crawl/crawl/master/crawl-ref/source/dat/des/arrival/simple.des"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def process_map_source(ascii_source):
    # split into lines
    lines = ascii_source.split("\n")
    # add space padding so that all lines are the same length
    max_length = max(len(line) for line in lines)
    lines = [line.ljust(max_length) for line in lines]

    # replace all symbols that are not `x` with spaces; replace `x` with `#`
    for i in range(len(lines)):
        original_line = lines[i]
        new_line = "".join(["#" if char == "x" else "." for char in original_line])
        lines[i] = new_line

    return "\n".join(lines)


def is_trivial(ascii_map):
    # if everything is blank, return true
    if all(line == " " * len(line) for line in ascii_map.split("\n")):
        return True

    return False


@dataclass
class DCSSMap:
    name: str
    pattern: str


def get_maps() -> list[DCSSMap]:
    simple = fetch_simple()

    # Split by 'NAME:' but keep the delimiter at the beginning of the subsequent parts using a lookahead assertion.
    # If the string starts with 'NAME:', the first part will be an empty string.
    parts = re.split(r"(?=NAME:)", simple)

    maps: list[DCSSMap] = []
    for part in parts:
        if "NAME:" not in part:
            continue  # preamble before the first map

        name = part.split("NAME:")[1].split("\n")[0].strip()

        # Cut the part between "MAP" and "ENDMAP"
        start_marker = "MAP\n"
        end_marker = "\nENDMAP"
        start = part.find(start_marker)
        end = part.find(end_marker)
        if start != -1 and end != -1:
            ascii_source = part[start + len(start_marker) : end]
            ascii_map = process_map_source(ascii_source)
            if is_trivial(ascii_map):
                continue
            maps.append(DCSSMap(name=name, pattern=ascii_map))

    return maps


dir = scenes_root / "dcss"
wfc_dir = dir / "wfc"


def process_map_entry(map_entry: DCSSMap):
    name = map_entry.name
    pattern = map_entry.pattern
    logger.info(f"Processing map: {name}")

    # convchain
    convchain_config = make_convchain_config_from_pattern(pattern)
    convchain_dir = dir / "convchain"
    convchain_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(convchain_config, convchain_dir / f"{name}.yaml")

    # wfc
    wfc_config = make_wfc_config_from_pattern(map_entry.pattern)
    if wfc_config is None:
        logger.warning(f"Invalid pattern for WFC: {map_entry.name}")
        return

    wfc_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(wfc_config, wfc_dir / f"{name}.yaml")


def generate_scenes_from_dcss_maps():
    maps = get_maps()

    cpus = os.cpu_count() or 1

    with Pool(cpus) as pool:
        pool.map(process_map_entry, maps)


hydraless_metta_script(generate_scenes_from_dcss_maps)
