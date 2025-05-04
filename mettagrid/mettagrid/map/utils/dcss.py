import logging
import urllib.request

from omegaconf import OmegaConf

from mettagrid.config.utils import scenes_root
from mettagrid.map.utils.make_scene_config import (
    make_convchain_config_from_pattern,
    make_wfc_config_from_pattern,
)

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
        new_line = "".join(["#" if char == "x" else " " for char in original_line])
        lines[i] = new_line

    return "\n".join(lines)


def is_trivial(ascii_map):
    # if everything is blank, return true
    if all(line == " " * len(line) for line in ascii_map.split("\n")):
        return True

    return False


def get_maps():
    simple = fetch_simple()
    import re

    # Split by 'NAME:' but keep the delimiter at the beginning of the subsequent parts using a lookahead assertion.
    # If the string starts with 'NAME:', the first part will be an empty string.
    parts = re.split(r"(?=NAME:)", simple)

    maps = []
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
            maps.append({"name": name, "map": ascii_map})

    return maps


def generate_scenes_from_dcss_maps():
    maps = get_maps()
    dir = scenes_root / "dcss"
    for map_entry in maps:
        name = map_entry["name"]
        pattern = map_entry["map"]
        logger.info(f"Processing map: {map_entry['name']}")

        convchain_config = make_convchain_config_from_pattern(pattern)
        convchain_dir = dir / "convchain"
        convchain_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(convchain_config, convchain_dir / f"{name}.yaml")

        wfc_config = make_wfc_config_from_pattern(pattern)
        if wfc_config is None:
            logger.warning(f"Invalid pattern for WFC: {name}")
            continue

        wfc_dir = dir / "wfc"
        wfc_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(wfc_config, wfc_dir / f"{name}.yaml")


if __name__ == "__main__":
    generate_scenes_from_dcss_maps()
