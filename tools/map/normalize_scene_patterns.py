#!/usr/bin/env -S uv run
import argparse
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from metta.map.utils.ascii_grid import char_grid_to_lines
from metta.mettagrid.char_encoder import normalize_grid_char


# OmegaConf has an ugly default formatter for multiline strings, so we use a custom yaml formatter.
def str_presenter(dumper, data):
    # if the string contains any newline, dump it as a literal block
    if "\n" in data:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str",
            data,
            style="|",  # force literal style
        )
    # otherwise use default (plain) style
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="Overwrite the yaml file with the normalized version")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.file)

    # both WFC and ConvChain have a pattern under `params.pattern`
    pattern = cfg.params.pattern

    lines, _, _ = char_grid_to_lines(pattern)
    updated_lines = []
    for line in lines:
        updated_lines.append("".join(normalize_grid_char(c) for c in line))

    updated_pattern = "\n".join(updated_lines)
    cfg.params.pattern = updated_pattern

    yaml.SafeDumper.add_representer(str, str_presenter)
    new_content = yaml.dump(OmegaConf.to_container(cfg), Dumper=yaml.SafeDumper, sort_keys=False)

    if args.in_place:
        with open(args.file, "w", encoding="utf-8") as f:
            f.write(new_content)
    else:
        print(new_content)


if __name__ == "__main__":
    main()
