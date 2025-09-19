#!/usr/bin/env -S uv run
import logging
import random
import string
from typing import Annotated, Optional

import typer

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.utils.show import show_map
from mettagrid.mapgen.utils.storable_map import StorableMap
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


# Based on heuristics, see https://github.com/Metta-AI/mettagrid/pull/108#discussion_r2054699842
def uri_is_file(uri: str) -> bool:
    last_part = uri.split("/")[-1]
    return "." in last_part and len(last_part.split(".")[-1]) <= 4


def main(
    env_fn: Annotated[str, typer.Argument(help="Path to the function that makes MettaGridConfig")],
    output_uri: Annotated[Optional[str], typer.Option(help="Output URI (file or directory)")] = None,
    show_mode: Annotated[
        Optional[str], typer.Option(help="Show the map in the specified mode (ascii, ascii_border, none)")
    ] = None,
    count: Annotated[int, typer.Option(help="Number of maps to generate")] = 1,
    env_override: Annotated[
        Optional[list[str]], typer.Option("--env-override", help="OmegaConf-style overrides for the env config")
    ] = None,
):
    """
    Generate one or more maps using a MettaGridConfig-producing function.
    """
    resolved_show_mode = show_mode
    if not resolved_show_mode and not output_uri:
        resolved_show_mode = "ascii_border"

    env_fn_name = env_fn
    fn = load_symbol(env_fn_name)
    if not callable(fn):
        raise ValueError(f"Env {env_fn_name} is not callable")

    mg_config = fn()
    if not isinstance(mg_config, MettaGridConfig):
        raise ValueError(f"Env config must be an instance of MettaGridConfig, got {type(mg_config)}")

    if env_override:
        for override in env_override:
            key, value = override.split("=")
            mg_config = mg_config.override(key, value)

    logger.info(f"Env config:\n{mg_config.model_dump_json(indent=2)}")

    if count > 1 and not output_uri:
        raise ValueError("Cannot generate more than one map without providing output_uri")

    # s3 can store things at s3://.../foo////file, so we need to remove trailing slashes
    out_uri = output_uri
    while out_uri and out_uri.endswith("/"):
        out_uri = out_uri[:-1]

    output_is_file = out_uri and uri_is_file(out_uri)

    if count > 1 and output_is_file:
        raise ValueError(f"{out_uri} looks like a file, cannot generate multiple maps in a single file")

    def make_output_uri() -> str | None:
        if not out_uri:
            return None  # the map won't be saved

        if output_is_file:
            return out_uri

        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{out_uri}/map_{random_suffix}.yaml"

    for i in range(count):
        if count > 1:
            logger.info(f"Generating map {i + 1} of {count}")

        storable_map = StorableMap.from_cfg(mg_config.game.map_builder)

        target_uri = make_output_uri()
        if target_uri:
            storable_map.save(target_uri)

        show_map(storable_map, resolved_show_mode)


if __name__ == "__main__":
    typer.run(main)
