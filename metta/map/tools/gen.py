#!/usr/bin/env -S uv run
import logging
import random
import string

from metta.common.config.tool import Tool
from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.utils.module import load_function

logger = logging.getLogger(__name__)


# Based on heuristics, see https://github.com/Metta-AI/mettagrid/pull/108#discussion_r2054699842
def uri_is_file(uri: str) -> bool:
    last_part = uri.split("/")[-1]
    return "." in last_part and len(last_part.split(".")[-1]) <= 4


class GenTool(Tool):
    env_fn: str  # Path to the function that makes EnvConfig

    output_uri: str | None = None  # Output URI
    show_mode: ShowMode | None = None  # Show the map in the specified mode
    count: int = 1  # Number of maps to generate
    env_overrides: list[str] = []  # OmegaConf-style overrides for the env config

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        show_mode = self.show_mode
        if not show_mode and not self.output_uri:
            # if not asked to save, show the map
            show_mode = "mettascope"

        output_uri = self.output_uri
        count = self.count
        env_fn_name = self.env_fn

        env_fn = load_function(env_fn_name)

        # TODO - support env_fn args?
        env_config = env_fn()

        if not isinstance(env_config, EnvConfig):
            raise ValueError(f"Env config must be an instance of EnvConfig, got {type(env_config)}")

        for override in self.env_overrides:
            key, value = override.split("=")
            env_config = env_config.override(key, value)

        logger.info(f"Env config:\n{env_config.model_dump_json(indent=2)}")

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
            storable_map = StorableMap.from_cfg(env_config.game.map_builder)

            # Save the map if requested
            target_uri = make_output_uri()
            if target_uri:
                storable_map.save(target_uri)

            # Show the map if requested
            show_map(storable_map, show_mode)
