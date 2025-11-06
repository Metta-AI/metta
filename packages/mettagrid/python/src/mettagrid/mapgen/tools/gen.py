#!/usr/bin/env -S uv run
import logging
import typing

import typer

import mettagrid.config.mettagrid_config
import mettagrid.mapgen.utils.show
import mettagrid.util.module

logger = logging.getLogger(__name__)


def main(
    env_fn: typing.Annotated[str, typer.Argument(help="Path to the function that makes MettaGridConfig")],
    show_mode: typing.Annotated[
        typing.Literal["ascii", "ascii_border"],
        typer.Option(help="Show the map in the specified mode (ascii, ascii_border, none)"),
    ] = "ascii_border",
    env_override: typing.Annotated[
        typing.Optional[list[str]], typer.Option("--env-override", help="OmegaConf-style overrides for the env config")
    ] = None,
):
    """
    Generate a map using a MettaGridConfig-producing function.
    """
    env_fn_name = env_fn
    fn = mettagrid.util.module.load_symbol(env_fn_name)
    if not callable(fn):
        raise ValueError(f"Env {env_fn_name} is not callable")

    mg_config = fn()
    if not isinstance(mg_config, mettagrid.config.mettagrid_config.MettaGridConfig):
        raise ValueError(f"Env config must be an instance of MettaGridConfig, got {type(mg_config)}")

    if env_override:
        for override in env_override:
            key, value = override.split("=")
            mg_config = mg_config.override(key, value)

    logger.info(f"Env config:\n{mg_config.model_dump_json(indent=2)}")

    game_map = mg_config.game.map_builder.create().build()
    mettagrid.mapgen.utils.show.show_game_map(game_map, show_mode)


if __name__ == "__main__":
    typer.run(main)
