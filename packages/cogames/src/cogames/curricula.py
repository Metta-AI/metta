"""Curriculum helpers for cycling through CoGames maps."""

import collections
import typing

import mettagrid.config.mettagrid_config


def make_rotation(
    missions: list[tuple[str, mettagrid.config.mettagrid_config.MettaGridConfig]],
) -> typing.Callable[[], mettagrid.config.mettagrid_config.MettaGridConfig]:
    if not missions:
        raise ValueError("Must have at least one mission in rotation")
    rotation = collections.deque(missions)

    def supplier() -> mettagrid.config.mettagrid_config.MettaGridConfig:
        _, cfg = rotation[0]
        rotation.rotate(-1)
        return cfg

    return supplier
