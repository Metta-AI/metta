from typing import Any, Callable, Optional, TypedDict

from metta.map.types import MapGrid

SceneCfg = Callable[[MapGrid], Any] | dict | str


class TypedChild(TypedDict):
    scene: SceneCfg
    where: Optional[Any]
    # TODO - more props; use dataclasses instead, or structured configs?
