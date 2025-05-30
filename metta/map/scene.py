from typing import Any, Callable, Literal

from metta.map.types import MapGrid
from metta.util.config import Config

SceneCfg = Callable[[MapGrid], Any] | dict | str


class AreaWhere(Config):
    tags: list[str] = []


class AreaQuery(Config):
    limit: int | None = None
    offset: int | None = None
    lock: str | None = None
    where: Literal["full"] | AreaWhere | None = None
    order_by: Literal["random", "first", "last"] = "random"
    order_by_seed: int | None = None


class ChildrenAction(AreaQuery):
    scene: SceneCfg
