from typing import Any, Callable, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from metta.common.util.config import Config

# Shaped version, `np.ndarray[tuple[int, int], np.dtype[np.str_]]`,
# would be better, but slices from numpy arrays are not typed properly, which makes it too annoying to use.
MapGrid: TypeAlias = npt.NDArray[np.str_]

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
