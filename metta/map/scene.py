import importlib
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, get_args, get_origin

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import Config
from metta.map.config import scenes_root
from metta.map.types import AreaQuery, ChildrenAction, MapGrid, SceneCfg
from metta.map.utils.random import MaybeSeed


@dataclass
class Area:
    id: int  # unique for areas in a scene; not unique across scenes.
    grid: MapGrid
    tags: list[str]


ParamsT = TypeVar("ParamsT", bound=Config)


class Scene(Generic[ParamsT]):
    """
    Base class for all map scenes.

    Subclasses must:
    1. Inherit from Scene[ParamsT], where ParamsT is a subclass of Config.
    2. Define a `render()` method.

    If you need to perform additional initialization, override `post_init()` instead of `__init__`.
    """

    Params: type[ParamsT]
    params: ParamsT

    _areas: list[Area]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # walk the subclass’s __orig_bases__ looking for Scene[…]
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Scene:
                # pull out the single type argument
                cls.Params = get_args(base)[0]
                return

        raise TypeError(f"{cls.__name__} must inherit from Scene[…], with a concrete Params class parameter")

    def __init__(
        self,
        grid: MapGrid,
        params: ParamsT | DictConfig | dict | None = None,
        children: Optional[List[ChildrenAction]] = None,
        seed: MaybeSeed = None,
    ):
        # Validate params - they can come from untyped yaml or from weakly typed dicts in python code.
        if params is None:
            params = {}
        if isinstance(params, DictConfig):
            self.params = self.Params(params)
        elif isinstance(params, dict):
            self.params = self.Params(**params)
        elif isinstance(params, self.Params):
            self.params = params
        else:
            raise ValueError(f"Invalid params: {params}")

        children = children or []
        self.children: list[ChildrenAction] = []
        for action in children:
            if not isinstance(action, ChildrenAction):
                action = ChildrenAction(**action)
            self.children.append(action)

        self.grid = grid
        self.height = grid.shape[0]
        self.width = grid.shape[1]

        self._areas = []

        # { "lock_name": [area_id1, area_id2, ...] }
        self._locks = {}
        self._full_area = Area(
            id=-1,
            grid=self.grid,
            tags=[],
        )

        self.rng = np.random.default_rng(seed)

        self.post_init()

    def post_init(self):
        """
        Override this method in subclasses to perform additional initialization.

        This is preferred over `__init__` because it's harder to make `__init__` type-safe in scene subclasses.
        """
        pass

    # Render implementations can do two things:
    # - update `self.grid` as it sees fit
    # - create areas of interest in a scene through `self.make_area()`
    def render(self):
        raise NotImplementedError("Subclass must implement render method")

    # Subclasses can override this to provide a list of children.
    # By default, children are static, which makes them configurable in the config file, but then can't depend
    # on the specific generated content.
    def get_children(self) -> List[ChildrenAction]:
        return self.children

    def render_with_children(self):
        self.render()
        for query in self.get_children():
            areas = self.select_areas(query)
            for area in areas:
                child_scene = make_scene(query.scene, area.grid)
                child_scene.render_with_children()

    def make_area(self, x: int, y: int, width: int, height: int, tags: Optional[List[str]] = None) -> Area:
        area = Area(
            id=len(self._areas),
            grid=self.grid[y : y + height, x : x + width],
            tags=tags or [],
        )
        self._areas.append(area)
        return area

    def select_areas(self, query: AreaQuery) -> list[Area]:
        areas = self._areas

        selected_areas: list[Area] = []

        where = query.where
        if where:
            if where == "full":
                selected_areas = [self._full_area]
            else:
                tags = where.tags
                for area in areas:
                    match = True
                    for tag in tags:
                        if tag not in area.tags:
                            match = False
                            break
                    if match:
                        selected_areas.append(area)
        else:
            selected_areas = areas

        # Filter out locked areas.
        lock = query.lock
        if lock:
            if lock not in self._locks:
                self._locks[lock] = set()

            # Remove areas that are locked.
            selected_areas = [area for area in selected_areas if area.id not in self._locks[lock]]

        limit = query.limit
        if limit is not None and limit < len(selected_areas):
            order_by = query.order_by
            offset = query.offset
            if order_by == "random":
                assert offset is None, "offset is not supported for random order"
                rng = np.random.default_rng(query.order_by_seed)
                selected_areas = list(rng.choice(selected_areas, size=int(limit), replace=False))  # type: ignore
            elif order_by == "first":
                offset = offset or 0
                selected_areas = selected_areas[offset : offset + limit]
            elif order_by == "last":
                if not offset:
                    selected_areas = selected_areas[-limit:]
                else:
                    selected_areas = selected_areas[-limit - offset : -offset]
            else:
                raise ValueError(f"Invalid order_by value: {order_by}")

        if lock:
            # Add final list of used areas to the lock.
            self._locks[lock].update([area.id for area in selected_areas])

        return selected_areas


def load_class(full_class_name: str) -> type[Scene]:
    module_name, class_name = full_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, Scene):
        raise ValueError(f"Class {cls} does not inherit from Scene")
    return cls


def make_scene(cfg: SceneCfg, grid: MapGrid) -> Scene:
    if callable(cfg):
        # useful for dynamically produced scenes in `get_children()`
        scene = cfg(grid)
        if not isinstance(scene, Scene):
            raise ValueError(f"Scene callback didn't return a valid scene: {scene}")
        return scene

    if isinstance(cfg, str):
        if cfg.startswith("/"):
            cfg = cfg[1:]
        cfg = OmegaConf.to_container(OmegaConf.load(f"{scenes_root}/{cfg}"))  # type: ignore

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid scene config: {cfg}, type: {type(cfg)}")

    cls = load_class(cfg["type"])
    return cls(grid=grid, params=cfg.get("params", {}), children=cfg.get("children", []))
