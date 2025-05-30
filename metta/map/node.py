import importlib
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.map.config import scenes_root
from metta.map.scene import SceneCfg, TypedChild
from metta.map.types import MapGrid
from metta.map.utils.random import MaybeSeed
from metta.util.config import Config


@dataclass
class Area:
    id: int  # unique for areas in a node; not unique across nodes.
    grid: MapGrid
    tags: list[str]


ParamsT = TypeVar("ParamsT", bound=Config)


# Base class for all map nodes.
class Node(Generic[ParamsT]):
    params_type: type[ParamsT]

    _areas: list[Area]

    params: ParamsT

    def __init__(
        self,
        grid: MapGrid,
        params: ParamsT | DictConfig | dict | None = None,
        children: Optional[List[TypedChild]] = None,
        seed: MaybeSeed = None,
    ):
        if params is None:
            params = {}
        if isinstance(params, DictConfig):
            self.params = self.params_type(params)
        elif isinstance(params, dict):
            self.params = self.params_type(**params)
        elif isinstance(params, self.params_type):
            self.params = params
        else:
            raise ValueError(f"Invalid params: {params}")

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

        self.children = children or []

        self.rng = np.random.default_rng(seed)

    # Render does two things:
    # - updates `self.grid` as it sees fit
    # - creates areas of interest in a node through `self.make_area()`
    def render(self):
        raise NotImplementedError("Subclass must implement render method")

    # Subclasses can override this to provide a list of children.
    # By default, children are static, which makes them configurable in the config file, but then can't depend
    # on the specific generated content.
    def get_children(self) -> List[TypedChild]:
        return self.children

    def render_with_children(self):
        self.render()
        for query in self.get_children():
            sweep = query.get("sweep")
            subqueries: list[TypedChild] = [query]
            if sweep:
                subqueries = [
                    OmegaConf.merge(entry, query)  # type: ignore
                    for entry in sweep
                ]

            for query in subqueries:
                areas = self.select_areas(query)
                for area in areas:
                    child_node = make_node(query["scene"], area.grid)
                    child_node.render_with_children()

    def make_area(self, x: int, y: int, width: int, height: int, tags: Optional[List[str]] = None) -> Area:
        area = Area(
            id=len(self._areas),
            grid=self.grid[y : y + height, x : x + width],
            tags=tags or [],
        )
        self._areas.append(area)
        return area

    def select_areas(self, query) -> list[Area]:
        areas = self._areas

        selected_areas: list[Area] = []

        where = query.get("where")
        if where:
            if isinstance(where, str) and where == "full":
                selected_areas = [self._full_area]
            else:
                # Type check and handling
                if isinstance(where, (DictConfig, dict)) and "tags" in where:
                    tags = where.get("tags", [])
                    if isinstance(tags, list) or isinstance(tags, ListConfig):
                        for area in areas:
                            match = True
                            for tag in tags:
                                if tag not in area.tags:
                                    match = False
                                    break
                            if match:
                                selected_areas.append(area)
                    else:
                        raise ValueError(f"Invalid 'tags' format in 'where' clause: expected list, got {type(tags)}")
                else:
                    raise ValueError(f"Invalid 'where' structure: {where}")
        else:
            selected_areas = areas

        # Filter out locked areas.
        lock = query.get("lock")
        if lock:
            if lock not in self._locks:
                self._locks[lock] = set()

            # Remove areas that are locked.
            selected_areas = [area for area in selected_areas if area.id not in self._locks[lock]]

        limit = query.get("limit")
        if limit is not None and limit < len(selected_areas):
            order_by = query.get("order_by", "random")
            offset = query.get("offset")
            if order_by == "random":
                assert offset is None, "offset is not supported for random order"
                rng = np.random.default_rng(query.get("order_by_seed"))
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


def load_class(cls: str):
    module_name, class_name = cls.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_node(cfg: SceneCfg, grid: MapGrid) -> Node:
    if callable(cfg):
        # useful for dynamically produced nodes in `get_children()`
        node = cfg(grid)
        if not isinstance(node, Node):
            raise ValueError(f"Node returned by {cfg} is not a valid node: {node}")
        return node

    if isinstance(cfg, str):
        if cfg.startswith("/"):
            cfg = cfg[1:]
        cfg = OmegaConf.to_container(OmegaConf.load(f"{scenes_root}/{cfg}"))  # type: ignore

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid scene config: {cfg}")

    cls = load_class(cfg["type"])
    return cls(grid=grid, params=cfg.get("params", {}), children=cfg.get("children", []))
