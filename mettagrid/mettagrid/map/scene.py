from typing import Any, List, Optional, TypedDict, Union, cast

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf

from mettagrid.config.utils import scenes_root, simple_instantiate

from .node import Node

SceneCfg = Union["Scene", DictConfig, str]


class TypedChild(TypedDict):
    scene: SceneCfg
    where: Optional[Any]
    # TODO - more props; use dataclasses instead, or structured configs?


def make_scene(cfg: SceneCfg) -> "Scene":
    if isinstance(cfg, str):
        if cfg.startswith("/"):
            cfg = cfg[1:]
        cfg = cast(SceneCfg, OmegaConf.load(f"{scenes_root}/{cfg}"))

    if isinstance(cfg, Scene):
        # already an instance, maybe recursive=True was enabled
        return cfg
    elif isinstance(cfg, DictConfig):
        # hydra-style dict with `_target_` key
        return simple_instantiate(cfg)
    else:
        raise ValueError(f"Invalid scene config: {cfg}")


# Base class for all map scenes.
class Scene:
    def __init__(self, children: Optional[List[TypedChild]] = None):
        self._children = children or []
        pass

    def make_node(self, grid: npt.NDArray[np.str_]):
        return Node(self, grid)

    # Render does two things:
    # - updates `node.grid` as it sees fit
    # - creates areas of interest in a node through `node.make_area()`
    def _render(self, node: Node) -> None:
        raise NotImplementedError("Subclass must implement render method")

    # Subclasses can override this to provide a list of children based on the node.
    # By default, children are static, which makes them configurable in the config file, but they can't depend
    # on the node.
    def get_children(self, node) -> List[TypedChild]:
        return self._children

    def render(self, node: Node):
        self._render(node)

        for query in self.get_children(node):
            sweep = query.get("sweep")
            subqueries: list[TypedChild] = [query]
            if sweep:
                subqueries = [
                    OmegaConf.merge(entry, query)  # type: ignore
                    for entry in sweep
                ]

            for query in subqueries:
                areas = node.select_areas(query)
                for area in areas:
                    scene = make_scene(query["scene"])

                    child_node = scene.make_node(area.grid)
                    child_node.render()
