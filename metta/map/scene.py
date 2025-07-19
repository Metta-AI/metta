import importlib
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar, get_args, get_origin

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import Config
from metta.map.config import scenes_root
from metta.map.random.int import MaybeSeed
from metta.map.types import Area, AreaQuery, ChildrenAction, SceneCfg

ParamsT = TypeVar("ParamsT", bound=Config)

SceneT = TypeVar("SceneT", bound="Scene")


# This class is useful for debugging: we store every scene we produce dynamically.
@dataclass
class ChildInfo:
    area: Area
    scene: "Scene"


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

    @classmethod
    def validate_params(cls, params: ParamsT | DictConfig | dict | None) -> ParamsT:
        if params is None:
            return cls.Params({})
        if isinstance(params, DictConfig):
            return cls.Params(params)
        elif isinstance(params, dict):
            return cls.Params(**params)
        elif isinstance(params, cls.Params):
            return params
        else:
            raise ValueError(f"Invalid params: {params}")

    def __init__(
        self,
        area: Area,
        params: ParamsT | DictConfig | dict | None = None,
        children: Optional[list[ChildrenAction]] = None,
        seed: MaybeSeed = None,
    ):
        # Validate params - they can come from untyped yaml or from weakly typed dicts in python code.
        self.params = self.validate_params(params)

        # `children` are not scenes, but queries that will be used to select areas and produce scenes in them.
        children = children or []
        self.children: list[ChildrenAction] = []
        for action in children:
            if not isinstance(action, ChildrenAction):
                action = ChildrenAction(**action)
            self.children.append(action)

        self.child_infos: list[ChildInfo] = []

        self.area = area
        self.grid = area.grid
        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]

        self._areas = []

        # { "lock_name": [area_id1, area_id2, ...] }
        self._locks = {}

        self.rng = np.random.default_rng(seed)

        self.post_init()

    def post_init(self):
        """
        Override this method in subclasses to perform additional initialization.

        This is preferred over `__init__` because it's harder to make `__init__` type-safe in scene subclasses.
        """
        pass

    def register_child(self, area: Area, child_scene: "Scene"):
        self.child_infos.append(ChildInfo(area=area, scene=child_scene))

    # Render implementations can do two things:
    # - update `self.grid` as it sees fit
    # - create areas of interest in a scene through `self.make_area()`
    def render(self):
        raise NotImplementedError("Subclass must implement render method")

    # Subclasses can override this to provide a list of children.
    # By default, children are static, which makes them configurable in the config file, but then can't depend
    # on the specific generated content.
    def get_children(self) -> list[ChildrenAction]:
        return self.children

    def get_scene_tree(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "params": self.params.model_dump(),
            "area": self.area.as_dict(),
            "children": [child.scene.get_scene_tree() for child in self.child_infos],
        }

    def print_scene_tree(self, indent=0):
        print(" " * indent + self.__class__.__name__)
        print(" " * indent + f"area: {self.area.as_dict()}")
        print(" " * indent + f"params: {self.params.model_dump()}")
        for child in self.child_infos:
            child.scene.print_scene_tree(indent + 2)

    def render_with_children(self):
        self.render()

        for query in self.get_children():
            areas = self.select_areas(query)
            for area in areas:
                child_rng = self.rng.spawn(1)[0]
                child_scene = make_scene(cfg=query.scene, area=area, rng=child_rng)
                self.register_child(area, child_scene)
                child_scene.render_with_children()

    def make_area(self, x: int, y: int, width: int, height: int, tags: Optional[list[str]] = None) -> Area:
        area = Area(
            x=x + self.area.x,
            y=y + self.area.y,
            width=width,
            height=height,
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
                selected_areas = [self.area]
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
            selected_areas = [area for area in selected_areas if id(area) not in self._locks[lock]]

        limit = query.limit
        if limit is not None and limit < len(selected_areas):
            order_by = query.order_by
            offset = query.offset
            if order_by == "random":
                assert offset is None, "offset is not supported for random order"
                selected_areas = list(self.rng.choice(selected_areas, size=int(limit), replace=False))  # type: ignore
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
            self._locks[lock].update([id(area) for area in selected_areas])

        return selected_areas

    @classmethod
    def factory(
        cls: Type[SceneT],
        params: dict | Config,
        children: Optional[list[ChildrenAction]] = None,
        seed: MaybeSeed = None,
    ) -> SceneCfg:
        return lambda area, rng: cls(area=area, params=params, seed=seed or rng, children=children)

    @classmethod
    def intrinsic_size(cls, params: ParamsT) -> tuple[int, int] | None:
        """
        Some scenes have a fixed size, which can be used to compute the size of
        the map.

        For example, an ascii map has a fixed size, which can be used to compute
        the size of the map.

        This is a class method, because an instantiated scene is already bound
        to an area, and we can't change it.

        Because of this limitation, the utility of intrinsically sized scenes is
        limited. The main way of sizing scenes is the top-down algorithm, where
        the size of the child scenes is computed based on the size of the parent
        scene, and the size of the top-level scene is determined by MapGen
        params.

        The returned pair is (height, width), same order as in numpy arrays.
        """

        return None

    def get_labels(self) -> list[str]:
        # default: use the scene class name as a label
        return [self.__class__.__name__]


def load_class(full_class_name: str) -> type[Scene]:
    module_name, class_name = full_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, Scene):
        raise ValueError(f"Class {cls} does not inherit from Scene")
    return cls


def scene_cfg_to_dict(cfg: SceneCfg) -> dict:
    if callable(cfg):
        raise ValueError("Callable scene configs are not supported")
    if isinstance(cfg, str):
        if cfg.startswith("/"):
            cfg = cfg[1:]
        cfg = OmegaConf.load(f"{scenes_root}/{cfg}")  # type: ignore

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg)  # type: ignore

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid scene config: {cfg}, type: {type(cfg)}")

    return cfg


def make_scene(cfg: SceneCfg, area: Area, rng: np.random.Generator) -> Scene:
    if callable(cfg):
        # Some scene configs are lambdas, usually produced by `Scene.factory()` helper.
        # These are often useful for dynamically produced children actions in `get_children()`.
        scene = cfg(area, rng)
        if not isinstance(scene, Scene):
            raise ValueError(f"Scene callback didn't return a valid scene: {scene}")
        return scene

    dict_cfg = scene_cfg_to_dict(cfg)

    cls = load_class(dict_cfg["type"])
    return cls(
        area=area,
        params=dict_cfg.get("params", {}),
        children=dict_cfg.get("children", []),
        seed=dict_cfg.get("seed", rng),
    )
