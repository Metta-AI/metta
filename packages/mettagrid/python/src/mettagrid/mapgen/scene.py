from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

import numpy as np
from pydantic import model_serializer

from mettagrid.config.config import Config
from mettagrid.mapgen.types import Area, AreaQuery, MapGrid
from mettagrid.util.module import load_symbol


class SceneConfig(Config):
    # will be defined by Scene.__init_subclass__ when the config is bound to a scene class
    _scene_cls: ClassVar[type[Scene]] | None = None
    children: list[ChildrenAction] = []
    seed: int | None = None

    @property
    def scene_cls(self) -> type[Scene]:
        if not self._scene_cls:
            raise ValueError(f"{self.__class__.__name__} is not bound to a scene class")
        return self._scene_cls

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)
        if not self._scene_cls:
            raise ValueError(f"{self.__class__.__name__} is not bound to a scene class")
        return {"type": f"{self._scene_cls.__module__}.{self._scene_cls.__name__}", **data}

    def create(self, area: Area, rng: np.random.Generator) -> Scene:
        return self.scene_cls(area=area, config=self, rng=rng)


def validate_any_scene_config(v: Any) -> SceneConfig:
    # See also: _validate_open_map_builder in map_builder.py
    # After Pydantic 2.12, we can simplify this by using SerializeAsAny.

    if isinstance(v, SceneConfig):
        return v

    if not isinstance(v, dict):
        raise ValueError("Scene config must be a dict")

    t = v.get("type")
    if t is None:
        raise ValueError("'type' is required")

    target = load_symbol(t) if isinstance(t, str) else t

    if isinstance(target, type) and issubclass(target, Scene):
        cfg_model = getattr(target, "Config", None)
        if not (isinstance(cfg_model, type) and issubclass(cfg_model, SceneConfig)):
            raise TypeError(f"{target.__name__} must define a nested class Config(SceneConfig).")
        data = {k: v for k, v in v.items() if k != "type"}
        return cfg_model.model_validate(data)

    raise TypeError(f"'type' must point to a Scene subclass; got {target!r}")


class ChildrenAction(AreaQuery):
    scene: SceneConfig


ConfigT = TypeVar("ConfigT", bound=SceneConfig)

SceneT = TypeVar("SceneT", bound="Scene")


class Scene(Generic[ConfigT]):
    """
    Base class for all map scenes.

    Subclasses must:
    1. Inherit from Scene[ConfigT], where ConfigT is a subclass of SceneConfig.
    2. Define a `render()` method.

    If you need to perform additional initialization, override `post_init()` instead of `__init__`.
    """

    Config: type[ConfigT]

    _areas: list[Area]
    children: list[Scene]

    # { "lock_name": [area_id1, area_id2, ...] }
    _locks: dict[str, set[int]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Look for Scene base class
        scene_bases = [base for base in getattr(cls, "__orig_bases__", ()) if get_origin(base) is Scene]
        if len(scene_bases) != 1:
            raise TypeError(f"{cls.__name__} must inherit from Scene[…], with a concrete Config class parameter")

        # Set the Config class - this allows to use Scene.Config shorthand
        Config = get_args(scene_bases[0])[0]
        if Config._scene_cls:
            raise ValueError(f"{Config.__name__} is already bound to another scene class: {Config._scene_cls.__name__}")
        Config._scene_cls = cls
        cls.Config = Config
        return

    def __init__(
        self,
        area: Area,
        rng: np.random.Generator,
        config: ConfigT | None = None,
    ):
        # Validate config - they can come from untyped yaml or from weakly typed dicts in python code.
        self.config = self.Config.model_validate(config or {})

        self.children = []

        self.area = area

        # shortcuts for common properties
        self.grid = area.grid
        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]

        self._areas = []
        self._locks = {}

        self.rng = np.random.default_rng(self.config.seed or rng)

        self.post_init()

    def post_init(self):
        """
        Override this method in subclasses to perform additional initialization.

        This is preferred over `__init__` because it's harder to make `__init__` type-safe in scene subclasses.
        """
        pass

    # Subclasses can override this to provide a list of children actions.
    # By default, children actions are static, which makes them configurable in the config file, but then can't depend
    # on the specific generated content.
    # TODO - rename to `get_children_actions()`?
    def get_children(self) -> list[ChildrenAction]:
        return self.config.children

    def get_scene_tree(self) -> dict:
        return {
            "config": self.config.model_dump(),
            "area": self.area.as_dict(),
            "children": [child.get_scene_tree() for child in self.children],
        }

    def print_scene_tree(self, indent=0):
        print(" " * indent + f"area: {self.area.as_dict()}")
        print(" " * indent + f"config: {self.config.model_dump()}")
        for child in self.children:
            child.print_scene_tree(indent + 2)

    # Render implementations can do two things:
    # - update `self.grid` as it sees fit
    # - create areas of interest in a scene through `self.make_area()`
    def render(self):
        raise NotImplementedError("Subclass must implement render method")

    def render_with_children(self):
        self.render()

        for action in self.get_children():
            areas = self.select_areas(action)
            for area in areas:
                child_rng = self.rng.spawn(1)[0]
                child_scene = action.scene.create(area, child_rng)
                self.children.append(child_scene)
                child_scene.render_with_children()

    def make_area(self, x: int, y: int, width: int, height: int, tags: list[str] | None = None) -> Area:
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
    def intrinsic_size(cls, config: ConfigT) -> tuple[int, int] | None:
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
        config.

        The returned pair is (height, width), same order as in numpy arrays.
        """

        return None

    def transplant_to_grid(self, grid: MapGrid, shift_x: int, shift_y: int, is_root: bool = True):
        """
        Transplants the scene to a new grid.

        `shift_x` and `shift_y` are the shift of the scene area relative to the previous grid.
        `grid` must point to the outer grid (the one that the area's `x` and `y`, absolute coordinates, are relative
        to).

        This method is useful for the multi-instance MapGen mode, where we sometimes render the scene on a temporary
        grid, because we don't know the size of the full multi-instance grid in advance.
        """

        # Caution: the implementation of this method is tricky, especially the relative positioning.
        # It's intended to be used by `TransplantScene` class only. If you call it from anywhere else, make sure it's
        # doing the right thing, especially when it has multiple levels of nested sub-scenes.

        if is_root:
            # This function is recursive, but we only want to copy the grid once, on top level of recursion.
            # Also, when we recurse into children, we don't need to update the scene's area, because it was already
            # updated when we transplant all `_areas`.
            original_grid = self.grid
            self.area.transplant_to_grid(grid, shift_x, shift_y)
            self.area.grid[:] = original_grid
            self.grid = self.area.grid

        # transplant all sub-areas
        for sub_area in self._areas:
            sub_area.transplant_to_grid(self.grid, shift_x, shift_y)

        # recurse into children scenes
        for child_scene in self.children:
            child_scene.transplant_to_grid(grid, shift_x, shift_y, is_root=False)


SceneConfig.model_rebuild()
