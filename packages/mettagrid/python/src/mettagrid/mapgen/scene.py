from __future__ import annotations

from typing import Any, Generic, Optional, Type, TypeVar, get_args, get_origin

import numpy as np
from pydantic import FieldSerializationInfo, ValidationInfo, field_serializer, field_validator

from mettagrid.config.config import Config
from mettagrid.mapgen.random.int import MaybeSeed
from mettagrid.mapgen.types import Area, AreaQuery, MapGrid
from mettagrid.util.module import load_symbol

ParamsT = TypeVar("ParamsT", bound=Config)

SceneT = TypeVar("SceneT", bound="Scene")


def _ensure_scene_cls(v: Any) -> type[Scene]:
    if isinstance(v, str):
        v = load_symbol(v)
    if not issubclass(v, Scene):
        raise ValueError(f"Class {v} does not inherit from Scene")
    return v


class SceneConfig(Config):
    type: type[Scene]
    params: Config
    children: list[ChildrenAction] | None = None
    seed: int | None = None

    # Turn strings into classes, ensure subclass of Scene
    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> type[Scene]:
        return _ensure_scene_cls(v)

    @field_serializer("type")
    def _serialize_type(self, type: type[Scene], _info):
        return f"{type.__module__}.{type.__name__}"

    # Validate/convert params using the already-validated 'type'
    @field_validator("params", mode="before")
    @classmethod
    def _validate_params(cls, v: Any, info: ValidationInfo) -> Any:
        scene_cls = info.data.get("type")
        if scene_cls is None:
            # Shouldn't happen because "type" is defined before "params"
            raise TypeError("'type' must be provided before 'params'")
        scene_cls = _ensure_scene_cls(scene_cls)
        return scene_cls.validate_params(v)

    @field_serializer("params")
    def _serialize_params(self, params: Config, _info: FieldSerializationInfo):
        return params.model_dump(
            exclude_unset=_info.exclude_unset,
            exclude_defaults=_info.exclude_defaults,
            # TODO - pass more? can we pass all flags?
        )

    def create(self, area: Area, rng: np.random.Generator) -> Scene:
        return self.type(area=area, params=self.params, seed=self.seed or rng, children_actions=self.children)


class ChildrenAction(AreaQuery):
    scene: SceneConfig
    # Deterministic seeding: optional additive offset applied to parent's seed
    seed_offset: int = 0


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
    children_actions: list[ChildrenAction]
    children: list[Scene]

    # { "lock_name": [area_id1, area_id2, ...] }
    _locks: dict[str, set[int]]

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
    def validate_params(cls, params: Any) -> ParamsT:
        return cls.Params.model_validate(params or {})

    def __init__(
        self,
        area: Area,
        params: ParamsT | None = None,
        children_actions: Optional[list[ChildrenAction]] = None,
        seed: MaybeSeed = None,
    ):
        # Validate params - they can come from untyped yaml or from weakly typed dicts in python code.
        self.params = self.validate_params(params)

        children_actions = children_actions or []
        self.children_actions = []
        for action in children_actions:
            if not isinstance(action, ChildrenAction):
                action = ChildrenAction(**action)
            self.children_actions.append(action)

        self.children = []

        self.area = area

        # shortcuts for common properties
        self.grid = area.grid
        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]

        self._areas = []
        self._locks = {}

        # Ensure we always have an integer seed so children can derive deterministic seeds
        if isinstance(seed, (int, np.integer)):
            self.seed_int = int(seed)
            self.rng = np.random.default_rng(self.seed_int)
        else:
            # Fall back (should not happen if SceneConfig.create passes an int)
            self.rng = np.random.default_rng(seed)
            # derive a reproducible int from the generator for child seeding
            self.seed_int = int(self.rng.integers(0, 2**63 - 1, dtype=np.int64))

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
        return self.children_actions

    def get_scene_tree(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "params": self.params.model_dump(),
            "area": self.area.as_dict(),
            "children": [child.get_scene_tree() for child in self.children],
        }

    def print_scene_tree(self, indent=0):
        print(" " * indent + self.__class__.__name__)
        print(" " * indent + f"area: {self.area.as_dict()}")
        print(" " * indent + f"params: {self.params.model_dump()}")
        for child in self.children:
            child.print_scene_tree(indent + 2)

    # Render implementations can do two things:
    # - update `self.grid` as it sees fit
    # - create areas of interest in a scene through `self.make_area()`
    def render(self):
        raise NotImplementedError("Subclass must implement render method")

    def render_with_children(self):
        self.render()

        for action_idx, action in enumerate(self.get_children()):
            areas = self.select_areas(action)
            for area_idx, area in enumerate(areas):
                # Deterministic, additive child seed from parent's seed
                child_seed = (
                    self.seed_int
                    + 1000003 * int(action_idx)
                    + 7919 * int(area_idx)
                    + int(getattr(action, "seed_offset", 0))
                ) & ((1 << 63) - 1)
                child_scene = action.scene.create(area, child_seed)
                self.children.append(child_scene)
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
        params: ParamsT | None = None,
        children_actions: Optional[list[ChildrenAction]] = None,
        seed: int | None = None,
    ) -> SceneConfig:
        return SceneConfig(
            type=cls,
            params=params or cls.Params(),
            children=children_actions,
            seed=seed,
        )

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
