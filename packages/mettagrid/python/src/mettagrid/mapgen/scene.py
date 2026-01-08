from __future__ import annotations

import inspect
import time
from enum import StrEnum, auto
from typing import Any, ClassVar, Final, Generic, Self, TypeVar, get_args, get_origin

import numpy as np
from pydantic import (
    ModelWrapValidatorHandler,
    SerializeAsAny,
    field_serializer,
    model_serializer,
    model_validator,
)

from mettagrid.base_config import Config
from mettagrid.map_builder import MapGrid
from mettagrid.mapgen.area import Area, AreaQuery
from mettagrid.util.module import load_symbol


class GridTransform(StrEnum):
    IDENTITY = auto()
    ROT_90 = auto()
    ROT_180 = auto()
    ROT_270 = auto()
    FLIP_H = auto()
    FLIP_V = auto()
    TRANSPOSE = auto()
    TRANSPOSE_ALT = auto()

    @property
    def transpose(self) -> bool:
        return TRANSFORM_FLAGS[self][0]

    @property
    def flip_v(self) -> bool:
        return TRANSFORM_FLAGS[self][1]

    @property
    def flip_h(self) -> bool:
        return TRANSFORM_FLAGS[self][2]

    def inverse(self) -> GridTransform:
        if self == GridTransform.ROT_90:
            return GridTransform.ROT_270
        elif self == GridTransform.ROT_270:
            return GridTransform.ROT_90
        else:
            return self

    def apply(self, grid: MapGrid) -> MapGrid:
        """
        Apply this transformation to a numpy array.

        All transformations are views, so editing them will edit the original grid.
        """
        if self == GridTransform.IDENTITY:
            return grid
        result = grid.T if self.transpose else grid
        if self.flip_v:
            result = np.flip(result, axis=0)
        if self.flip_h:
            result = np.flip(result, axis=1)
        return result

    def apply_to_coords(self, grid: MapGrid, x: int, y: int) -> tuple[int, int]:
        """
        Apply this transformation to a coordinate.
        """
        H, W = grid.shape

        if self.transpose:
            x, y = y, x
            H, W = W, H  # Shape changes after transpose

        if self.flip_v:
            y = H - 1 - y
        if self.flip_h:
            x = W - 1 - x

        return x, y

    def compose(self, other: GridTransform):
        """Return the transform equivalent to applying self then other."""
        # Use a canonical test grid to determine composition
        test_grid = np.array([[0, 1], [2, 3]])
        composed_result = other.apply(self.apply(test_grid))

        # Find which single transform produces the same result
        for transform in GridTransform:
            if np.array_equal(transform.apply(test_grid), composed_result):
                return transform

        raise RuntimeError("Composition not found")  # Should never happen


TRANSFORM_FLAGS: Final[dict[GridTransform, tuple[bool, bool, bool]]] = {
    GridTransform.IDENTITY: (False, False, False),
    GridTransform.ROT_90: (True, False, True),
    GridTransform.ROT_180: (False, True, True),
    GridTransform.ROT_270: (True, True, False),
    GridTransform.FLIP_H: (False, False, True),
    GridTransform.FLIP_V: (False, True, False),
    GridTransform.TRANSPOSE: (True, False, False),
    GridTransform.TRANSPOSE_ALT: (True, True, True),
}


class SceneConfig(Config):
    # will be defined by Scene.__init_subclass__ when the config is bound to a scene class
    _scene_cls: ClassVar[type[Scene]] | None = None
    children: list[ChildrenAction] = []
    seed: int | None = None

    # Transform relative to the area that this scene config receives in `create`.
    transform: GridTransform = GridTransform.IDENTITY

    @field_serializer("transform")
    def _ser_transform(self, value: GridTransform):
        # Emit as the enum value (str) to avoid UnexpectedValue warnings during dumps
        return value.value

    def model_dump(self, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump_json(**kwargs)

    @property
    def scene_cls(self) -> type[Scene]:
        if not self._scene_cls:
            raise ValueError(f"{self.__class__.__name__} is not bound to a scene class")
        return self._scene_cls

    @classmethod
    def _type_str(cls) -> str:
        if not cls._scene_cls:
            raise ValueError(f"{cls.__class__.__name__} is not bound to a scene class")
        return f"{cls._scene_cls.__module__}.{cls._scene_cls.__name__}.Config"

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)
        return {"type": self._type_str(), **data}

    @model_validator(mode="wrap")
    @classmethod
    def _validate_with_type(cls, v: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        # This code is copy-pasted from MapBuilderConfig. Refer to that file for a better commented version.
        # (Soon it will be refactored to use PolymorphicConfig.)
        if isinstance(v, SceneConfig):
            if not isinstance(v, cls):
                raise TypeError(f"Expected {cls.__qualname__} subclass, got {type(v).__qualname__}")
            return v

        if not isinstance(v, dict):
            raise ValueError("SceneConfig params must be a dict")

        t = v.get("type")
        if t is None:
            return handler(v)

        type_cls = load_symbol(t) if isinstance(t, str) else t

        if not inspect.isclass(type_cls):
            raise TypeError("'type' must point to a class")

        if not issubclass(type_cls, cls):
            raise TypeError(f"'type' {t} is not a subclass of {cls._type_str()}")

        data = {k: v for k, v in v.items() if k != "type"}
        result = type_cls.model_validate(data)

        assert isinstance(result, cls)

        return result

    def create_root(
        self,
        area: Area,
        rng: np.random.Generator | None = None,
        instance_id: int | None = None,
        use_instance_id_for_team_assignment: bool = False,
    ) -> Scene:
        effective_instance_id = instance_id if use_instance_id_for_team_assignment else None
        return self.scene_cls(
            area=area,
            config=self,
            rng=rng or np.random.default_rng(),
            instance_id=effective_instance_id,
            use_instance_id_for_team_assignment=use_instance_id_for_team_assignment,
        )

    def create_as_child(
        self,
        parent_scene: Scene,
        area: Area,
        instance_id: int | None = None,
        use_instance_id_for_team_assignment: bool = False,
    ) -> Scene:
        rng = parent_scene.rng.spawn(1)[0]
        inherited_instance_id = instance_id if instance_id is not None else getattr(parent_scene, "instance_id", None)
        effective_instance_id = inherited_instance_id if use_instance_id_for_team_assignment else None

        return self.scene_cls(
            area=area,
            config=self,
            rng=rng,
            parent_scene=parent_scene,
            instance_id=effective_instance_id,
            use_instance_id_for_team_assignment=use_instance_id_for_team_assignment,
        )


AnySceneConfig = SerializeAsAny[SceneConfig]


class ChildrenAction(AreaQuery):
    scene: AnySceneConfig
    instance_id: int | None = None
    use_instance_id_for_team_assignment: bool | None = None


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

    # Full transform relative to the root grid.
    # This can be different from `self.config.transform`, which is a local transform relative to the parent scene.
    transform: GridTransform

    # { "lock_name": [area_id1, area_id2, ...] }
    _locks: dict[str, set[int]]

    # Will be set by Scene.render_with_children()
    _render_start_time: float = 0
    _render_end_time: float = 0
    _render_with_children_end_time: float = 0

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
        config: ConfigT,
        parent_scene: Scene | None = None,
        instance_id: int | None = None,
        use_instance_id_for_team_assignment: bool = False,
    ):
        # Validate config - they can come from untyped yaml or from weakly typed dicts in python code.
        self.config = self.Config.model_validate(config)

        self.children = []

        self.area = area
        self.parent_scene = parent_scene
        self.transform = (
            parent_scene.transform.compose(self.config.transform) if parent_scene else self.config.transform
        )

        self.use_instance_id_for_team_assignment = use_instance_id_for_team_assignment

        if self.use_instance_id_for_team_assignment:
            if instance_id is not None:
                self.instance_id = instance_id
            elif parent_scene is not None:
                self.instance_id = getattr(parent_scene, "instance_id", None)
            else:
                self.instance_id = None
        else:
            self.instance_id = None

        self._update_shortcuts()

        self._areas = []
        self._locks = {}

        self.rng = np.random.default_rng(self.config.seed or rng)

        self.post_init()

    def _update_shortcuts(self):
        # shortcuts for common properties

        grid = self.area.grid
        # Render on the inversed transformed grid, so the end result looks like the correct transformation from the
        # point of view of the original grid
        grid = self.transform.inverse().apply(grid)

        self.grid = grid
        self.height = grid.shape[0]
        self.width = grid.shape[1]

    def post_init(self):
        """
        Override this method in subclasses to perform additional initialization.

        This is preferred over `__init__` because it's harder to make `__init__` type-safe in scene subclasses.
        """
        pass

    def get_children(self) -> list[ChildrenAction]:
        """
        Subclasses can override this method to provide a list of dynamically generated children actions.

        The list of static children actions from scene config will always be appended to the list returned by this
        method.

        Examples:
        1) `RandomScene` picks a random scene and proxies rendering to it with `where="full"`.
        2) `Mirror` scene renders its child scene on a half of the grid and then mirrors it.
        3) `Auto` scene encapsulates the complex scene tree through a simple top-level config.
        """

        return []

    def get_scene_tree(self) -> dict:
        return {
            "config": self.config.model_dump(),
            "area": self.area.as_dict(),
            "children": [child.get_scene_tree() for child in self.children],
            "render_start_time": self._render_start_time,
            "render_end_time": self._render_end_time,
            "render_with_children_end_time": self._render_with_children_end_time,
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
        # First, render the scene itself.
        self._render_start_time = time.time()
        self.render()
        self._render_end_time = time.time()

        # Then, render the children scenes based on the children actions.
        children_actions = self.get_children()
        children_actions.extend(self.config.children)
        for action in children_actions:
            areas = self.select_areas(action)
            for area in areas:
                use_instance_id_for_team_assignment = (
                    action.use_instance_id_for_team_assignment
                    if action.use_instance_id_for_team_assignment is not None
                    else getattr(self, "use_instance_id_for_team_assignment", False)
                )
                child_scene = action.scene.create_as_child(
                    self,
                    area,
                    instance_id=action.instance_id,
                    use_instance_id_for_team_assignment=use_instance_id_for_team_assignment,
                )
                self.children.append(child_scene)
                child_scene.render_with_children()

        self._render_with_children_end_time = time.time()

    def make_area(self, x: int, y: int, width: int, height: int, tags: list[str] | None = None) -> Area:
        inverse_transform = self.transform.inverse()
        # Transform both corners, then find the bounds of the area in untransformed coordinates.
        (orig_x1, orig_y1) = inverse_transform.apply_to_coords(self.grid, x, y)
        (orig_x2, orig_y2) = inverse_transform.apply_to_coords(self.grid, x + width - 1, y + height - 1)
        if orig_x1 > orig_x2:
            orig_x1, orig_x2 = orig_x2, orig_x1
        if orig_y1 > orig_y2:
            orig_y1, orig_y2 = orig_y2, orig_y1
        orig_width = orig_x2 - orig_x1 + 1
        orig_height = orig_y2 - orig_y1 + 1

        area = self.area.make_subarea(
            x=orig_x1,
            y=orig_y1,
            width=orig_width,
            height=orig_height,
            tags=tags,
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
            self.area.transplant_to_grid(grid, shift_x, shift_y, copy_grid=True)

        # Scene's area could be modified by previous levels of recursion.
        self._update_shortcuts()

        # transplant all sub-areas
        for sub_area in self._areas:
            sub_area.transplant_to_grid(self.grid, shift_x, shift_y, copy_grid=False)

        # recurse into children scenes
        for child_scene in self.children:
            child_scene.transplant_to_grid(grid, shift_x, shift_y, is_root=False)


SceneConfig.model_rebuild()
