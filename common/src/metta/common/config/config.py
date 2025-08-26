from __future__ import annotations

from typing import Any, NoReturn, Self, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, TypeAdapter


class Config(BaseModel):
    """
    Common extension of Pydantic's BaseModel that:
    - sets `extra="forbid"` by default
    - adds `override` and `update` methods for overriding values based on `path.to.value` keys
    """

    model_config = ConfigDict(extra="forbid")

    def _auto_initialize_field(
        self, parent_obj: "Config", field_name: str, traversed_path: list[str], fail
    ) -> "Config | None":
        """Auto-initialize a None Config field if possible."""

        field = type(parent_obj).model_fields.get(field_name)
        if field is None:
            return None

        field_type = self._unwrap_optional(field.annotation)

        if isinstance(field_type, type) and issubclass(field_type, Config):
            try:
                new_instance = field_type()
                setattr(parent_obj, field_name, new_instance)
                return new_instance
            except TypeError:
                return None

        return None

    def _unwrap_optional(self, field_type):
        """Unwrap Optional[T] → T if applicable, else return original type."""
        if get_origin(field_type) is Union:
            non_none = [arg for arg in get_args(field_type) if arg is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return field_type

    def override(self, key: str, value: Any) -> Self:
        """Override a value in the config."""
        key_path = key.split(".")

        def fail(error: str) -> NoReturn:
            raise ValueError(
                f"Override failed. Full config:\n {self.model_dump_json(indent=2)}\nOverride {key} failed: {error}"
            )

        inner_cfg: Config | dict[str, Any] = self
        traversed_path: list[str] = []
        for key_part in key_path[:-1]:
            if isinstance(inner_cfg, dict):
                if key_part not in inner_cfg:
                    fail(f"key {key} not found")
                inner_cfg = inner_cfg[key_part]
                traversed_path.append(key_part)
                continue

            if not hasattr(inner_cfg, key_part):
                failed_path = ".".join(traversed_path + [key_part])
                fail(f"key {failed_path} not found")

            next_inner_cfg = getattr(inner_cfg, key_part)
            if next_inner_cfg is None:
                # If the field is None, try to auto-initialize it
                next_inner_cfg = self._auto_initialize_field(inner_cfg, key_part, traversed_path, fail)
                if next_inner_cfg is None:
                    failed_path = ".".join(traversed_path + [key_part])
                    fail(f"Cannot auto-initialize None field {failed_path}")

            if not isinstance(next_inner_cfg, Config) and not isinstance(next_inner_cfg, dict):
                failed_path = ".".join(traversed_path + [key_part])
                fail(f"key {failed_path} is not a Config object")

            inner_cfg = next_inner_cfg
            traversed_path.append(key_part)

        if isinstance(inner_cfg, dict):
            if key_path[-1] not in inner_cfg:
                fail(f"key {key} not found")
        elif isinstance(inner_cfg, Config):
            if not hasattr(inner_cfg, key_path[-1]):
                fail(f"key {key} not found")

        if isinstance(inner_cfg, dict):
            inner_cfg[key_path[-1]] = value
            return self

        cls = type(inner_cfg)
        field = cls.model_fields.get(key_path[-1])
        if field is None:
            fail(f"key {key} is not a valid field")

        value = TypeAdapter(field.annotation).validate_python(value)
        setattr(inner_cfg, key_path[-1], value)

        return self

    def update(self, updates: dict[str, Any]) -> Self:
        """Applies multiple overrides to the config."""
        for key, value in updates.items():
            self.override(key, value)
        return self
