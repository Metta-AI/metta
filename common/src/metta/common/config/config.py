from __future__ import annotations

from typing import Any, NoReturn, Self, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter


class Config(BaseModel):
    """
    Common extension of Pydantic's BaseModel that:
    - sets `extra="forbid"` by default
    - adds `override` and `update` methods for overriding values based on `path.to.value` keys
    """

    model_config = ConfigDict(extra="forbid")

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
                # If the field is None, try to initialize it with a default Config
                cls = type(inner_cfg)
                field = cls.model_fields.get(key_part)
                if field is None:
                    failed_path = ".".join(traversed_path + [key_part])
                    fail(f"key {failed_path} not found")

                # Get the actual type (handle Optional[T] -> T)
                field_type = field.annotation
                if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                    # For Optional[T], find the non-None type
                    field_type = next((arg for arg in field_type.__args__ if arg is not type(None)), None)
                    if field_type is None:
                        failed_path = ".".join(traversed_path + [key_part])
                        fail(f"Unable to determine non-None field type for {failed_path}")

                try:
                    if field_type and issubclass(field_type, Config):
                        try:
                            next_inner_cfg = field_type()
                            setattr(inner_cfg, key_part, next_inner_cfg)
                        except Exception as e:
                            failed_path = ".".join(traversed_path + [key_part])
                            fail(f"Cannot initialize {failed_path}: {str(e)}")
                    else:
                        failed_path = ".".join(traversed_path + [key_part])
                        fail(f"key {failed_path} is not a Config object")
                except TypeError:
                    failed_path = ".".join(traversed_path + [key_part])
                    fail(f"key {failed_path} is not a Config object")
            elif not isinstance(next_inner_cfg, Config) and not isinstance(next_inner_cfg, dict):
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
