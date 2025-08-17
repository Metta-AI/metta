from __future__ import annotations

from typing import Any, NoReturn, Self, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

T = TypeVar("T")


class Config(BaseModel):
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
            if not hasattr(inner_cfg, key_part):
                failed_path = ".".join(traversed_path + [key_part])
                fail(f"key {failed_path} not found")

            next_inner_cfg = getattr(inner_cfg, key_part)
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
        """Update a value in the config."""
        for key, value in updates.items():
            self.override(key, value)
        return self
