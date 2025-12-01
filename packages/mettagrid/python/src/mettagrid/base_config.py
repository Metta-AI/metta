from __future__ import annotations

from typing import Any, NoReturn, Self, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, TypeAdapter


# Please don't move this class to mettagrid.config, it would cause circular import issues that are difficult to avoid.
class Config(BaseModel):
    """
    Common extension of Pydantic's BaseModel that:
    - sets `extra="forbid"` by default
    - adds `override` and `update` methods for overriding values based on `path.to.value` keys
    """

    model_config = ConfigDict(extra="forbid")

    def _auto_initialize_field(self, parent_obj: Config, field_name: str) -> Config | None:
        """Auto-initialize a None Config field if possible."""
        field = type(parent_obj).model_fields.get(field_name)
        if not field:
            return None

        field_type = self._unwrap_optional(field.annotation)
        if not (isinstance(field_type, type) and issubclass(field_type, Config)):
            return None

        try:
            new_instance = field_type()
            setattr(parent_obj, field_name, new_instance)
            return new_instance
        except (TypeError, ValueError):
            return None

    def _unwrap_optional(self, field_type):
        """Unwrap Optional[T] → T if applicable, else return original type."""
        if get_origin(field_type) is Union:
            non_none_types = [arg for arg in get_args(field_type) if arg is not type(None)]
            return non_none_types[0] if len(non_none_types) == 1 else field_type
        return field_type

    def override(self, key: str, value: Any) -> Self:
        """Override a value in the config.

        Supports dictionary keys with dots by checking if the remaining path (including dots)
        exists as a single key in the dict before splitting further.
        """
        key_path = key.split(".")

        def fail(error: str) -> NoReturn:
            raise ValueError(
                f"Override failed. Full config:\n {self.model_dump_json(indent=2)}\nOverride {key} failed: {error}"
            )

        inner_cfg: Config | dict[str, Any] = self
        traversed_path: list[str] = []
        i = 0
        while i < len(key_path) - 1:
            key_part = key_path[i]

            if isinstance(inner_cfg, dict):
                # Check if the key_part exists in the dict
                if key_part in inner_cfg:
                    inner_cfg = inner_cfg[key_part]
                    traversed_path.append(key_part)
                    i += 1
                    continue

                # If key_part doesn't exist, check if remaining path (with dots) exists as a single key
                # This handles cases like stats["carbon.gained"] where the key contains dots
                remaining_path = ".".join(key_path[i:])
                if remaining_path in inner_cfg:
                    # We found the full path as a single key - set it directly
                    inner_cfg[remaining_path] = value
                    return self

                # If we're at the last part before the final key, allow creating new dict keys
                # This allows creating new keys like stats["silicon.gained"]
                if i == len(key_path) - 2:
                    # We're about to set the final key, so create it with the remaining path
                    inner_cfg[remaining_path] = value
                    return self

                # Otherwise, this is an error (can't traverse further)
                fail(f"key {key} not found in dict at path {'.'.join(traversed_path)}")

            if not hasattr(inner_cfg, key_part):
                failed_path = ".".join(traversed_path + [key_part])
                fail(f"key {failed_path} not found")

            next_inner_cfg = getattr(inner_cfg, key_part)
            if next_inner_cfg is None:
                # Auto-initialize None Config fields
                next_inner_cfg = self._auto_initialize_field(inner_cfg, key_part)
                if next_inner_cfg is None:
                    failed_path = ".".join(traversed_path + [key_part])
                    fail(f"Cannot auto-initialize None field {failed_path}")

            if not isinstance(next_inner_cfg, (Config, dict)):
                failed_path = ".".join(traversed_path + [key_part])
                fail(f"key {failed_path} is not a Config object")

            inner_cfg = next_inner_cfg
            traversed_path.append(key_part)
            i += 1

        # We allow dicts to get new keys, but not Configs. This is because we want to allow overrides like
        # env_cfg.game.agent.rewards.inventory.ore_red = 0.1
        # without requiring that "ore_red" was already in the inventory dict. Note that allowing overrides / updates
        # to dicts like this leads to an obnoxious inconsistency in the way dicts are updated via overrides
        # (foo.bar.baz = 1) vs how they're set in Python (foo.bar["baz"] = 1).
        if isinstance(inner_cfg, Config):
            if not hasattr(inner_cfg, key_path[-1]):
                fail(f"key {key} not found")

        if isinstance(inner_cfg, dict):
            # Final key - check if it exists, or if the full remaining path (with dots) exists
            final_key = key_path[-1]
            if final_key in inner_cfg:
                inner_cfg[final_key] = value
            else:
                # Check if remaining path (including final key) exists as a single key with dots
                remaining = ".".join(key_path[i:])
                if remaining in inner_cfg:
                    inner_cfg[remaining] = value
                else:
                    # Create new key (allows new dict keys to be added)
                    inner_cfg[final_key] = value
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


__all__ = ["Config"]
