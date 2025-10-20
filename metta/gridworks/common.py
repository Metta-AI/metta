from typing_extensions import TypedDict

from mettagrid.base_config import Config


class ErrorResult(TypedDict):
    error: str


def dump_config_with_implicit_info(config: Config | list[Config] | dict[str, Config]) -> dict:
    fields_unset: list[str] = []

    def traverse(obj: Config, prefix: str = ""):
        def with_prefix(field: str) -> str:
            return f"{prefix}.{field}" if prefix else field

        model_fields = set(type(obj).model_fields.keys())
        model_fields_unset = model_fields - obj.model_fields_set
        for field in model_fields_unset:
            fields_unset.append(with_prefix(field))

        for field in obj.model_fields_set:
            value = getattr(obj, field)
            if isinstance(value, Config):
                traverse(value, with_prefix(field))

    if isinstance(config, list):
        for item in config:
            traverse(item)
    elif isinstance(config, dict):
        for key, value in config.items():
            traverse(value, key)
    else:
        traverse(config)

    return {
        "value": config,
        "unset_fields": fields_unset,
    }
