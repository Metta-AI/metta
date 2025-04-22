from __future__ import annotations

import logging
import os
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import boto3
import hydra
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
from omegaconf import DictConfig, OmegaConf

D = TypeVar("D")


def dictconfig_to_dataclass(
    cls: Type[D],
    config: DictConfig | dict,
    *,
    allow_extra_keys: bool = False,
) -> D:
    """
    Convert a Hydra DictConfig (or plain dict) into a fully‑typed dataclass
    `cls`, applying any preprocessing hooks encountered during the build,
    then validate it.

    Args:
        cls: The dataclass type to convert to
        config: A DictConfig or dict to convert
        allow_extra_keys: If True, ignore extra fields in the config not present in the dataclass

    Returns:
        An instance of the dataclass with properly converted types
    """
    # Handle both parameter names

    # Convert DictConfig to dict if needed
    data = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    data = data or {}

    # Build and validate the dataclass
    obj = _build_dataclass(cls, data, cls.__name__, allow_extra=allow_extra_keys)
    validate_dataclass(obj, cls)
    return obj


def propagate_cfg(
    parent: Mapping[str, object],
    children: MutableMapping[str, Mapping[str, object]] | list[Mapping[str, object]],
    cls: Type,
) -> None:
    """
    Copy every field that belongs to `cls` from `parent` into each element
    of *children* **unless the child already set a non‑None value**.

    Pre‑condition: this is called on raw (OmegaConf‑compatible) dicts,
    *before* they're turned into dataclass instances.
    """
    # the fields we're allowed to propagate
    shared = {f.name for f in fields(cls)}

    # normalize children into an iterable of mutable mappings
    if isinstance(children, Mapping):  # Dict[str, child]
        it = children.values()
    else:  # List[child]
        it = children

    for child in it:
        for key in shared:
            if key not in parent:  # nothing to propagate
                continue
            # child already set an explicit value → leave it alone
            if key in child and child[key] is not None:
                continue
            # otherwise inject the parent's value
            child[key] = parent[key]


# ---------------------------------------------------------------------------
# Optional handling
# ---------------------------------------------------------------------------


def _unwrap_optional(ftype: Any) -> tuple[bool, Any]:
    """Extract the inner type from Optional[T]."""
    origin = get_origin(ftype)
    if origin is Union:
        args = tuple(a for a in get_args(ftype) if a is not type(None))  # noqa: E721
        if len(args) == 1:
            return True, args[0]
    return False, ftype


# ---------------------------------------------------------------------------
# Conversion helpers – build‑time
# ---------------------------------------------------------------------------


def _convert_node(value: Any, ftype: Any, path: str, *, allow_extra: bool) -> Any:  # noqa: C901, PLR0911
    """Recursively convert *value* to comply with *ftype*.

    Strictly enforces annotations; raises ``TypeError`` / ``ValueError`` on the
    first mismatch so configuration errors fail fast.
    """
    if value is None or ftype is None or ftype is Any:
        return value

    # Optional[T]
    is_opt, inner = _unwrap_optional(ftype)
    if is_opt:
        return None if value is None else _convert_node(value, inner, path, allow_extra=allow_extra)
    ftype = inner

    origin = get_origin(ftype)
    args = get_args(ftype)

    # --------------------------- Generic containers ------------------------
    if origin in (list, List) and args:
        if not isinstance(value, list):
            raise TypeError(f"{path}: expected list, got {type(value).__name__}")
        elem_t = args[0]
        return [_convert_node(v, elem_t, f"{path}[{i}]", allow_extra=allow_extra) for i, v in enumerate(value)]

    if origin in (tuple, Tuple) and args:
        if not isinstance(value, tuple):
            raise TypeError(f"{path}: expected tuple, got {type(value).__name__}")
        # Homogeneous Tuple[T, ...]
        if len(args) == 2 and args[1] is ...:
            elem_t = args[0]
            return tuple(_convert_node(v, elem_t, f"{path}[{i}]", allow_extra=allow_extra) for i, v in enumerate(value))
        # Fixed‑length Tuple[T1, T2, ...]
        if len(args) != len(value):
            raise TypeError(f"{path}: expected tuple of length {len(args)}, got {len(value)}")
        return tuple(
            _convert_node(v, t, f"{path}[{i}]", allow_extra=allow_extra)
            for i, (v, t) in enumerate(zip(value, args, strict=False))
        )

    if origin in (set, frozenset) and args:
        cls = set if origin is set else frozenset
        if not isinstance(value, cls):
            raise TypeError(f"{path}: expected {cls.__name__}, got {type(value).__name__}")
        elem_t = args[0]
        return cls(_convert_node(v, elem_t, f"{path}[{i}]", allow_extra=allow_extra) for i, v in enumerate(value))

    if origin in (dict, Dict, Mapping, MutableMapping) and len(args) == 2:
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected dict, got {type(value).__name__}")
        k_t, v_t = args
        out: dict[Any, Any] = {}
        for k, v in value.items():
            if k_t is not Any and not isinstance(k, k_t):
                raise TypeError(f"{path}: key {k!r} expected {k_t.__name__}, got {type(k).__name__}")
            out[k] = _convert_node(v, v_t, f"{path}[{k!r}]", allow_extra=allow_extra)
        return out

    # --------------------------- Special cases -----------------------------
    if is_dataclass(ftype) and isinstance(value, dict):
        return _build_dataclass(ftype, value, path, allow_extra=allow_extra)

    if origin is Callable or ftype is Callable:
        if not callable(value):
            raise TypeError(f"{path}: expected callable, got {type(value).__name__}")
        return value

    if isinstance(ftype, type) and issubclass(ftype, Enum):
        if not isinstance(value, ftype):
            raise TypeError(f"{path}: expected {ftype.__name__}, got {type(value).__name__}")
        return value

    if isinstance(ftype, type):
        if not isinstance(value, ftype):
            raise TypeError(f"{path}: expected {ftype.__name__}, got {type(value).__name__}")
        return value

    raise TypeError(f"{path}: unsupported annotation {ftype!r}")


# ---------------------------------------------------------------------------
# Dataclass builder
# ---------------------------------------------------------------------------


def _build_dataclass(cls: Type[D], data: Dict[str, Any], path: str, *, allow_extra: bool) -> D:
    """Build a dataclass instance from a dictionary."""
    logger = logging.getLogger(__name__)
    if not is_dataclass(cls):
        raise TypeError(f"{path}: {cls.__name__} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected dict, got {type(data).__name__}")

    # ------------------------------------------------------------------ #
    # 0.  Pre‑processing hook (opt‑in)                                    #
    # ------------------------------------------------------------------ #
    preprocess = getattr(cls, "__dictconfig_pre__", None) or getattr(cls, "__preprocess_dictconfig__", None)
    if callable(preprocess):
        # work on a shallow copy to avoid side‑effects outside builder
        data = preprocess(dict(data)) or data

    # ------------------------------------------------------------------ #
    # 1.  Extra keys check
    # NOTE: This is done by the dataclass constructor, but this allows us
    # to print the right field keys.                               #
    # ------------------------------------------------------------------ #
    field_names = {f.name for f in fields(cls)}
    extra = set(data) - field_names
    if extra:
        if not allow_extra:
            raise ValueError(f"{path}: unexpected fields {sorted(extra)}")
        logger.debug("%s: ignoring extra fields %s", path, sorted(extra))

    # ------------------------------------------------------------------ #
    # 2.  Check for missing required fields
    # NOTE: This is done by the dataclass constructor, but this allows us
    # to print the right field keys.
    # ------------------------------------------------------------------ #
    required_fields = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
    missing = required_fields - set(data.keys())
    if missing:
        raise TypeError(
            f"{path} is missing required fields: {sorted(missing)}. "
            f"Required fields are: {sorted(required_fields)}. "
            f"Provided fields were: {sorted(data.keys())}"
        )

    # ------------------------------------------------------------------ #
    # 3.  Build field values                                             #
    # ------------------------------------------------------------------ #
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _convert_node(
                data[f.name], hints.get(f.name, Any), f"{path}.{f.name}", allow_extra=allow_extra
            )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Runtime validation helpers
# ---------------------------------------------------------------------------


def validate_dataclass(instance: Any, cls: Type[D], path: str | None = None) -> None:
    """
    Recursively validate that `instance` is a proper `cls` dataclass,
    raising TypeError/ValueError with full nested paths when something is wrong.
    """
    if path is None:
        path = cls.__name__

    if not is_dataclass(cls):
        raise TypeError(f"{path}: {cls.__name__} is not a dataclass")

    if not isinstance(instance, cls):
        raise TypeError(f"{path}: got {type(instance).__name__}, expected {cls.__name__}")

    hints = get_type_hints(cls)
    for f in fields(cls):
        if not hasattr(instance, f.name):
            raise AttributeError(f"{path}: missing attribute {f.name!r}")

        fpath = f"{path}.{f.name}"
        value = getattr(instance, f.name)

        # Missing required field (shouldn't happen due to dataclass constructor checks)
        if value is MISSING:
            raise ValueError(f"{fpath}: required field is missing")

        # Skip None values
        if value is None:
            continue

        ftype = hints.get(f.name, Any)
        _validate_value(value, ftype, fpath)


def _validate_value(value: Any, ftype: Any, path: str) -> None:
    """Recursively validate that `value` matches the expected `ftype`."""
    if ftype is Any or value is None:
        return

    is_opt, inner = _unwrap_optional(ftype)
    if is_opt:
        if value is None:
            return
        ftype = inner

    origin = get_origin(ftype)
    args = get_args(ftype)

    # --------------------------- Generic containers ------------------------
    if origin in (list, List) and args:
        if not isinstance(value, list):
            raise TypeError(f"{path}: expected list, got {type(value).__name__}")
        elem_t = args[0]
        for i, item in enumerate(value):
            _validate_value(item, elem_t, f"{path}[{i}]")
        return

    if origin in (tuple, Tuple) and args:
        if not isinstance(value, tuple):
            raise TypeError(f"{path}: expected tuple, got {type(value).__name__}")
        # Homogeneous Tuple[T, ...]
        if len(args) == 2 and args[1] is ...:
            elem_t = args[0]
            for i, item in enumerate(value):
                _validate_value(item, elem_t, f"{path}[{i}]")
        # Fixed‑length Tuple[T1, T2, ...]
        else:
            if len(args) != len(value):
                raise TypeError(f"{path}: expected tuple of length {len(args)}, got {len(value)}")
            for i, (item, elem_t) in enumerate(zip(value, args, strict=False)):
                _validate_value(item, elem_t, f"{path}[{i}]")
        return

    if origin in (set, frozenset) and args:
        cls = set if origin is set else frozenset
        if not isinstance(value, cls):
            raise TypeError(f"{path}: expected {cls.__name__}, got {type(value).__name__}")
        elem_t = args[0]
        for i, item in enumerate(value):
            _validate_value(item, elem_t, f"{path}[{i}]")
        return

    if origin in (dict, Dict, Mapping, MutableMapping) and len(args) == 2:
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected dict, got {type(value).__name__}")
        k_t, v_t = args
        for k, v in value.items():
            if k_t is not Any and not isinstance(k, k_t):
                raise TypeError(f"{path}: key {k!r} expected {k_t.__name__}, got {type(k).__name__}")
            _validate_value(v, v_t, f"{path}[{k!r}]")
        return

    # --------------------------- Special cases -----------------------------
    if is_dataclass(ftype):
        validate_dataclass(value, ftype, path)
        return

    if origin is Callable or ftype is Callable:
        if not callable(value):
            raise TypeError(f"{path}: expected callable, got {type(value).__name__}")
        return

    if isinstance(ftype, type) and issubclass(ftype, Enum):
        if not isinstance(value, ftype):
            raise TypeError(f"{path}: expected {ftype.__name__}, got {type(value).__name__}")
        return

    if isinstance(ftype, type):
        if not isinstance(value, ftype):
            raise TypeError(f"{path}: expected {ftype.__name__}, got {type(value).__name__}")
        return

    raise TypeError(f"{path}: unsupported annotation {ftype!r}")


def config_from_path(config_path: str, overrides: DictConfig = None) -> DictConfig:
    if config_path is None:
        raise ValueError("Config path cannot be None")

    env_cfg = hydra.compose(config_name=config_path)
    if config_path.startswith("/"):
        config_path = config_path[1:]
    for p in config_path.split("/")[:-1]:
        env_cfg = env_cfg[p]
    if overrides not in [None, {}]:
        if env_cfg._target_ == "mettagrid.mettagrid_env.MettaGridEnvSet":
            raise NotImplementedError("Cannot parse overrides when using multienv_mettagrid")
        env_cfg = OmegaConf.merge(env_cfg, overrides)
    return env_cfg


def check_aws_credentials() -> bool:
    """Check if valid AWS credentials are available from any source."""
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        # This check is primarily for github actions.
        return True
    try:
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return True
    except (NoCredentialsError, ClientError):
        return False


def check_wandb_credentials() -> bool:
    """Check if valid W&B credentials are available."""
    if "WANDB_API_KEY" in os.environ:
        # This check is primarily for github actions.
        return True
    try:
        return wandb.login(anonymous="never", timeout=10)
    except Exception:
        return False


def setup_metta_environment(cfg: DictConfig, require_aws: bool = True, require_wandb: bool = True):
    if require_aws:
        # Check that AWS is good to go.
        if not check_aws_credentials():
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("python ./devops/aws/setup_sso.py")
            print("Alternatively, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            exit(1)
    if cfg.wandb.track and require_wandb:
        # Check that W&B is good to go.
        if not check_wandb_credentials():
            print("W&B is not configured, please install:")
            print("pip install wandb")
            print("and run:")
            print("wandb login")
            print("Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured.")
            exit(1)
