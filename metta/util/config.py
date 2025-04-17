import logging
import os
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Type, TypeVar, get_args, get_origin, get_type_hints

import boto3
import hydra
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")
logger = logging.getLogger(__name__)


def propagate_cfg(
    parent: Mapping[str, object],
    children: MutableMapping[str, Mapping[str, object]] | list[Mapping[str, object]],
    cls: Type,
) -> None:
    """
    Copy every field that belongs to `cls` from `parent` into each element
    of *children* **unless the child already set a non‑None value**.

    Pre‑condition: this is called on raw (OmegaConf‑compatible) dicts,
    *before* they’re turned into dataclass instances.
    """
    # the fields we’re allowed to propagate
    shared = {f.name for f in fields(cls)}

    # normalise children into an iterable of mutable mappings
    if isinstance(children, Mapping):  # Dict[str, child]
        it = children.values()
    else:  # List[child]
        it = children

    for child in it:
        for key in shared:
            if key not in parent:  # nothing to propagate
                continue
            # 1️⃣ child already set an explicit value → leave it alone
            if key in child and child[key] is not None:
                continue
            # 2️⃣ otherwise inject the parent’s value
            child[key] = parent[key]


def validate_dataclass(instance: Any, cls: Type[T], path: str = None) -> None:
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
        fpath = f"{path}.{f.name}"
        value = getattr(instance, f.name)

        # Missing required field
        if value is MISSING:
            # This mostly shouldn't happen -- you can't create dataclass instances
            # without providing all required fields, so this would have to be
            # set after instantiation.
            raise ValueError(f"{fpath}: required field is missing")

        # None is always allowed
        if value is None:
            continue

        ftype = hints.get(f.name)
        origin = get_origin(ftype)
        args = get_args(ftype)

        # List[Dataclass]   --------------------------------------------------
        if origin in (list, List) and args and is_dataclass(args[0]):
            if not isinstance(value, list):
                raise TypeError(f"{fpath}: expected list[{args[0].__name__}], got {type(value).__name__}")
            for i, item in enumerate(value):
                validate_dataclass(item, args[0], f"{fpath}[{i}]")
            continue

        # Dict[str, Dataclass] ----------------------------------------------
        if origin in (dict, Dict) and len(args) == 2 and is_dataclass(args[1]):
            if not isinstance(value, dict):
                raise TypeError(f"{fpath}: expected dict[str, {args[1].__name__}], got {type(value).__name__}")
            for k, v in value.items():
                validate_dataclass(v, args[1], f"{fpath}[{k!r}]")
            continue

        # Embedded dataclass -------------------------------------------------
        if is_dataclass(ftype):
            validate_dataclass(value, ftype, fpath)
            continue

        # Primitive type -----------------------------------------------------
        if isinstance(ftype, type) and not isinstance(value, ftype):
            raise TypeError(f"{fpath}: expected {ftype.__name__}, got {type(value).__name__}")


# --------------------------------------------------------------------------- #
# Internal helpers for building dataclasses                                   #
# --------------------------------------------------------------------------- #
def _get_parent_config_for_child(child_cls: Type, data: Mapping[str, Any]) -> Dict[str, Any]:
    child_fields = {f.name for f in fields(child_cls)}
    return {k: data[k] for k in child_fields if k in data}


def _build_dataclass(
    cls: Type[T],
    data: Dict[str, Any],
    path: str,
    *,
    allow_extra: bool,
) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{path}: {cls.__name__} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected dict, got {type(data).__name__}")

    # ------------------------------------------------------------------ #
    # 0.  Pre‑processing hook (opt‑in)                                    #
    # ------------------------------------------------------------------ #
    preprocess = getattr(cls, "__preprocess_dictconfig__", None)
    if callable(preprocess):
        # work on a shallow copy to avoid side‑effects outside builder
        data = preprocess(dict(data)) or data

    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 1.  Extra keys check                                               #
    # ------------------------------------------------------------------ #
    known = {f.name for f in fields(cls)}
    extra = set(data) - known
    if extra:
        if not allow_extra:
            raise ValueError(f"{path}: unexpected fields {sorted(extra)}")
        logger.warning(f"{path}: ignoring extra fields {sorted(extra)}")

    # ------------------------------------------------------------------ #
    # 2.  Build field values                                             #
    # ------------------------------------------------------------------ #
    for f in fields(cls):
        key, fpath = f.name, f"{path}.{f.name}"
        if key not in data:
            continue
        val = data[key]

        # primitive None
        if val is None:
            kwargs[key] = None
            continue

        ftype = hints.get(key)
        origin, args = get_origin(ftype), get_args(ftype)

        # List[Dataclass]   --------------------------------------------
        if origin in (list, List) and args and is_dataclass(args[0]) and isinstance(val, list):
            item_type = args[0]
            parent_defaults = _get_parent_config_for_child(item_type, data)
            kwargs[key] = [
                _build_dataclass(
                    item_type,
                    itm,
                    f"{fpath}[{i}]",
                    allow_extra=allow_extra,
                )
                if isinstance(itm, dict)
                else itm
                for i, itm in enumerate(val)
            ]
            continue

        # Dict[str, Dataclass] ----------------------------------------
        if origin in (dict, Dict) and len(args) == 2 and is_dataclass(args[1]) and isinstance(val, dict):
            value_type = args[1]
            parent_defaults = _get_parent_config_for_child(value_type, data)
            kwargs[key] = {
                k: _build_dataclass(
                    value_type,
                    sub,
                    f"{fpath}[{k!r}]",
                    allow_extra=allow_extra,
                )
                if isinstance(sub, dict)
                else sub
                for k, sub in val.items()
            }
            continue

        # Embedded dataclass ------------------------------------------
        if is_dataclass(ftype) and isinstance(val, dict):
            parent_defaults = _get_parent_config_for_child(ftype, data)
            merged = {**parent_defaults, **val}
            kwargs[key] = _build_dataclass(
                ftype,
                merged,
                fpath,
                allow_extra=allow_extra,
            )
            continue

        # Primitive field ---------------------------------------------
        kwargs[key] = val

    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Public entry‐point                                                         #
# --------------------------------------------------------------------------- #
def dictconfig_to_dataclass(
    cls: Type[T],
    config: DictConfig | dict,
    *,
    allow_extra_keys: bool = False,
) -> T:
    """
    Convert a Hydra DictConfig (or plain dict) into a fully‑typed dataclass
    `cls`, applying any   `__preprocess_dictconfig__`   hooks encountered
    during the build, then validate it.
    """
    data = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    data = data or {}

    instance = _build_dataclass(
        cls,
        data,
        path=cls.__name__,
        allow_extra=allow_extra_keys,
    )
    validate_dataclass(instance, cls, path=cls.__name__)
    return instance


# --------------------------------------------------------------------------- #
# (unchanged)  AWS / wandb helpers                                            #
# --------------------------------------------------------------------------- #
def config_from_path(config_path: str, overrides: DictConfig = None) -> DictConfig:
    if config_path is None:
        # Handle the None case appropriately
        # For example, return a default configuration or raise a more informative error
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
