import os
import pathlib
from typing import IO, Any, Union

from omegaconf import OmegaConf
from omegaconf.base import DictKeyType
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class PatchedDictConfig(DictConfig):
    def _get_impl(self, key: DictKeyType, default_value: Any, validate_key: bool = True) -> Any:
        if not isinstance(key, str):
            # `key$load` feature is only supported for string keys
            return super()._get_impl(key, default_value, validate_key)

        if key.endswith("$load"):
            return super()._get_impl(key, default_value, validate_key)

        load_key = f"{key}$load"
        load_val = None
        if load_key in self:
            source_file = self._metadata._source_file  # type: ignore
            load_val = MettaConf.load(os.path.join(os.path.dirname(source_file), self[load_key]))

        if load_val is None:
            # no load key, fall back to the default implementation
            return super()._get_impl(key, default_value, validate_key)

        if key not in self:
            # we have `key$load` but no `key`
            return load_val

        # both `key` and `key$load` exist
        val = super()._get_impl(key, default_value, validate_key)
        val = OmegaConf.merge(load_val, val)

        return val


class PatchedListConfig(ListConfig):
    # TODO
    pass


def patch_cfg(cfg: Union[DictConfig, ListConfig], source: str) -> Union[PatchedDictConfig, PatchedListConfig]:
    # pick the correct subclass
    if isinstance(cfg, DictConfig):
        target_cls = PatchedDictConfig
    else:
        target_cls = PatchedListConfig

    object.__setattr__(cfg, "__class__", target_cls)

    # stash the filename on the existing metadata object
    cfg._metadata._source_file = source  # type: ignore

    return cfg  # type: ignore


class MettaConf(OmegaConf):
    @staticmethod
    def load(file_: Union[str, pathlib.Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
        cfg = OmegaConf.load(file_)

        if isinstance(file_, (str, pathlib.Path)):
            cfg = patch_cfg(cfg, str(file_))

        return cfg
