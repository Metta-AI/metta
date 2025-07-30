import os
from dataclasses import dataclass
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypedDict

from metta.common.util.config import config_from_path

METTAGRID_CFG_ROOT = "env/mettagrid"


CfgKind = Literal["env", "curriculum", "map", "unknown"]


@dataclass
class MettagridCfgFileMetadata:
    path: str
    kind: CfgKind

    @staticmethod
    def from_path(path: str) -> "MettagridCfgFileMetadata":
        # Fast path-based classification first
        if path.startswith("game/map_builder/"):
            kind = "map"
        elif path.startswith("curriculum/"):
            kind = "curriculum"
        else:
            kind = "env"

        # Only do expensive content-based detection for files that might be misclassified
        # This includes files in navigation/training/ that might be curriculums
        if (
            path.startswith("navigation/training/")
            or path.startswith("multiagent/experiments/")
            or path.startswith("cooperation/experimental/")
            or path.startswith("navigation_sequence/experiments/")
        ):
            try:
                with hydra.initialize(config_path="../../../../../configs", version_base=None):
                    cfg = config_from_path(METTAGRID_CFG_ROOT + "/" + path)
                    if isinstance(cfg, DictConfig):
                        target = cfg.get("_target_", "")
                        # Check if it's a curriculum
                        if "curriculum" in target.lower():
                            kind = "curriculum"
                        # Check if it's a map generator
                        elif "MapGen" in target or ("map" in target.lower() and "scene" in target.lower()):
                            kind = "map"
            except Exception:
                # If we can't load the config, keep the path-based classification
                pass

        return MettagridCfgFileMetadata(path=path, kind=kind)

    @staticmethod
    def get_all() -> dict[CfgKind, list["MettagridCfgFileMetadata"]]:
        metadata_by_kind: dict[CfgKind, list[MettagridCfgFileMetadata]] = {}

        for root, _, files in os.walk("configs/" + METTAGRID_CFG_ROOT):
            for f in files:
                # there are .map files in config dir
                if not f.endswith(".yaml"):
                    continue
                path = os.path.relpath(os.path.join(root, f), "configs/" + METTAGRID_CFG_ROOT)
                metadata = MettagridCfgFileMetadata.from_path(path)
                if metadata.kind not in metadata_by_kind:
                    metadata_by_kind[metadata.kind] = []
                metadata_by_kind[metadata.kind].append(metadata)

        return metadata_by_kind

    def get_cfg(self) -> "MettagridCfgFile":
        with hydra.initialize(config_path="../../../../../configs", version_base=None):
            cfg = config_from_path(METTAGRID_CFG_ROOT + "/" + self.path)
            if not isinstance(cfg, DictConfig):
                raise ValueError(f"Invalid config type: {type(cfg)}")

        return MettagridCfgFile(metadata=self, cfg=cfg)

    def absolute_path(self) -> str:
        return os.path.join(os.getcwd(), "configs", METTAGRID_CFG_ROOT, self.path)

    def to_dict(self):
        return {
            "absolute_path": self.absolute_path(),
            "path": self.path,
            "kind": self.kind,
        }


@dataclass
class MettagridCfgFile:
    metadata: MettagridCfgFileMetadata
    cfg: DictConfig

    class AsDict(TypedDict):
        metadata: dict
        cfg: dict

    def to_dict(self) -> AsDict:
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=False)
        assert isinstance(cfg_dict, dict)
        return {
            "metadata": self.metadata.to_dict(),
            "cfg": cfg_dict,
        }

    @staticmethod
    def from_path(path: str) -> "MettagridCfgFile":
        return MettagridCfgFileMetadata.from_path(path).get_cfg()

    def get_map_cfg(self) -> DictConfig:
        map_cfg = None
        if self.metadata.kind == "map":
            map_cfg = self.cfg
        elif self.metadata.kind == "env":
            # Check if the config has a game section before trying to access it
            if hasattr(self.cfg, "game") and hasattr(self.cfg.game, "map_builder"):
                map_cfg = self.cfg.game.map_builder
            else:
                raise ValueError(
                    f"Config {self.metadata.path} is an environment config but doesn't have a game.map_builder section"
                )
        else:
            raise ValueError(f"Config {self.metadata.path} is not a map or env (it's a {self.metadata.kind})")

        return map_cfg
