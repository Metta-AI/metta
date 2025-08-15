import os
from dataclasses import dataclass
from typing import Literal

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypedDict

from metta.common.util.config import config_from_path
from metta.mettagrid.curriculum.util import curriculum_from_config_path

METTAGRID_CFG_ROOT = "env/mettagrid"


CfgKind = Literal["env", "curriculum", "map", "unknown"]


@dataclass
class MettagridCfgFileMetadata:
    path: str
    kind: CfgKind

    @staticmethod
    def from_path(path: str) -> "MettagridCfgFileMetadata":
        kind = "unknown"

        # Detect config kind with heuristics.
        # We could load the cfg and parse it, but Hydra takes too long for 150+ configs.
        if path.startswith("game/map_builder/"):
            kind = "map"
        elif path.startswith("curriculum/"):
            kind = "curriculum"
        else:
            with open("configs/" + METTAGRID_CFG_ROOT + "/" + path, "r") as f:
                cfg = yaml.safe_load(f)
                if "game" in cfg:
                    kind = "env"
                elif "curriculum" in cfg.get("_target_", ""):
                    kind = "curriculum"
                elif "env_overrides" in cfg:
                    kind = "curriculum"
                elif "mapgen" in cfg.get("_target_", ""):
                    kind = "map"
                else:
                    kind = "unknown"

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
            map_cfg = self.cfg.game.map_builder
        elif self.metadata.kind == "curriculum":
            hydra_path = "/env/mettagrid/" + self.metadata.path.replace(".yaml", "")
            with hydra.initialize(config_path="../../../../../configs", version_base=None):
                curriculum = curriculum_from_config_path(hydra_path, OmegaConf.create({}))
            task = curriculum.get_task()
            map_cfg = task.original_env_cfg().game.map_builder
        else:
            raise ValueError(f"Config {self.metadata.path} is not a map or env")

        return map_cfg
