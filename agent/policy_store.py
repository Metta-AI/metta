"""
This file implements a PolicyStore class that manages loading and caching of trained policies.
It provides functionality to:
- Load policies from local files or remote URIs
- Cache loaded policies to avoid reloading
- Select policies based on metadata filters
- Track policy metadata and versioning

The PolicyStore is used by the training system to manage opponent policies and checkpoints.
"""

from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import wandb
from rl.pufferlib.puffer_agent_wrapper import make_policy
import torch
import os
import warnings
import logging
from typing import List
import random
from wandb.sdk import wandb_run
from torch import nn

logger = logging.getLogger("policy_store")

class PolicyRecord:
    def __init__(self, policy_store, name: str, uri: str, metadata: dict):
        self._policy_store = policy_store
        self.name = name
        self.uri = uri
        self.metadata = metadata
        self._policy = None
        self._local_path = None

        if self.uri.startswith("file://"):
            self._local_path = self.uri[len("file://"):]

    def policy(self) -> nn.Module:
        if self._policy is None:
            self._policy = self._policy_store._load_from_uri(self.uri).policy()
        return self._policy

    def num_params(self):
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self):
        return self._local_path

class PolicyStore:
    def __init__(self, cfg: OmegaConf, wandb_run: wandb_run.Run):
        self._cfg = cfg
        self._device = cfg.device
        self._wandb_run = wandb_run
        self._cached_prs = {}

    def policy(self, policy_selector_cfg: OmegaConf) -> PolicyRecord:
        if isinstance(policy_selector_cfg, str):
            return self._load_from_uri(policy_selector_cfg)
        prs = self.policies(policy_selector_cfg)
        assert len(prs) == 1, f"Expected 1 policy, got {len(prs)}"
        return prs[0]

    def get_prs(self, uri, ptype, n, metric):
        version = None
        if uri.startswith("wandb://"):
            wandb_uri = uri[len("wandb://"):]
            if ":" in wandb_uri:
                wandb_uri, version = wandb_uri.split(":")
            if wandb_uri.startswith("run/"):
                run_id = wandb_uri[len("run/"):]
                prs = self._prs_from_wandb_run(run_id, version)
            elif wandb_uri.startswith("sweep/"):
                sweep_name = wandb_uri[len("sweep/"):]
                prs = self._prs_from_wandb_sweep(sweep_name, version)
            else:
                prs = self._prs_from_wandb_artifact(wandb_uri, version)

        elif uri.startswith("file://"):
            prs = self._prs_from_path(uri[len("file://"):])
        else:
            prs = self._prs_from_path(uri)

        if ptype == "rand":
            return [random.choice(prs)]

        elif ptype == "top":
            metric = metric

            top = sorted(prs, key=lambda x: x.metadata.get(metric, 0))[-n:]
            if len(top) < n:
                logger.warning(f"Only found {len(top)} policies matching criteria, requested {n}")

            logger.info(f"Top {n} policies by {metric}:")
            logger.info(f"{'Policy':<40} | {metric:<20}")
            logger.info("-" * 62)
            for pr in top:
                logger.info(f"{pr.name:<40} | {pr.metadata.get(metric, 0):<20.4f}")

            return top[-n:]
        else:
            raise ValueError(f"Invalid selector type {ptype}")


    def policies(self, policy_selector_cfg: OmegaConf) -> List[PolicyRecord]:
        prs = []
        if isinstance(policy_selector_cfg.uri, ListConfig):
            for uri in policy_selector_cfg.uri:
                prs += self.get_prs(uri, policy_selector_cfg.type, policy_selector_cfg.range, policy_selector_cfg.metric)
        else:
            prs = self.get_prs(policy_selector_cfg.uri, policy_selector_cfg.type, policy_selector_cfg.range, policy_selector_cfg.metric)
                
        for k,v in policy_selector_cfg.filters.items():
            prs = [pr for pr in prs if pr.metadata.get(k, None) == v]

        if len(prs) == 0:
            logger.warning("No policies found matching criteria")
            return []

        return prs

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> PolicyRecord:
        policy = make_policy(env, self._cfg)
        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)
        pr = self.save(name, path, policy, {
            "action_names": env.action_names(),
            "agent_step": 0,
            "epoch": 0,
            "generation": 0,
        })
        pr._policy = policy
        return pr

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict):
        logger.info(f"Saving policy to {path}")
        pr = PolicyRecord(self, path, "file://" + path, metadata)
        pr._policy = policy
        pr._policy_store = None
        torch.save(pr, path)
        pr._policy_store = self
        self._cached_prs[path] = pr
        return pr

    def add_to_wandb_run(self, run_id: str, pr: PolicyRecord, additional_files=None):
        return self.add_to_wandb_artifact(
            run_id,
            "model",
            pr.metadata,
            pr.local_path(),
            additional_files
        )

    def add_to_wandb_sweep(self, sweep_name: str, pr: PolicyRecord, additional_files=None):
        return self.add_to_wandb_artifact(
            sweep_name,
            "sweep_model",
            pr.metadata,
            pr.local_path(),
            additional_files
        )

    def add_to_wandb_artifact(self, name: str, type: str, metadata: dict, local_path: str, additional_files=None):
        if self._wandb_run is None:
            raise ValueError("PolicyStore was not initialized with a wandb run")

        additional_files = additional_files or []

        artifact = wandb.Artifact(name, type=type, metadata=metadata)
        artifact.add_file(local_path, name="model.pt")
        for file in additional_files:
            artifact.add_file(file)
        artifact.save()
        artifact.wait()
        logger.info(f"Added artifact {artifact.qualified_name}")
        self._wandb_run.log_artifact(artifact)

    def _prs_from_path(self, path: str) -> List[PolicyRecord]:
        paths = []

        if path.endswith(".pt"):
            paths.append(path)
        else:
            paths.extend([os.path.join(path, p) for p in os.listdir(path) if p.endswith(".pt")])

        return [
            self._load_from_file(path, metadata_only=True)
            for path in paths
        ]

    def _prs_from_wandb_artifact(self, uri: str, version: str = None) -> List[PolicyRecord]:
        entity, project, artifact_type, name = uri.split("/")
        path = f"{entity}/{project}/{name}"
        if not wandb.Api().artifact_collection_exists(type=artifact_type, name=path):
            logger.warning(f"No artifact collection found at {uri}")
            return []
        artifact_collection = wandb.Api().artifact_collection(type_name=artifact_type, name=path)

        artifacts = artifact_collection.artifacts()

        if version is not None:
            artifacts = [a for a in artifacts if a.version == version]

        return [
            PolicyRecord(
                self,
                name=a.name,
                uri="wandb://" + a.qualified_name,
                metadata=a.metadata
            ) for a in artifacts
        ]

    def _prs_from_wandb_sweep(self, sweep_name: str, version: str = None) -> List[PolicyRecord]:
        return self._prs_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/sweep_model/{sweep_name}",
            version
        )

    def _prs_from_wandb_run(self, run_id: str, version: str = None) -> List[PolicyRecord]:
        return self._prs_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/model/{run_id}",
            version
        )

    def _load_from_uri(self, uri: str):
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://"):])
        elif uri.startswith("file://"):
            return self._load_from_file(uri[len("file://"):])
        else:
            return self._load_from_file(uri)

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        if path in self._cached_prs:
            return self._cached_prs[path]

        logger.info(f"Loading policy from {path}")
        assert path.endswith('.pt'), f"Policy file {path} does not have a .pt extension"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pr = torch.load(
                path,
                map_location=self._device,
                weights_only=False,
            )
            pr._policy_store = self
            pr._local_path = path
            self._cached_prs[path] = pr
            if metadata_only:
                pr._policy = None
                pr._local_path = None
            logger.info(f"Loaded policy from {path} with metadata {pr.metadata}")
            return pr

    def _load_wandb_artifact(self, qualified_name: str):
        logger.info(f"Loading policy from wandb artifact {qualified_name}")

        artifact = wandb.Api().artifact(qualified_name)

        data_dir = artifact.download(
            root=os.path.join(self._cfg.data_dir, "artifacts", artifact.name))
        logger.info(f"Downloaded artifact {artifact.name} to {data_dir}")

        pr = self._load_from_file(
            os.path.join(data_dir, "model.pt")
        )
        pr.metadata.update(artifact.metadata)
        return pr
