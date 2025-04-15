"""
This file implements a PolicyStore class that manages loading and caching of trained policies.
It provides functionality to:
- Load policies from local files or remote URIs
- Cache loaded policies to avoid reloading
- Select policies based on metadata filters
- Track policy metadata and versioning

The PolicyStore is used by the training system to manage opponent policies and checkpoints.
"""

import logging
import os
import random
import warnings
from typing import List, Union

import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from wandb.sdk import wandb_run

from metta.agent.metta_agent import make_policy

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
            self._local_path = self.uri[len("file://") :]

    def policy(self) -> nn.Module:
        if self._policy is None:
            pr = self._policy_store._load_from_uri(self.uri)
            self._policy = pr.policy()
            self._local_path = pr.local_path()
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

    def policy(self, policy: Union[str, OmegaConf], selector_type: str = "top", n=1, metric="score") -> PolicyRecord:
        if not isinstance(policy, str):
            policy = policy.uri
        prs = self._policy_records(policy, selector_type, n, metric)
        assert len(prs) == 1, f"Expected 1 policy, got {len(prs)}"
        return prs[0]

    def policies(
        self, policy: Union[str, OmegaConf], selector_type: str = "top", n=1, metric="score"
    ) -> List[PolicyRecord]:
        if not isinstance(policy, str):
            policy = policy.uri
        return self._policy_records(policy, selector_type, n, metric)

    def _policy_records(self, uri, selector_type="top", n=1, metric="score"):
        version = None
        if uri.startswith("wandb://"):
            wandb_uri = uri[len("wandb://") :]
            if ":" in wandb_uri:
                wandb_uri, version = wandb_uri.split(":")
            if wandb_uri.startswith("run/"):
                run_id = wandb_uri[len("run/") :]
                prs = self._prs_from_wandb_run(run_id, version)
            elif wandb_uri.startswith("sweep/"):
                sweep_name = wandb_uri[len("sweep/") :]
                prs = self._prs_from_wandb_sweep(sweep_name, version)
            else:
                prs = self._prs_from_wandb_artifact(wandb_uri, version)

        elif uri.startswith("file://"):
            prs = self._prs_from_path(uri[len("file://") :])
        else:
            prs = self._prs_from_path(uri)

        if len(prs) == 0:
            raise ValueError(f"No policies found at {uri}")

        if selector_type == "all":
            return prs

        elif selector_type == "latest":
            return [prs[0]]

        elif selector_type == "rand":
            return [random.choice(prs)]

        elif selector_type == "top":
            if metric not in prs[0].metadata or prs[0].metadata[metric] is None:
                logger.warning(f"Metric {metric} not found in policy metadata, returning latest policy")
                return [prs[0]]  # return latest if metric not found
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
            raise ValueError(f"Invalid selector type {selector_type}")

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> PolicyRecord:
        policy = make_policy(env, self._cfg)
        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)
        pr = self.save(
            name,
            path,
            policy,
            {
                "action_names": env.action_names(),
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            },
        )
        pr._policy = policy
        return pr

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict):
        logger.info(f"Saving policy to {path}")
        pr = PolicyRecord(self, path, "file://" + path, metadata)
        pr._policy = policy
        pr._policy_store = None
        torch.save(pr, path)
        pr._policy_store = self
        # Don't cache the policy that we just saved,
        # since it might be updated later. We always
        # load the policy from the file when needed.
        pr._policy = None
        self._cached_prs[path] = pr
        return pr

    def add_to_wandb_run(self, run_id: str, pr: PolicyRecord, additional_files=None):
        return self.add_to_wandb_artifact(run_id, "model", pr.metadata, pr.local_path(), additional_files)

    def add_to_wandb_sweep(self, sweep_name: str, pr: PolicyRecord, additional_files=None):
        return self.add_to_wandb_artifact(sweep_name, "sweep_model", pr.metadata, pr.local_path(), additional_files)

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

        return [self._load_from_file(path, metadata_only=True) for path in paths]

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
            PolicyRecord(self, name=a.name, uri="wandb://" + a.qualified_name, metadata=a.metadata) for a in artifacts
        ]

    def _prs_from_wandb_sweep(self, sweep_name: str, version: str = None) -> List[PolicyRecord]:
        return self._prs_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/sweep_model/{sweep_name}", version
        )

    def _prs_from_wandb_run(self, run_id: str, version: str = None) -> List[PolicyRecord]:
        return self._prs_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/model/{run_id}", version
        )

    def _load_from_uri(self, uri: str):
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://") :])
        elif uri.startswith("file://"):
            return self._load_from_file(uri[len("file://") :])
        else:
            return self._load_from_file(uri)

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        if path in self._cached_prs:
            if metadata_only or self._cached_prs[path]._policy is not None:
                return self._cached_prs[path]
        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])
        logger.info(f"Loading policy from {path}")

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"
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

        artifact_path = os.path.join(self._cfg.data_dir, "artifacts", artifact.name)

        if not os.path.exists(artifact_path):
            artifact.download(root=artifact_path)

        logger.info(f"Downloaded artifact {artifact.name} to {artifact_path}")

        pr = self._load_from_file(os.path.join(artifact_path, "model.pt"))
        pr.metadata.update(artifact.metadata)
        return pr
