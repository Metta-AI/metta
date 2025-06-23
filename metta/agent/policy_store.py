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
from types import SimpleNamespace
from typing import List, Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch import nn

from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_record import PolicyRecord
from metta.util.config import Config
from metta.util.wandb.wandb_context import WandbRun

logger = logging.getLogger("policy_store")


class PolicySelectorConfig(Config):
    type: str = "top"
    metric: str = "score"


class PolicyStore:
    def __init__(self, cfg: ListConfig | DictConfig, wandb_run: WandbRun | None):
        self._cfg = cfg
        self._device = cfg.device
        self._wandb_run = wandb_run
        self._cached_prs = {}

    def policy(
        self, policy: Union[str, ListConfig | DictConfig], selector_type: str = "top", n=1, metric="score"
    ) -> PolicyRecord:
        if not isinstance(policy, str):
            policy = policy.uri
        prs = self._policy_records(policy, selector_type, n, metric)
        assert len(prs) == 1, f"Expected 1 policy, got {len(prs)}"
        return prs[0]

    def policies(
        self, policy: Union[str, ListConfig | DictConfig], selector_type: str = "top", n: int = 1, metric: str = "score"
    ) -> List[PolicyRecord]:
        if not isinstance(policy, str):
            policy = policy.uri
        return self._policy_records(policy, selector_type=selector_type, n=n, metric=metric)

    def _policy_records(self, uri, selector_type="top", n=1, metric: str = "score"):
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
        elif uri.startswith("pytorch://"):
            prs = self._prs_from_pytorch(uri[len("pytorch://") :])
        else:
            prs = self._prs_from_path(uri)

        if len(prs) == 0:
            raise ValueError(f"No policies found at {uri}")

        logger.info(f"Found {len(prs)} policies at {uri}")

        if selector_type == "all":
            return prs
        elif selector_type == "latest":
            return [prs[0]]
        elif selector_type == "rand":
            return [random.choice(prs)]
        elif selector_type == "top":
            # Try to find metric in metadata
            if (
                "eval_scores" in prs[0].metadata
                and prs[0].metadata["eval_scores"] is not None
                and metric in prs[0].metadata["eval_scores"]
            ):
                policy_scores = {p: p.metadata.get("eval_scores", {}).get(metric, None) for p in prs}
            elif metric in prs[0].metadata:
                policy_scores = {p: p.metadata.get(metric, None) for p in prs}
            else:
                logger.warning(f"Metric '{metric}' not found, returning latest policy")
                return [prs[0]]

            policies_with_scores = [p for p, s in policy_scores.items() if s is not None]

            # If more than 20% of the policies have no score, return the latest policy
            if len(policies_with_scores) < len(prs) * 0.8:
                logger.warning("Too many invalid scores, returning latest policy")
                return [prs[0]]

            # Sort by metric score (assuming higher is better)
            def get_policy_score(policy: PolicyRecord) -> float:
                score = policy_scores.get(policy)
                return float("-inf") if score is None else score

            top = sorted(policies_with_scores, key=get_policy_score)[-n:]

            if len(top) < n:
                logger.warning(f"Only found {len(top)} policies matching criteria, requested {n}")

            return top[-n:]
        else:
            raise ValueError(f"Invalid selector type {selector_type}")

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> PolicyRecord:
        """Create a new policy and save it with torch.package."""
        # Create observation space for the policy
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Create MettaAgent directly with component mode
        policy = hydra.utils.instantiate(
            self._cfg.agent,
            obs_space=obs_space,
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            action_space=env.single_action_space,
            feature_normalizations=env.feature_normalizations,
            device=self._cfg.device,
            _target_="metta.agent.metta_agent.MettaAgent",
            _recursive_=False,
        )

        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)

        # Create PolicyRecord with metadata
        pr = PolicyRecord(
            self,
            name,
            f"file://{path}",
            {
                "action_names": env.action_names,
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            },
        )

        # Save with torch.package
        pr.save(path, policy)
        pr._policy = policy

        return pr

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict):
        """Save a policy using PolicyRecord's save method."""
        pr = PolicyRecord(self, name, "file://" + path, metadata)
        return pr.save(path, policy)

    def add_to_wandb_run(self, run_id: str, pr: PolicyRecord, additional_files=None):
        local_path = pr.local_path()
        if local_path is None:
            raise ValueError("PolicyRecord has no local path")
        return self.add_to_wandb_artifact(run_id, "model", pr.metadata, local_path, additional_files)

    def add_to_wandb_sweep(self, sweep_name: str, pr: PolicyRecord, additional_files=None):
        local_path = pr.local_path()
        if local_path is None:
            raise ValueError("PolicyRecord has no local path")
        return self.add_to_wandb_artifact(sweep_name, "sweep_model", pr.metadata, local_path, additional_files)

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

    def _prs_from_wandb_artifact(self, uri: str, version: Optional[str] = None) -> List[PolicyRecord]:
        # Check if wandb is disabled before proceeding
        if (
            not hasattr(self._cfg, "wandb")
            or not hasattr(self._cfg.wandb, "entity")
            or not hasattr(self._cfg.wandb, "project")
        ):
            raise ValueError(
                f"Cannot load wandb artifact '{uri}' when wandb is disabled (wandb=off). "
                "Either enable wandb or use a local policy URI (file://) instead."
            )
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

    def _prs_from_wandb_sweep(self, sweep_name: str, version: Optional[str] = None) -> List[PolicyRecord]:
        return self._prs_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/sweep_model/{sweep_name}", version
        )

    def _prs_from_wandb_run(self, run_id: str, version: Optional[str] = None) -> List[PolicyRecord]:
        return self._prs_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/model/{run_id}", version
        )

    def _prs_from_pytorch(self, path: str) -> List[PolicyRecord]:
        return [self._load_from_pytorch(path)]

    def load_from_uri(self, uri: str) -> PolicyRecord:
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://") :])
        if uri.startswith("file://"):
            return self._load_from_file(uri[len("file://") :])
        if uri.startswith("pytorch://"):
            return self._load_from_pytorch(uri[len("pytorch://") :])
        if "://" not in uri:
            return self._load_from_file(uri)

        raise ValueError(f"Invalid URI: {uri}")

    def _load_from_pytorch(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a policy from a PyTorch checkpoint or JIT model."""
        name = os.path.basename(path)

        # Common metadata for pytorch loads
        default_metadata = {
            "action_names": [],
            "agent_step": 0,
            "epoch": 0,
            "generation": 0,
            "train_time": 0,
        }

        # Try to load as a JIT model first
        try:
            jit_model = torch.jit.load(path, map_location=self._device)
            pr = PolicyRecord(self, name, "pytorch://" + name, default_metadata)
            # Create MettaAgent in PyTorch mode
            pr._policy = MettaAgent(wrapped_policy=jit_model)
            return pr
        except Exception:
            # Fall back to regular PyTorch checkpoint loading
            policy = self._build_pytorch_policy(path, self._device, pytorch_cfg=self._cfg.get("pytorch"))
            pr = PolicyRecord(self, name, "pytorch://" + name, default_metadata)
            pr._policy = policy
            return pr

    def _build_pytorch_policy(self, path: str, device: str = "cpu", pytorch_cfg: Optional[DictConfig] = None):
        """Build a policy from a PyTorch checkpoint."""
        weights = torch.load(path, map_location=device, weights_only=True)

        try:
            num_actions, hidden_size = weights["policy.actor.0.weight"].shape
            num_action_args, _ = weights["policy.actor.1.weight"].shape
            _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
        except Exception as e:
            logger.warning(f"Failed automatic parse from weights: {e}")
            # TODO -- fix all magic numbers
            num_actions, num_action_args = 9, 10
            _, obs_channels = 128, 34

        # Create environment namespace
        env = SimpleNamespace(
            single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
            single_observation_space=SimpleNamespace(shape=tuple(torch.tensor([obs_channels, 11, 11]).tolist())),
        )

        policy = instantiate(pytorch_cfg, env=env, policy=None)
        policy.load_state_dict(weights)

        # Create MettaAgent in PyTorch mode
        wrapped_agent = MettaAgent(wrapped_policy=policy)
        wrapped_agent.hidden_size = hidden_size
        return wrapped_agent.to(device)

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a policy from a file using PolicyRecord's load method."""
        if path in self._cached_prs:
            if metadata_only or self._cached_prs[path]._policy is not None:
                return self._cached_prs[path]

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])

        logger.info(f"Loading policy from {path}")

        # First try to load as a torch.package
        try:
            from torch.package import PackageImporter

            importer = PackageImporter(path)

            # Try to load the policy record
            pr = importer.load_pickle("policy_record", "data.pkl")
            pr._policy_store = self

            if not metadata_only:
                pr._policy = pr.load(path, self._device)

            pr._local_path = path
            self._cached_prs[path] = pr
            return pr

        except Exception as e:
            logger.debug(f"Not a torch.package file: {e}")

            # Try to load as a regular checkpoint
            try:
                checkpoint = torch.load(path, map_location=self._device, weights_only=False)

                # For legacy checkpoints without torch.package, we can't load them
                raise ValueError(
                    f"Cannot load policy from {path}: This appears to be a legacy checkpoint "
                    "without torch.package. Please ensure policies are saved using torch.package."
                )

            except Exception as e:
                if "Cannot load" in str(e):
                    raise  # Re-raise our custom errors
                raise ValueError(f"Failed to load policy from {path}: {e}")

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
