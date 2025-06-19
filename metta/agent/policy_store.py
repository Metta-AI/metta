"""
This file implements a PolicyStore class that manages loading and caching of trained policies.
It provides functionality to:
- Load policies from local files or remote URIs
- Cache loaded policies to avoid reloading
- Select policies based on metadata filters
- Track policy metadata and versioning

The PolicyStore is used by the training system to manage opponent policies and checkpoints.
"""

import collections
import logging
import os
import random
import sys
import warnings
from typing import List, Optional, Union

import torch
import wandb
from omegaconf import DictConfig, ListConfig
from torch import nn

from metta.agent.metta_agent import MettaAgent, make_policy
from metta.rl.policy import load_policy
from metta.util.config import Config
from metta.util.wandb.wandb_context import WandbRun

logger = logging.getLogger("policy_store")


class PolicySelectorConfig(Config):
    type: str = "top"
    metric: str = "score"


# For backward compatibility
PolicyRecord = MettaAgent


class PolicyStore:
    def __init__(self, cfg: ListConfig | DictConfig, wandb_run: WandbRun | None):
        self._cfg = cfg
        self._device = cfg.device
        self._wandb_run = wandb_run
        self._cached_mas = {}
        self._made_codebase_backwards_compatible = False

    def policy(
        self, policy: Union[str, ListConfig | DictConfig], selector_type: str = "top", n=1, metric="score"
    ) -> MettaAgent:
        if not isinstance(policy, str):
            policy = policy.uri
        mas = self._policy_records(policy, selector_type, n, metric)
        assert len(mas) == 1, f"Expected 1 policy, got {len(mas)}"
        return mas[0]

    def policies(
        self, policy: Union[str, ListConfig | DictConfig], selector_type: str = "top", n: int = 1, metric: str = "score"
    ) -> List[MettaAgent]:
        if not isinstance(policy, str):
            policy = policy.uri
        return self._policy_records(policy, selector_type, n=n, metric=metric)

    def _policy_records(self, uri, selector_type="top", n=1, metric: str = "score"):
        version = None
        if uri.startswith("wandb://"):
            wandb_uri = uri[len("wandb://") :]
            if ":" in wandb_uri:
                wandb_uri, version = wandb_uri.split(":")
            if wandb_uri.startswith("run/"):
                run_id = wandb_uri[len("run/") :]
                mas = self._mas_from_wandb_run(run_id, version)
            elif wandb_uri.startswith("sweep/"):
                sweep_name = wandb_uri[len("sweep/") :]
                mas = self._mas_from_wandb_sweep(sweep_name, version)
            else:
                mas = self._mas_from_wandb_artifact(wandb_uri, version)
        elif uri.startswith("file://"):
            mas = self._mas_from_path(uri[len("file://") :])
        elif uri.startswith("pytorch://"):
            mas = self._mas_from_pytorch(uri[len("pytorch://") :])
        else:
            mas = self._mas_from_path(uri)

        if len(mas) == 0:
            raise ValueError(f"No policies found at {uri}")

        logger.info(f"Found {len(mas)} policies at {uri}")

        if selector_type == "all":
            logger.info(f"Returning all {len(mas)} policies")
            return mas
        elif selector_type == "latest":
            selected = [mas[0]]
            logger.info(f"Selected latest policy: {selected[0].name}")
            return selected
        elif selector_type == "rand":
            selected = [random.choice(mas)]
            logger.info(f"Selected random policy: {selected[0].name}")
            return selected
        elif selector_type == "top":
            if (
                "eval_scores" in mas[0].metadata
                and mas[0].metadata["eval_scores"] is not None
                and metric in mas[0].metadata["eval_scores"]
            ):
                # Metric is in eval_scores
                logger.info(f"Found metric '{metric}' in metadata['eval_scores']")
                policy_scores = {p: p.metadata.get("eval_scores", {}).get(metric, None) for p in mas}
            elif metric in mas[0].metadata:
                # Metric is directly in metadata
                logger.info(f"Found metric '{metric}' directly in metadata")
                policy_scores = {p: p.metadata.get(metric, None) for p in mas}
            else:
                # Metric not found anywhere
                logger.warning(
                    f"Metric '{metric}' not found in policy metadata or eval_scores, returning latest policy"
                )
                selected = [mas[0]]
                logger.info(f"Selected latest policy (due to missing metric): {selected[0].name}")
                return selected

            policies_with_scores = [p for p, s in policy_scores.items() if s is not None]

            # If more than 20% of the policies have no score, return the latest policy
            if len(policies_with_scores) < len(mas) * 0.8:
                logger.warning("Too many invalid scores, returning latest policy")
                selected = [mas[0]]  # return latest if metric not found
                logger.info(f"Selected latest policy (due to too many invalid scores): {selected[0].name}")
                return selected

            # Sort by metric score (assuming higher is better)
            def get_policy_score(policy: MettaAgent) -> float:  # Explicitly return a comparable type
                score = policy_scores.get(policy)
                if score is None:
                    return float("-inf")  # Or another appropriate default
                return score

            top = sorted(policies_with_scores, key=get_policy_score)[-n:]

            if len(top) < n:
                logger.warning(f"Only found {len(top)} policies matching criteria, requested {n}")

            logger.info(f"Top {len(top)} policies by {metric}:")
            logger.info(f"{'Policy':<40} | {metric:<20}")
            logger.info("-" * 62)
            for ma in top:
                score = policy_scores[ma]
                logger.info(f"{ma.name:<40} | {score:<20.4f}")

            selected = top[-n:]
            logger.info(f"Selected {len(selected)} top policies by {metric}")
            for i, ma in enumerate(selected):
                logger.info(f"  {i + 1}. {ma.name} (score: {policy_scores[ma]:.4f})")

            return selected
        else:
            raise ValueError(f"Invalid selector type {selector_type}")

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> MettaAgent:
        policy = make_policy(env, self._cfg)
        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)
        ma = self.save(
            name,
            path,
            policy,
            {
                "action_names": env.action_names,
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            },
        )
        return ma

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict):
        logger.info(f"Saving policy to {path}")
        # If policy is already a MettaAgent, update its metadata
        if isinstance(policy, MettaAgent):
            policy.name = path
            policy.uri = "file://" + path
            policy.metadata = metadata
            policy._local_path = path
            # Clear _policy_store temporarily to avoid circular reference issues during save
            temp_policy_store = getattr(policy, "_policy_store", None)
            policy._policy_store = None
            ma = policy
        else:
            # Create a wrapper MettaAgent for old-style policies
            ma = MettaAgent(policy_store=self, name=path, uri="file://" + path, metadata=metadata)
            ma._policy = policy
            temp_policy_store = None

        torch.save(ma, path)

        # Restore _policy_store after save
        ma._policy_store = self

        # Don't cache the policy that we just saved,
        # since it might be updated later. We always
        # load the policy from the file when needed.
        if hasattr(ma, "_policy"):
            ma._policy = None
        self._cached_mas[path] = ma
        return ma

    def add_to_wandb_run(self, run_id: str, ma: MettaAgent, additional_files=None):
        local_path = ma.local_path()
        if local_path is None:
            raise ValueError("MettaAgent has no local path")
        return self.add_to_wandb_artifact(run_id, "model", ma.metadata, local_path, additional_files)

    def add_to_wandb_sweep(self, sweep_name: str, ma: MettaAgent, additional_files=None):
        local_path = ma.local_path()
        if local_path is None:
            raise ValueError("MettaAgent has no local path")
        return self.add_to_wandb_artifact(sweep_name, "sweep_model", ma.metadata, local_path, additional_files)

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

    def _mas_from_path(self, path: str) -> List[MettaAgent]:
        paths = []

        if path.endswith(".pt"):
            paths.append(path)
        else:
            paths.extend([os.path.join(path, p) for p in os.listdir(path) if p.endswith(".pt")])

        return [self._load_from_file(path, metadata_only=True) for path in paths]

    def _mas_from_wandb_artifact(self, uri: str, version: Optional[str] = None) -> List[MettaAgent]:
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
            MettaAgent(policy_store=self, name=a.name, uri="wandb://" + a.qualified_name, metadata=a.metadata)
            for a in artifacts
        ]

    def _mas_from_wandb_sweep(self, sweep_name: str, version: Optional[str] = None) -> List[MettaAgent]:
        return self._mas_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/sweep_model/{sweep_name}", version
        )

    def _mas_from_wandb_run(self, run_id: str, version: Optional[str] = None) -> List[MettaAgent]:
        return self._mas_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/model/{run_id}", version
        )

    def _mas_from_pytorch(self, path: str) -> List[MettaAgent]:
        return [self._load_from_pytorch(path)]

    def load_from_uri(self, uri: str) -> MettaAgent:
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://") :])
        if uri.startswith("file://"):
            return self._load_from_file(uri[len("file://") :])
        if uri.startswith("pytorch://"):
            return self._load_from_pytorch(uri[len("pytorch://") :])
        if "://" not in uri:
            return self._load_from_file(uri)

        raise ValueError(f"Invalid URI: {uri}")

    def _make_codebase_backwards_compatible(self):
        """
        torch.load expects the codebase to be in the same structure as when the model was saved.

        We can use this function to alias old layout structures. For now we are supporting:
        - agent --> metta.agent
        """
        # Memoize
        if getattr(self, "_made_codebase_backwards_compatible", False):
            return
        self._made_codebase_backwards_compatible = True

        # Handle agent --> metta.agent
        sys.modules["agent"] = sys.modules["metta.agent"]
        modules_queue = collections.deque(["metta.agent"])

        processed = set()
        while modules_queue:
            module_name = modules_queue.popleft()
            if module_name in processed:
                continue
            processed.add(module_name)

            if module_name not in sys.modules:
                continue
            module = sys.modules[module_name]
            old_name = module_name.replace("metta.agent", "agent")
            sys.modules[old_name] = module

            # Find all submodules
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                except (ImportError, AttributeError):
                    continue
                if hasattr(attr, "__module__"):
                    attr_module = getattr(attr, "__module__", None)

                    # If it's a module and part of metta.agent, queue it
                    if attr_module and attr_module.startswith("metta.agent"):
                        modules_queue.append(attr_module)

                submodule_name = f"{module_name}.{attr_name}"
                if submodule_name in sys.modules:
                    modules_queue.append(submodule_name)

    def _load_from_pytorch(self, path: str, metadata_only: bool = False) -> MettaAgent:
        policy = load_policy(path, self._device, puffer=self._cfg.puffer)
        name = os.path.basename(path)
        ma = MettaAgent(
            policy_store=self,
            name=name,
            uri="pytorch://" + name,
            metadata={
                "action_names": [],
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            },
        )
        ma._policy = policy
        return ma

    def _load_from_file(self, path: str, metadata_only: bool = False) -> MettaAgent:
        if path in self._cached_mas:
            if metadata_only or (
                hasattr(self._cached_mas[path], "_policy") and self._cached_mas[path]._policy is not None
            ):
                return self._cached_mas[path]
        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])
        logger.info(f"Loading policy from {path}")

        self._make_codebase_backwards_compatible()

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            loaded = torch.load(
                path,
                map_location=self._device,
                weights_only=False,
            )

            # Handle backward compatibility - check if we loaded an old PolicyRecord
            if hasattr(loaded, "_policy_store"):
                # This is either an old PolicyRecord or a new MettaAgent with metadata
                ma = loaded
                ma._policy_store = self
                ma._local_path = path

                # Important: if this MettaAgent has components, it's a real MettaAgent, not a PolicyRecord
                # We don't need to set _policy in this case
                if not hasattr(ma, "components"):
                    # This is likely a PolicyRecord-style MettaAgent
                    # The _policy will be loaded lazily when needed
                    pass
            else:
                # This is a raw policy, wrap it in MettaAgent
                name = os.path.basename(path)
                ma = MettaAgent(
                    policy_store=self,
                    name=name,
                    uri="file://" + path,
                    metadata={
                        "action_names": [],
                        "agent_step": 0,
                        "epoch": 0,
                        "generation": 0,
                        "train_time": 0,
                    },
                )
                ma._policy = loaded
                ma._local_path = path

            self._cached_mas[path] = ma
            if metadata_only:
                if hasattr(ma, "_policy"):
                    ma._policy = None
                ma._local_path = None
            return ma

    def _load_wandb_artifact(self, qualified_name: str):
        logger.info(f"Loading policy from wandb artifact {qualified_name}")

        artifact = wandb.Api().artifact(qualified_name)

        artifact_path = os.path.join(self._cfg.data_dir, "artifacts", artifact.name)

        if not os.path.exists(artifact_path):
            artifact.download(root=artifact_path)

        logger.info(f"Downloaded artifact {artifact.name} to {artifact_path}")

        ma = self._load_from_file(os.path.join(artifact_path, "model.pt"))
        ma.metadata.update(artifact.metadata)
        return ma
