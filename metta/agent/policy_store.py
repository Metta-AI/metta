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
from typing import Any, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import wandb
import wandb.sdk.wandb_run
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.package import PackageExporter, PackageImporter

from metta.agent.metta_agent import make_policy
from metta.agent.policy_record import PolicyRecord
from metta.rl.policy import load_pytorch_policy

logger = logging.getLogger("policy_store")


class PolicySelectorConfig:
    """Simple config class for policy selection without pydantic dependency."""

    def __init__(self, type: str = "top", metric: str = "score"):
        self.type = type
        self.metric = metric


class PolicyStore:
    def __init__(self, cfg: ListConfig | DictConfig, wandb_run):
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
            logger.info(f"Returning all {len(prs)} policies")
            return prs
        elif selector_type == "latest":
            selected = [prs[0]]
            logger.info(f"Selected latest policy: {selected[0].name}")
            return selected
        elif selector_type == "rand":
            selected = [random.choice(prs)]
            logger.info(f"Selected random policy: {selected[0].name}")
            return selected
        elif selector_type == "top":
            if (
                "eval_scores" in prs[0].metadata
                and prs[0].metadata["eval_scores"] is not None
                and metric in prs[0].metadata["eval_scores"]
            ):
                # Metric is in eval_scores
                logger.info(f"Found metric '{metric}' in metadata['eval_scores']")
                policy_scores = {p: p.metadata.get("eval_scores", {}).get(metric, None) for p in prs}
            elif metric in prs[0].metadata:
                # Metric is directly in metadata
                logger.info(f"Found metric '{metric}' directly in metadata")
                policy_scores = {p: p.metadata.get(metric, None) for p in prs}
            else:
                # Metric not found anywhere
                logger.warning(
                    f"Metric '{metric}' not found in policy metadata or eval_scores, returning latest policy"
                )
                selected = [prs[0]]
                logger.info(f"Selected latest policy (due to missing metric): {selected[0].name}")
                return selected

            policies_with_scores = [p for p, s in policy_scores.items() if s is not None]

            # If more than 20% of the policies have no score, return the latest policy
            if len(policies_with_scores) < len(prs) * 0.8:
                logger.warning("Too many invalid scores, returning latest policy")
                selected = [prs[0]]  # return latest if metric not found
                logger.info(f"Selected latest policy (due to too many invalid scores): {selected[0].name}")
                return selected

            # Sort by metric score (assuming higher is better)
            def get_policy_score(policy: PolicyRecord) -> float:  # Explicitly return a comparable type
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
            for pr in top:
                score = policy_scores[pr]
                logger.info(f"{pr.name:<40} | {score:<20.4f}")

            selected = top[-n:]
            logger.info(f"Selected {len(selected)} top policies by {metric}")
            for i, pr in enumerate(selected):
                logger.info(f"  {i + 1}. {pr.name} (score: {policy_scores[pr]:.4f})")

            return selected
        else:
            raise ValueError(f"Invalid selector type {selector_type}")

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> PolicyRecord:
        policy = make_policy(env, self._cfg)
        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)
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
        self._save_policy(path, policy, pr)
        pr._policy = policy
        return pr

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict) -> PolicyRecord:
        """Convenience method to create and save a policy in one step."""
        pr = PolicyRecord(self, name, "file://" + path, metadata)
        return self._save_policy(path, policy, pr)

    def _save_policy(self, path: str, policy: nn.Module, pr: PolicyRecord) -> PolicyRecord:
        """Save a policy and its metadata using torch.package."""
        logger.info(f"Saving policy to {path} using torch.package")

        policy_class_name = policy.__class__.__module__
        if "torch_package_" in policy_class_name:
            logger.error("Policy class name with torch_package_ prefixes! Did you forget to rebuild the agent?")
            logger.error("Skipping save to prevent pickle errors.")
            return pr

        try:
            with PackageExporter(path, debug=False) as exporter:
                # Apply all packaging rules
                self._apply_packaging_rules(exporter, policy.__class__.__module__, policy.__class__)

                # Save the policy and metadata
                clean_metadata = pr._clean_metadata_for_packaging(pr.metadata)
                exporter.save_pickle("policy_record", "data.pkl", PolicyRecord(None, pr.name, pr.uri, clean_metadata))
                exporter.save_pickle("policy", "model.pkl", policy)

        except Exception as e:
            logger.error(f"torch.package save failed: {e}")
            raise RuntimeError(f"Failed to save policy using torch.package: {e}") from e

        pr._local_path = path
        pr.uri = "file://" + path
        return pr

    def add_to_wandb_run(self, run_id: str, pr: PolicyRecord, additional_files: list[str] | None = None) -> str:
        local_path = pr.local_path()
        if local_path is None:
            raise ValueError("PolicyRecord has no local path")
        return self.add_to_wandb_artifact(run_id, "model", pr.metadata, local_path, additional_files)

    def add_to_wandb_sweep(self, sweep_name: str, pr: PolicyRecord, additional_files: list[str] | None = None) -> str:
        local_path = pr.local_path()
        if local_path is None:
            raise ValueError("PolicyRecord has no local path")
        return self.add_to_wandb_artifact(sweep_name, "sweep_model", pr.metadata, local_path, additional_files)

    def add_to_wandb_artifact(
        self, name: str, type: str, metadata: dict[str, Any], local_path: str, additional_files: list[str] | None = None
    ) -> str:
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
        return artifact.qualified_name

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

    def _load_from_pytorch(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        name = os.path.basename(path)
        pr = PolicyRecord(
            self,
            name,
            "pytorch://" + name,
            {"action_names": [], "agent_step": 0, "epoch": 0, "generation": 0, "train_time": 0},
        )
        pr._policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._cfg.get("pytorch"))
        return pr

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        if path in self._cached_prs and (metadata_only or self._cached_prs[path]._policy is not None):
            return self._cached_prs[path]

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])
        logger.info(f"Loading policy from {path}")

        self._make_codebase_backwards_compatible()

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            importer = PackageImporter(path)
            pr = importer.load_pickle("policy_record", "data.pkl")
            pr._policy_store = self
            if not metadata_only:
                pr._policy = pr.load(path, self._device)
            pr._local_path = path
            self._cached_prs[path] = pr
            return pr
        except Exception as e:
            logger.debug(f"Not a torch.package file: {e}")
            if "PytorchStreamReader failed locating file .data/extern_modules" in str(e):
                logger.info("Detected old checkpoint format, loading as regular PyTorch checkpoint")
                return self._load_legacy_checkpoint(path, metadata_only)
            raise ValueError(f"Failed to load policy from {path}: {e}") from e

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

    def _load_legacy_checkpoint(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        logger.info(f"Loading legacy checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        if isinstance(checkpoint, PolicyRecord):
            pr = checkpoint
            pr._policy_store = self
            pr._local_path = path
            if pr._policy is None and not metadata_only:
                raise ValueError("Legacy PolicyRecord has no policy attached")
            self._cached_prs[path] = pr
            return pr

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Create PolicyRecord with metadata
        pr = PolicyRecord(
            self,
            os.path.basename(path),
            f"file://{path}",
            {
                k: checkpoint.get(k, 0 if k != "action_names" else [])
                for k in ["action_names", "agent_step", "epoch", "generation", "train_time"]
            },
        )
        pr._local_path = path

        if not metadata_only:
            try:
                from types import SimpleNamespace

                # Create mock environment
                obs_shape = checkpoint.get("obs_shape", [34, 11, 11])
                env = SimpleNamespace(
                    single_observation_space=gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                    obs_width=obs_shape[1],
                    obs_height=obs_shape[2],
                    single_action_space=gym.spaces.MultiDiscrete(checkpoint.get("action_space_nvec", [9, 10])),
                    feature_normalizations=checkpoint.get("feature_normalizations", {}),
                    global_features=[],
                )

                policy = make_policy(env, self._cfg)

                # Load state dict
                state_key = next((k for k in ["model_state_dict", "state_dict"] if k in checkpoint), None)
                policy.load_state_dict(checkpoint.get(state_key, checkpoint))

                pr._policy = policy
                logger.info("Successfully loaded legacy checkpoint as MettaAgent")
            except Exception as e:
                raise ValueError(f"Cannot load legacy checkpoint as MettaAgent: {e}") from e

        self._cached_prs[path] = pr
        return pr

    def _apply_packaging_rules(self, exporter, policy_module: Optional[str], policy_class: type) -> None:
        """Apply packaging rules to the exporter based on a configuration."""
        # Define packaging rules using wildcards for conciseness
        rules = [
            # Extern rules: Third-party libs and modules with pydantic dependencies
            (
                "extern",
                [
                    "sys",
                    "torch.**",
                    "numpy.**",
                    "scipy.**",
                    "sklearn.**",
                    "matplotlib.**",
                    "gymnasium.**",
                    "gym.**",
                    "tensordict.**",
                    "einops.**",
                    "hydra.**",
                    "omegaconf.**",
                    "mettagrid.**",
                    "metta.mettagrid.**",
                    "metta.common.util.config",
                    "metta.rl.vecenv",
                    "metta.eval.dashboard_data",
                    "metta.sim.simulation_config",
                    "metta.agent.policy_store",
                ],
            ),
            # Intern rules: Essential metta code for loading policies
            (
                "intern",
                [
                    "metta.agent.policy_record",
                    "metta.agent.lib.**",
                    "metta.agent.util.**",
                    "metta.agent.metta_agent",
                    "metta.agent.brain_policy",
                    "metta.agent.policy_state",
                    "metta.common.util.omegaconf",
                    "metta.common.util.runtime_configuration",
                    "metta.common.util.logger",
                    "metta.common.util.decorators",
                    "metta.common.util.resolvers",
                ],
            ),
            # Mock rules: Exclude these completely
            (
                "mock",
                [
                    "wandb.**",
                    "pufferlib.**",
                    "pydantic.**",
                    "boto3.**",
                    "botocore.**",
                    "duckdb.**",
                    "pandas.**",
                    "typing_extensions",
                    "seaborn",
                    "plotly",
                ],
            ),
        ]

        # Apply rules from the configuration
        for action, patterns in rules:
            for pattern in patterns:
                getattr(exporter, action)(pattern)

        # Handle special cases for the policy's own module
        if policy_module:
            if policy_module == "__main__":
                import inspect

                try:
                    source = inspect.getsource(policy_class)
                    exporter.save_source_string("__main__", f"import torch\nimport torch.nn as nn\n\n{source}")
                except Exception:
                    exporter.extern("__main__")
            elif "test" in policy_module:
                # Extern test modules to prevent them from being packaged
                exporter.extern(policy_module)
