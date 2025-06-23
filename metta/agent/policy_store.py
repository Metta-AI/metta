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
from typing import List, Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
import wandb

# Import wandb types directly to avoid pydantic dependency
import wandb.sdk.wandb_run
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.package import PackageExporter, PackageImporter

from metta.rl.policy import load_pytorch_policy

logger = logging.getLogger("policy_store")


class PolicySelectorConfig:
    """Simple config class for policy selection without pydantic dependency."""

    def __init__(self, type: str = "top", metric: str = "score"):
        self.type = type
        self.metric = metric


class PolicyRecord:
    def __init__(self, policy_store: Optional["PolicyStore"], name: str, uri: str, metadata: dict):
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
            pr = self._policy_store.load_from_uri(self.uri)
            self._policy = pr.policy()
            self._local_path = pr.local_path()
        return self._policy

    def policy_as_metta_agent(self):
        """Get the policy as a MettaAgent, DistributedMettaAgent, or PytorchAgent."""
        policy = self.policy()
        # Check by class name to avoid importing at module level
        valid_types = {"MettaAgent", "DistributedMettaAgent", "PytorchAgent"}
        if type(policy).__name__ not in valid_types:
            raise TypeError(f"Expected MettaAgent, DistributedMettaAgent, or PytorchAgent, got {type(policy).__name__}")
        return policy

    def num_params(self) -> int:
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self) -> Optional[str]:
        return self._local_path

    def __repr__(self):
        """Generate a detailed representation of the PolicyRecord."""
        # Basic policy record info
        lines = [f"PolicyRecord(name={self.name}, uri={self.uri})"]

        # Add key metadata if available
        important_keys = ["epoch", "agent_step", "generation", "score"]
        metadata_items = []
        for k in important_keys:
            if k in self.metadata:
                metadata_items.append(f"{k}={self.metadata[k]}")

        if metadata_items:
            lines.append(f"Metadata: {', '.join(metadata_items)}")

        # Load policy if not already loaded
        try:
            policy = self.policy()

            # Add total parameter count
            total_params = sum(p.numel() for p in policy.parameters())
            trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

            # Add module structure (simplified version)
            lines.append("\nKey Modules:")
            # Check if it's a component-based policy by looking for components attribute
            items = policy.components.items() if hasattr(policy, "components") else policy.named_modules()
            # Component-based policy (MettaAgent)
            for name, module in items:
                if name and "." not in name:  # Top-level modules only
                    module_type = module.__class__.__name__
                    param_count = sum(p.numel() for p in module.parameters())
                    if param_count > 0:
                        lines.append(f"  {name}: {module_type} ({param_count:,} params)")

        except Exception as e:
            lines.append(f"Error loading policy: {str(e)}")

        return "\n".join(lines)

    def _clean_metadata_for_packaging(self, metadata: dict) -> dict:
        """Clean metadata to remove any objects that can't be packaged."""
        import copy

        def clean_value(v):
            # Check if it's a wandb object
            if hasattr(v, "__module__") and v.__module__ and "wandb" in v.__module__:
                return None  # Remove wandb objects
            elif isinstance(v, dict):
                return {k: clean_value(val) for k, val in v.items() if clean_value(val) is not None}
            elif isinstance(v, list):
                return [clean_value(item) for item in v if clean_value(item) is not None]
            elif isinstance(v, (str, int, float, bool, type(None))):
                return v
            elif hasattr(v, "__dict__"):
                # For other objects, try to convert to a simple representation
                try:
                    return str(v)
                except:
                    return None
            else:
                return v

        return clean_value(copy.deepcopy(metadata))

    def save(self, path: str, policy: nn.Module) -> "PolicyRecord":
        """Save a policy using torch.package for automatic dependency management."""
        logger.info(f"Saving policy to {path} using torch.package")

        # Update local path
        self._local_path = path
        self.uri = "file://" + path

        # Use the policy directly (no unwrapping needed since MettaAgent is now the actual policy)
        actual_policy = policy

        try:
            # Use torch.package to save the policy with all dependencies
            with PackageExporter(path, debug=False) as exporter:
                # Extern metta.util.config first since it depends on pydantic
                exporter.extern("metta.util.config")

                # Intern all metta modules to include them in the package
                exporter.intern("metta.**")

                # Check if the policy comes from __main__ (common in scripts/notebooks)
                if actual_policy.__class__.__module__ == "__main__":
                    # Get the source code of the class
                    import inspect

                    try:
                        source = inspect.getsource(actual_policy.__class__)
                        # Prepend necessary imports to the source
                        full_source = "import torch\nimport torch.nn as nn\n\n" + source
                        # Save it as a module source
                        exporter.save_source_string("__main__", full_source)
                    except:
                        # If we can't get source, just extern it and hope for the best
                        exporter.extern("__main__")

                # External modules that should use the system version
                exporter.extern("torch")
                exporter.extern("torch.**")
                exporter.extern("numpy")
                exporter.extern("numpy.**")
                exporter.extern("scipy")
                exporter.extern("scipy.**")
                exporter.extern("sklearn")
                exporter.extern("sklearn.**")
                exporter.extern("matplotlib")
                exporter.extern("matplotlib.**")
                exporter.extern("gymnasium")
                exporter.extern("gymnasium.**")
                exporter.extern("gym")
                exporter.extern("gym.**")
                exporter.extern("tensordict")
                exporter.extern("tensordict.**")
                exporter.extern("einops")
                exporter.extern("einops.**")
                exporter.extern("hydra")
                exporter.extern("hydra.**")
                exporter.extern("omegaconf")
                exporter.extern("omegaconf.**")
                exporter.extern("torch_scatter")
                exporter.extern("torch_geometric")
                exporter.extern("torch_sparse")

                # Mock ALL wandb modules more comprehensively
                exporter.mock("wandb")
                exporter.mock("wandb.**")
                exporter.mock("wandb.*")
                exporter.mock("wandb.sdk")
                exporter.mock("wandb.sdk.**")
                exporter.mock("wandb.sdk.wandb_run")

                # Mock other modules
                exporter.mock("pufferlib")
                exporter.mock("pufferlib.**")
                exporter.mock("pydantic")
                exporter.mock("pydantic.**")
                exporter.mock("typing_extensions")
                exporter.mock("boto3")
                exporter.mock("boto3.**")
                exporter.mock("botocore")
                exporter.mock("botocore.**")
                exporter.mock("duckdb")
                exporter.mock("duckdb.**")
                exporter.mock("pandas")
                exporter.mock("pandas.**")
                exporter.mock("seaborn")
                exporter.mock("plotly")

                # Handle C extension modules
                exporter.extern("mettagrid.mettagrid_c")
                exporter.extern("mettagrid")
                exporter.extern("mettagrid.**")

                # Create a clean copy of metadata without wandb references
                clean_metadata = self._clean_metadata_for_packaging(self.metadata)

                # Save a minimal PolicyRecord with clean metadata
                minimal_pr = PolicyRecord(None, self.name, self.uri, clean_metadata)
                exporter.save_pickle("policy_record", "data.pkl", minimal_pr)

                # Save the actual policy
                exporter.save_pickle("policy", "model.pkl", actual_policy)

            logger.info(f"Saved policy with torch.package to {path}")

        except Exception as e:
            logger.error(f"torch.package save failed: {e}")
            raise RuntimeError(f"Failed to save policy using torch.package: {e}") from e

        return self

    def load(self, path: str, device: str = "cpu") -> nn.Module:
        """Load a policy from a torch.package file."""
        logger.info(f"Loading policy from {path}")

        try:
            # Use torch.package to load the policy
            importer = PackageImporter(path)

            # First try to load the policy directly
            try:
                policy = importer.load_pickle("policy", "model.pkl", map_location=device)
                logger.info("Successfully loaded policy using torch.package")
                return policy

            except Exception as e:
                logger.warning(f"Could not load policy directly: {e}")
                # Fall back to loading from policy_record
                pr = importer.load_pickle("policy_record", "data.pkl", map_location=device)
                if hasattr(pr, "_policy") and pr._policy is not None:
                    return pr._policy
                else:
                    raise ValueError("PolicyRecord in package does not contain a policy")

        except Exception as e:
            # Not a torch.package file
            logger.info(f"Not a torch.package file ({e})")

            # We don't support non-torch.package files
            raise ValueError(
                f"Cannot load policy from {path}: This file is not a valid torch.package file. "
                "All policies must be saved using torch.package."
            )

    def key_and_version(self) -> tuple[str, int]:
        """
        Extract the policy key and version from the URI.

        Returns:
            tuple: (policy_key, version)
                - policy_key is the clean name without path or version
                - version is the numeric version or 0 if not present
        """
        # Get the last part after splitting by slash
        base_name = self.uri.split("/")[-1]

        # Check if it has a version number in format ":vNUM"
        if ":" in base_name and ":v" in base_name:
            parts = base_name.split(":v")
            key = parts[0]
            try:
                version = int(parts[1])
            except ValueError:
                version = 0
        else:
            # No version, use the whole thing as key and version = 0
            key = base_name
            version = 0

        return key, version

    def key(self) -> str:
        return self.key_and_version()[0]

    def version(self) -> int:
        return self.key_and_version()[1]


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
        """Load a policy from a PyTorch checkpoint."""
        name = os.path.basename(path)

        # Common metadata for pytorch loads
        default_metadata = {
            "action_names": [],
            "agent_step": 0,
            "epoch": 0,
            "generation": 0,
            "train_time": 0,
        }

        # Load PyTorch checkpoint using PytorchAgent wrapper
        policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._cfg.get("pytorch"))
        pr = PolicyRecord(self, name, "pytorch://" + name, default_metadata)
        pr._policy = policy
        return pr

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

            # Fallback for old checkpoints (pre-torch.package)
            if "PytorchStreamReader failed locating file .data/extern_modules" in str(e):
                logger.info("Detected old checkpoint format, loading as regular PyTorch checkpoint")
                return self._load_legacy_checkpoint(path, metadata_only)

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

    def _load_legacy_checkpoint(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a legacy checkpoint (pre-torch.package format)."""
        logger.info(f"Loading legacy checkpoint from {path}")

        name = os.path.basename(path)

        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        # Check if checkpoint is already a PolicyRecord
        if isinstance(checkpoint, PolicyRecord):
            logger.info("Checkpoint contains a pickled PolicyRecord")
            pr = checkpoint
            pr._policy_store = self
            pr._local_path = path

            # The policy might still be None, so we need to ensure it's loaded
            if pr._policy is None and not metadata_only:
                raise ValueError("Legacy PolicyRecord has no policy attached")

            self._cached_prs[path] = pr
            return pr

        # Otherwise, assume it's a state dict or checkpoint dict
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Extract metadata from checkpoint if available
        metadata = {
            "action_names": checkpoint.get("action_names", []),
            "agent_step": checkpoint.get("agent_step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "generation": checkpoint.get("generation", 0),
            "train_time": checkpoint.get("train_time", 0),
        }

        # Create PolicyRecord
        pr = PolicyRecord(self, name, f"file://{path}", metadata)
        pr._local_path = path

        if not metadata_only:
            # Extract model configuration from checkpoint if available
            obs_shape = checkpoint.get("obs_shape", [34, 11, 11])  # Default shape
            action_space_nvec = checkpoint.get("action_space_nvec", [9, 10])  # Default action space

            # Create observation and action spaces
            obs_space = gym.spaces.Dict(
                {
                    "grid_obs": gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                    "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
                }
            )

            action_space = gym.spaces.MultiDiscrete(action_space_nvec)

            # Create MettaAgent
            try:
                policy = hydra.utils.instantiate(
                    self._cfg.agent,
                    obs_space=obs_space,
                    obs_width=obs_shape[1],
                    obs_height=obs_shape[2],
                    action_space=action_space,
                    feature_normalizations=checkpoint.get("feature_normalizations", {}),
                    device=self._device,
                    _target_="metta.agent.metta_agent.MettaAgent",
                    _recursive_=False,
                )

                # Load the state dict
                if "model_state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["state_dict"])
                else:
                    # Try to load the checkpoint directly as state dict
                    policy.load_state_dict(checkpoint)

                pr._policy = policy
                logger.info("Successfully loaded legacy checkpoint as MettaAgent")

            except Exception as e:
                logger.error(f"Failed to create MettaAgent from legacy checkpoint: {e}")
                raise ValueError(f"Cannot load legacy checkpoint as MettaAgent: {e}")

        self._cached_prs[path] = pr
        return pr
