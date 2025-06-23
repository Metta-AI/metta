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

import gymnasium as gym
import hydra
import numpy as np
import torch
import wandb
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
        policy = self.policy()
        if type(policy).__name__ not in {"MettaAgent", "DistributedMettaAgent", "PytorchAgent"}:
            raise TypeError(f"Expected MettaAgent, DistributedMettaAgent, or PytorchAgent, got {type(policy).__name__}")
        return policy

    def num_params(self) -> int:
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self) -> Optional[str]:
        return self._local_path

    def __repr__(self):
        """Generate a detailed representation of the PolicyRecord with weight shapes.."""
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
        policy = None
        if self._policy is None:
            try:
                policy = self.policy()
            except Exception as e:
                lines.append(f"Error loading policy: {str(e)}")
                return "\n".join(lines)
        else:
            policy = self._policy

        # Add total parameter count
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Add module structure with detailed weight shapes
        lines.append("\nModule Structure with Weight Shapes:")

        for name, module in policy.named_modules():
            # Skip top-level module
            if name == "":
                continue

            # Create indentation based on module hierarchy
            indent = "  " * name.count(".")

            # Get module type
            module_type = module.__class__.__name__

            # Start building the module info line
            module_info = f"{indent}{name}: {module_type}"

            # Get parameters for this module (non-recursive)
            params = list(module.named_parameters(recurse=False))

            # Add detailed parameter information
            if params:
                # For common layer types, add specialized shape information
                if isinstance(module, torch.nn.Conv2d):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        out_channels, in_channels, kernel_h, kernel_w = weight.shape
                        module_info += " ["
                        module_info += f"out_channels={out_channels}, "
                        module_info += f"in_channels={in_channels}, "
                        module_info += f"kernel=({kernel_h}, {kernel_w})"
                        module_info += "]"

                elif isinstance(module, torch.nn.Linear):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        out_features, in_features = weight.shape
                        module_info += f" [in_features={in_features}, out_features={out_features}]"

                elif isinstance(module, torch.nn.LSTM):
                    module_info += " ["
                    module_info += f"input_size={module.input_size}, "
                    module_info += f"hidden_size={module.hidden_size}, "
                    module_info += f"num_layers={module.num_layers}"
                    module_info += "]"

                elif isinstance(module, torch.nn.Embedding):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        num_embeddings, embedding_dim = weight.shape
                        module_info += f" [num_embeddings={num_embeddings}, embedding_dim={embedding_dim}]"

                # Add all parameter shapes
                param_shapes = []
                for param_name, param in params:
                    param_shapes.append(f"{param_name}={list(param.shape)}")

                if param_shapes and not any(
                    x in module_info for x in ["out_channels", "in_features", "hidden_size", "num_embeddings"]
                ):
                    module_info += f" ({', '.join(param_shapes)})"

            # Add formatted module info to output
            lines.append(module_info)

        # Add section for buffer shapes (non-parameter tensors like running_mean in BatchNorm)
        buffers = list(policy.named_buffers())
        if buffers:
            lines.append("\nBuffer Shapes:")
            for name, buffer in buffers:
                lines.append(f"  {name}: {list(buffer.shape)}")
        return "\n".join(lines)

    def _clean_metadata_for_packaging(self, metadata: dict) -> dict:
        import copy

        def clean_value(v):
            if hasattr(v, "__module__") and v.__module__ and "wandb" in v.__module__:
                return None
            elif isinstance(v, dict):
                return {k: clean_value(val) for k, val in v.items() if clean_value(val) is not None}
            elif isinstance(v, list):
                return [clean_value(item) for item in v if clean_value(item) is not None]
            elif isinstance(v, (str, int, float, bool, type(None))):
                return v
            elif hasattr(v, "__dict__"):
                try:
                    return str(v)
                except Exception:
                    return None
            else:
                return v

        return clean_value(copy.deepcopy(metadata))

    def save(self, path: str, policy: nn.Module) -> "PolicyRecord":
        logger.info(f"Saving policy to {path} using torch.package")
        self._local_path = path
        self.uri = "file://" + path

        try:
            with PackageExporter(path, debug=False) as exporter:
                # Extern metta.util.config first since it depends on pydantic
                exporter.extern("metta.util.config")

                # Intern all metta modules to include them in the package
                exporter.intern("metta.**")

                if policy.__class__.__module__ == "__main__":
                    import inspect

                    try:
                        source = inspect.getsource(policy.__class__)
                        exporter.save_source_string("__main__", f"import torch\nimport torch.nn as nn\n\n{source}")
                    except Exception:
                        exporter.extern("__main__")

                for module in [
                    "torch",
                    "numpy",
                    "scipy",
                    "sklearn",
                    "matplotlib",
                    "gymnasium",
                    "gym",
                    "tensordict",
                    "einops",
                    "hydra",
                    "omegaconf",
                ]:
                    exporter.extern(module)
                    exporter.extern(f"{module}.**")
                for module in ["torch_scatter", "torch_geometric", "torch_sparse"]:
                    exporter.extern(module)

                for pattern in ["wandb", "wandb.**", "wandb.*", "wandb.sdk", "wandb.sdk.**", "wandb.sdk.wandb_run"]:
                    exporter.mock(pattern)

                for module in ["pufferlib", "pydantic", "boto3", "botocore", "duckdb", "pandas"]:
                    exporter.mock(module)
                    exporter.mock(f"{module}.**")
                exporter.mock("typing_extensions")
                exporter.mock("seaborn")
                exporter.mock("plotly")

                exporter.extern("mettagrid.mettagrid_c")
                exporter.extern("mettagrid")
                exporter.extern("mettagrid.**")
                exporter.extern("sys")

                clean_metadata = self._clean_metadata_for_packaging(self.metadata)
                exporter.save_pickle(
                    "policy_record", "data.pkl", PolicyRecord(None, self.name, self.uri, clean_metadata)
                )
                exporter.save_pickle("policy", "model.pkl", policy)

        except Exception as e:
            logger.error(f"torch.package save failed: {e}")
            raise RuntimeError(f"Failed to save policy using torch.package: {e}") from e

        return self

    def load(self, path: str, device: str = "cpu") -> nn.Module:
        logger.info(f"Loading policy from {path}")
        try:
            importer = PackageImporter(path)
            try:
                return importer.load_pickle("policy", "model.pkl", map_location=device)
            except Exception as e:
                logger.warning(f"Could not load policy directly: {e}")
                pr = importer.load_pickle("policy_record", "data.pkl", map_location=device)
                if hasattr(pr, "_policy") and pr._policy is not None:
                    return pr._policy
                raise ValueError("PolicyRecord in package does not contain a policy") from e
        except Exception as e:
            logger.info(f"Not a torch.package file ({e})")
            raise ValueError(f"Cannot load policy from {path}: This file is not a valid torch.package file.") from e

    def key_and_version(self) -> tuple[str, int]:
        """
        Extract the policy key and version from the URI.
        TODO: store these on the metadata for new policies,
        eventually read them from the metadata instead of parsing the URI.
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
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

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
        pr.save(path, policy)
        pr._policy = policy
        return pr

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict):
        return PolicyRecord(self, name, "file://" + path, metadata).save(path, policy)

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
            paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".pt")]
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

        name = os.path.basename(path)
        pr = PolicyRecord(
            self,
            name,
            f"file://{path}",
            {
                "action_names": checkpoint.get("action_names", []),
                "agent_step": checkpoint.get("agent_step", 0),
                "epoch": checkpoint.get("epoch", 0),
                "generation": checkpoint.get("generation", 0),
                "train_time": checkpoint.get("train_time", 0),
            },
        )
        pr._local_path = path

        if not metadata_only:
            obs_shape = checkpoint.get("obs_shape", [34, 11, 11])
            obs_space = gym.spaces.Dict(
                {
                    "grid_obs": gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                    "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
                }
            )

            try:
                policy = hydra.utils.instantiate(
                    self._cfg.agent,
                    obs_space=obs_space,
                    obs_width=obs_shape[1],
                    obs_height=obs_shape[2],
                    action_space=gym.spaces.MultiDiscrete(checkpoint.get("action_space_nvec", [9, 10])),
                    feature_normalizations=checkpoint.get("feature_normalizations", {}),
                    device=self._device,
                    _target_="metta.agent.metta_agent.MettaAgent",
                    _recursive_=False,
                )

                if "model_state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["state_dict"])
                else:
                    policy.load_state_dict(checkpoint)

                pr._policy = policy
                logger.info("Successfully loaded legacy checkpoint as MettaAgent")
            except Exception as e:
                logger.error(f"Failed to create MettaAgent from legacy checkpoint: {e}")
                raise ValueError(f"Cannot load legacy checkpoint as MettaAgent: {e}") from e

        self._cached_prs[path] = pr
        return pr
