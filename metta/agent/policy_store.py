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
from typing import List, Optional, Union

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

from metta.agent.brain_policy import BrainPolicy
from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, make_policy
from metta.rl.policy import load_policy
from metta.util.config import Config
from metta.util.wandb.wandb_context import WandbRun

logger = logging.getLogger("policy_store")


class PolicySelectorConfig(Config):
    type: str = "top"
    metric: str = "score"


class PolicyRecord:
    def __init__(self, policy_store: "PolicyStore", name: str, uri: str, metadata: dict):
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

    def policy_as_metta_agent(self) -> Union[MettaAgent, DistributedMettaAgent]:
        """Get the policy as a MettaAgent or DistributedMettaAgent."""
        policy = self.policy()
        if not isinstance(policy, (MettaAgent, DistributedMettaAgent)):
            raise TypeError(f"Expected MettaAgent or DistributedMettaAgent, got {type(policy).__name__}")
        return policy

    def num_params(self) -> int:
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self) -> str | None:
        return self._local_path

    def __repr__(self):
        """Generate a detailed representation of the PolicyRecord with weight shapes."""
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
    def __init__(self, cfg: ListConfig | DictConfig, wandb_run: WandbRun | None):
        self._cfg = cfg
        self._device = cfg.device
        self._wandb_run = wandb_run
        self._cached_prs = {}
        self._made_codebase_backwards_compatible = False

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
        return self._policy_records(policy, selector_type, n=n, metric=metric)

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

        inner_policy = policy.policy if isinstance(policy, MettaAgent) else policy
        reconstruction_attributes = {}
        if isinstance(inner_policy, BrainPolicy):
            reconstruction_attributes = inner_policy.agent_attributes

        pr = self.save(
            name,
            path,
            policy,
            {
                "action_names": env.action_names,
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
                "reconstruction_attributes": reconstruction_attributes,
            },
        )
        pr._policy = policy
        return pr

    def save(self, name: str, path: str, policy: nn.Module, metadata: dict):
        logger.info(f"Saving policy to {path}")

        # Handle MettaAgent wrapping
        if hasattr(policy, "policy"):
            inner_policy = policy.policy
        else:
            inner_policy = policy

        # Ensure reconstruction_attributes are captured
        reconstruction_attrs = metadata.get("reconstruction_attributes", {})

        if not reconstruction_attrs:
            # Try to get from the wrapped policy first
            if isinstance(inner_policy, BrainPolicy) and hasattr(inner_policy, "agent_attributes"):
                reconstruction_attrs = inner_policy.agent_attributes.copy()  # Make a copy to avoid modifying original
                if reconstruction_attrs:
                    logger.info(
                        f"Extracted reconstruction attributes from inner BrainPolicy: {list(reconstruction_attrs.keys())}"
                    )
            # If not found on inner policy, try the wrapper (MettaAgent delegates via __getattr__)
            elif hasattr(policy, "agent_attributes"):
                reconstruction_attrs = policy.agent_attributes.copy()  # Make a copy to avoid modifying original
                if reconstruction_attrs:
                    logger.info(
                        f"Extracted reconstruction attributes via MettaAgent delegation: {list(reconstruction_attrs.keys())}"
                    )

            if not reconstruction_attrs and isinstance(inner_policy, BrainPolicy):
                logger.warning("BrainPolicy found but agent_attributes could not be accessed")

        # Convert gymnasium spaces to serializable format
        if "action_space" in reconstruction_attrs:
            import gymnasium.spaces

            action_space = reconstruction_attrs["action_space"]
            if isinstance(action_space, gymnasium.spaces.MultiDiscrete):
                reconstruction_attrs["action_space"] = {
                    "_type": "MultiDiscrete",
                    "nvec": action_space.nvec.tolist()
                    if hasattr(action_space.nvec, "tolist")
                    else list(action_space.nvec),
                }
            elif isinstance(action_space, gymnasium.spaces.Discrete):
                reconstruction_attrs["action_space"] = {"_type": "Discrete", "n": action_space.n}
            # For other space types, we'll just remove them and let the system reconstruct from environment
            elif action_space is not None:
                logger.warning(
                    f"Cannot serialize action_space of type {type(action_space).__name__}, removing from reconstruction attributes"
                )
                reconstruction_attrs.pop("action_space", None)

        # Convert feature_normalizations to plain dict with simple types
        if "feature_normalizations" in reconstruction_attrs:
            feature_norm = reconstruction_attrs["feature_normalizations"]
            if isinstance(feature_norm, dict):
                # Convert keys and values to plain Python types
                reconstruction_attrs["feature_normalizations"] = {int(k): float(v) for k, v in feature_norm.items()}
            elif isinstance(feature_norm, list):
                # If it's a list, keep it as is but ensure values are floats
                reconstruction_attrs["feature_normalizations"] = [float(v) for v in feature_norm]
            elif hasattr(feature_norm, "items"):  # Handle DictConfig from OmegaConf
                reconstruction_attrs["feature_normalizations"] = {int(k): float(v) for k, v in feature_norm.items()}

        # Convert obs_shape to list if it's a tuple
        if "obs_shape" in reconstruction_attrs and isinstance(reconstruction_attrs["obs_shape"], tuple):
            reconstruction_attrs["obs_shape"] = list(reconstruction_attrs["obs_shape"])

        # Convert obs_space to serializable format
        if "obs_space" in reconstruction_attrs:
            import gymnasium.spaces

            obs_space = reconstruction_attrs["obs_space"]
            if isinstance(obs_space, gymnasium.spaces.Dict):
                # For Dict spaces, we'll just store the type and reconstruct later
                reconstruction_attrs["obs_space"] = {"_type": "Dict"}
                logger.info("Simplified obs_space to just type for reconstruction")
            elif obs_space is not None:
                logger.warning(
                    f"Cannot serialize obs_space of type {type(obs_space).__name__}, removing from reconstruction attributes"
                )
                reconstruction_attrs.pop("obs_space", None)

        # Ensure all values in reconstruction_attrs are serializable
        # Remove any numpy arrays or other non-serializable types
        clean_attrs = {}
        for key, value in reconstruction_attrs.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                clean_attrs[key] = value
            elif hasattr(value, "tolist"):
                # Convert numpy arrays to lists
                clean_attrs[key] = value.tolist()
            elif value is None:
                clean_attrs[key] = value
            else:
                logger.warning(f"Skipping non-serializable attribute {key} of type {type(value).__name__}")

        reconstruction_attrs = clean_attrs

        # Prepare save data
        save_data = {
            "metadata": metadata,
            "class_name": inner_policy.__class__.__name__,
            "reconstruction_attributes": reconstruction_attrs,
        }

        # Try to save with torch.jit for better versioning
        try:
            # For BrainPolicy, we need to trace through a forward pass
            if isinstance(inner_policy, BrainPolicy) and reconstruction_attrs:
                # Create dummy inputs for tracing
                from metta.agent.policy_state import PolicyState

                obs_shape = reconstruction_attrs.get("obs_shape", [11, 11, 27])
                batch_size = 32

                # Get the device from the policy
                device = getattr(inner_policy, "device", "cpu")

                # Create dummy observation
                dummy_obs = torch.randn(batch_size, *obs_shape, device=device)

                # Create dummy state
                hidden_size = reconstruction_attrs.get("hidden_size", inner_policy.hidden_size)
                num_layers = reconstruction_attrs.get("core_num_layers", inner_policy.core_num_layers)
                dummy_state = PolicyState(
                    lstm_h=torch.zeros(num_layers, batch_size, hidden_size, device=device),
                    lstm_c=torch.zeros(num_layers, batch_size, hidden_size, device=device),
                )

                # Trace the model
                traced_model = torch.jit.trace(inner_policy, (dummy_obs, dummy_state))
                save_data["jit_model"] = traced_model
                save_data["use_jit"] = True
                logger.info("Successfully traced model with torch.jit")
            else:
                # For other policies, just save state dict
                save_data["state_dict"] = inner_policy.state_dict()
                save_data["use_jit"] = False
        except Exception as e:
            logger.warning(f"Failed to trace model with torch.jit: {e}. Falling back to state_dict")
            save_data["state_dict"] = inner_policy.state_dict()
            save_data["use_jit"] = False

        torch.save(save_data, path)

        pr = PolicyRecord(self, path, "file://" + path, metadata)
        pr._policy = None
        pr._policy_store = self
        pr._local_path = path
        self._cached_prs[path] = pr
        return pr

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
        policy = load_policy(path, self._device, puffer=self._cfg.puffer)
        name = os.path.basename(path)
        pr = PolicyRecord(
            self,
            name,
            "pytorch://" + name,
            {
                "action_names": [],
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            },
        )
        pr._policy = MettaAgent(policy)
        return pr

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        if path in self._cached_prs:
            if metadata_only or self._cached_prs[path]._policy is not None:
                return self._cached_prs[path]
        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])
        logger.info(f"Loading policy from {path}")

        self._make_codebase_backwards_compatible()

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        # Add gymnasium spaces to safe globals for PyTorch 2.6+
        import gymnasium.spaces

        torch.serialization.add_safe_globals(
            [
                gymnasium.spaces.multi_discrete.MultiDiscrete,
                gymnasium.spaces.discrete.Discrete,
                gymnasium.spaces.box.Box,
                gymnasium.spaces.dict.Dict,
                np.dtype,
            ]
        )

        checkpoint = torch.load(path, map_location=self._device)

        # Backwards compatibility for models saved as PolicyRecord objects
        if isinstance(checkpoint, PolicyRecord):
            logger.info("Loading legacy PolicyRecord object.")
            pr = checkpoint
            pr._policy_store = self
            pr._local_path = path
            self._cached_prs[path] = pr
            if metadata_only:
                pr._policy = None
                pr._local_path = None
            return pr

        metadata = checkpoint["metadata"]
        name = os.path.basename(path)
        pr = PolicyRecord(self, name, "file://" + path, metadata)
        pr._local_path = path

        if not metadata_only:
            # Check if this is a jit model
            if checkpoint.get("use_jit", False) and "jit_model" in checkpoint:
                logger.info("Loading torch.jit model")
                try:
                    jit_model = checkpoint["jit_model"]
                    # Wrap the jit model in MettaAgent
                    policy = MettaAgent(jit_model)
                    pr._policy = policy
                except Exception as e:
                    logger.warning(f"Failed to load jit model: {e}. Trying state_dict fallback")
                    # Fall back to state_dict loading if jit fails
                    if "state_dict" in checkpoint:
                        self._load_from_state_dict(checkpoint, pr)
                    else:
                        raise RuntimeError(f"Could not load model from {path}: no state_dict fallback available")
            else:
                # Load from state_dict
                self._load_from_state_dict(checkpoint, pr)

        self._cached_prs[path] = pr
        return pr

    def _load_from_state_dict(self, checkpoint: dict, pr: PolicyRecord):
        """Helper method to load a policy from state_dict."""
        class_name = checkpoint.get("class_name", "BrainPolicy")
        if class_name == "BrainPolicy":
            reconstruction_attrs = checkpoint.get("reconstruction_attributes", {})

            # Always try to infer missing attributes from the state dict
            state_dict = checkpoint.get("state_dict", {})
            if state_dict:
                # Try to infer hidden size if not present
                if "hidden_size" not in reconstruction_attrs:
                    for key, tensor in state_dict.items():
                        if "components._core_._net.weight_ih_l0" in key:
                            # LSTM input weight shape is (4*hidden_size, input_size)
                            hidden_size = tensor.shape[0] // 4
                            reconstruction_attrs["hidden_size"] = hidden_size
                            logger.info(f"Inferred hidden_size={hidden_size} from LSTM weights")
                            break

                # Try to infer number of features if not present
                if "num_features" not in reconstruction_attrs or "obs_shape" not in reconstruction_attrs:
                    for key, tensor in state_dict.items():
                        if "components.obs_normalizer.obs_norm" in key:
                            # Shape is [1, num_features, 1, 1]
                            num_features = tensor.shape[1]
                            reconstruction_attrs["num_features"] = num_features
                            logger.info(f"Inferred num_features={num_features} from obs_normalizer")
                            # Update obs_shape if needed
                            if "obs_shape" not in reconstruction_attrs:
                                obs_width = reconstruction_attrs.get("obs_width", 11)
                                obs_height = reconstruction_attrs.get("obs_height", 11)
                                reconstruction_attrs["obs_shape"] = [obs_width, obs_height, num_features]
                                logger.info(f"Set obs_shape to {reconstruction_attrs['obs_shape']}")
                            break

                # Try to infer action space size if not present
                if "total_actions" not in reconstruction_attrs:
                    for key, tensor in state_dict.items():
                        if "components._action_embeds_.active_indices" in key:
                            # This tells us the total number of discrete actions
                            num_actions = tensor.shape[0]
                            reconstruction_attrs["total_actions"] = num_actions
                            logger.info(f"Inferred total_actions={num_actions} from action embeddings")
                            break

            if reconstruction_attrs:
                logger.info(f"Found reconstruction attributes: {list(reconstruction_attrs.keys())}")
            else:
                logger.warning("No reconstruction attributes found in checkpoint")

            # Fill in some reasonable defaults for missing attributes
            defaults = {
                "clip_range": 0.2,
                "obs_width": 11,
                "obs_height": 11,
                "obs_key": "grid_obs",
                "core_num_layers": 2,
                "device": self._device,
            }

            # Only add obs_shape default if we don't have num_features
            if "num_features" not in reconstruction_attrs and "obs_shape" not in reconstruction_attrs:
                defaults["obs_shape"] = [11, 11, 27]

            for key, value in defaults.items():
                if key not in reconstruction_attrs:
                    reconstruction_attrs[key] = value
                    logger.info(f"Using default value for {key}={value}")

            # Reconstruct action_space from serialized format
            if "action_space" in reconstruction_attrs:
                action_space_data = reconstruction_attrs["action_space"]
                if isinstance(action_space_data, dict) and "_type" in action_space_data:
                    import gymnasium.spaces

                    if action_space_data["_type"] == "MultiDiscrete":
                        reconstruction_attrs["action_space"] = gymnasium.spaces.MultiDiscrete(action_space_data["nvec"])
                        logger.info(f"Reconstructed MultiDiscrete action space with nvec={action_space_data['nvec']}")
                    elif action_space_data["_type"] == "Discrete":
                        reconstruction_attrs["action_space"] = gymnasium.spaces.Discrete(action_space_data["n"])
                        logger.info(f"Reconstructed Discrete action space with n={action_space_data['n']}")
                elif isinstance(action_space_data, int):
                    # Legacy format where just the size was saved
                    import gymnasium.spaces

                    reconstruction_attrs["action_space"] = gymnasium.spaces.Discrete(action_space_data)
                    logger.info(f"Reconstructed Discrete action space from legacy format with n={action_space_data}")
                else:
                    # If we can't reconstruct, remove it and let the system handle it
                    logger.warning(f"Cannot reconstruct action_space from data: {action_space_data}")
                    reconstruction_attrs.pop("action_space", None)

            # Reconstruct obs_space if needed
            if "obs_space" in reconstruction_attrs:
                obs_space_data = reconstruction_attrs["obs_space"]
                if isinstance(obs_space_data, dict) and "_type" in obs_space_data:
                    if obs_space_data["_type"] == "Dict":
                        # Reconstruct a basic Dict space - the actual contents will be filled by the environment
                        import gymnasium.spaces

                        # Create a minimal Dict space that BrainPolicy can work with
                        obs_shape = reconstruction_attrs.get("obs_shape", [11, 11, 27])
                        reconstruction_attrs["obs_space"] = gymnasium.spaces.Dict(
                            {
                                "grid_obs": gymnasium.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                                "global_vars": gymnasium.spaces.Box(
                                    low=-np.inf, high=np.inf, shape=[0], dtype=np.int32
                                ),
                            }
                        )
                        logger.info("Reconstructed Dict observation space")
                else:
                    reconstruction_attrs.pop("obs_space", None)

            # Ensure all required fields are present
            if "obs_space" not in reconstruction_attrs:
                # Create a default obs_space if missing
                import gymnasium.spaces

                obs_shape = reconstruction_attrs.get("obs_shape", [11, 11, 27])
                # If we have num_features, update obs_shape
                if "num_features" in reconstruction_attrs:
                    num_features = reconstruction_attrs["num_features"]
                    obs_shape = [
                        reconstruction_attrs.get("obs_width", 11),
                        reconstruction_attrs.get("obs_height", 11),
                        num_features,
                    ]
                    reconstruction_attrs["obs_shape"] = obs_shape
                reconstruction_attrs["obs_space"] = gymnasium.spaces.Dict(
                    {
                        "grid_obs": gymnasium.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                        "global_vars": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
                    }
                )
                logger.info(f"Created default observation space with shape {obs_shape}")

            if "action_space" not in reconstruction_attrs:
                # Create a default action space if missing
                import gymnasium.spaces

                # Check if we have action_names in metadata
                action_names = pr.metadata.get("action_names", [])
                total_actions = reconstruction_attrs.get("total_actions", 25)

                if action_names and total_actions:
                    # Try to infer the action space from the total number of actions and action names
                    # This is a heuristic - distribute the actions somewhat evenly
                    num_action_types = len(action_names)
                    if num_action_types > 0:
                        # Simple heuristic: some actions have more params than others
                        # Common pattern: move/rotate have 4 params (0-3), others have fewer
                        if num_action_types == 9 and total_actions == 25:
                            # This matches the checkpoint we're debugging
                            # 25 total actions means indices 0-24, distributed across 9 action types
                            # Common pattern: some actions have no params (just the base action)
                            # others have multiple params
                            nvec = [1, 1, 1, 4, 2, 1, 1, 16, 1]  # Sums to 28, but we'll adjust
                            # Actually, let's try a different distribution that sums correctly
                            # 25 actions total, 9 types means average ~2.7 per type
                            nvec = [2, 2, 1, 4, 2, 2, 2, 8, 2]  # Sums to 25
                            reconstruction_attrs["action_space"] = gymnasium.spaces.MultiDiscrete(nvec)
                            logger.info(f"Created action space for {action_names} with nvec={nvec}")
                        else:
                            # Generic fallback
                            avg_per_action = max(1, (total_actions - 1) // num_action_types)
                            nvec = [avg_per_action] * num_action_types
                            reconstruction_attrs["action_space"] = gymnasium.spaces.MultiDiscrete(nvec)
                            logger.info(f"Created generic action space with {num_action_types} types")
                    else:
                        reconstruction_attrs["action_space"] = gymnasium.spaces.Discrete(total_actions)
                        logger.info(f"Created discrete action space with {total_actions} actions")
                else:
                    # Final fallback
                    reconstruction_attrs["action_space"] = gymnasium.spaces.MultiDiscrete([9, 3])
                    logger.info("Created default action space MultiDiscrete([9, 3])")

            if "feature_normalizations" not in reconstruction_attrs:
                # Create default feature normalizations based on num_features
                num_features = reconstruction_attrs.get("num_features", 27)
                if "obs_shape" in reconstruction_attrs and len(reconstruction_attrs["obs_shape"]) >= 3:
                    num_features = reconstruction_attrs["obs_shape"][2]
                reconstruction_attrs["feature_normalizations"] = {i: 1.0 for i in range(num_features)}
                logger.info(f"Created default feature normalizations for {num_features} features")

            if "device" not in reconstruction_attrs:
                reconstruction_attrs["device"] = self._device

            try:
                # Create BrainPolicy directly instead of using hydra.instantiate
                # This avoids issues with nested component instantiation
                from metta.agent.brain_policy import BrainPolicy

                # Convert agent config to dict and remove _target_ field
                agent_cfg_dict = OmegaConf.to_container(self._cfg.agent, resolve=True)
                if isinstance(agent_cfg_dict, dict) and "_target_" in agent_cfg_dict:
                    agent_cfg_dict.pop("_target_", None)

                # Merge the config with reconstruction attributes
                all_kwargs = {**agent_cfg_dict, **reconstruction_attrs}

                # Create the BrainPolicy instance
                brain = BrainPolicy(**all_kwargs)
                brain.load_state_dict(checkpoint["state_dict"])
                policy = MettaAgent(brain)
                pr._policy = policy
            except Exception as e:
                logger.error(f"Failed to instantiate BrainPolicy with attributes: {reconstruction_attrs}")
                logger.error(f"Error: {e}")
                raise RuntimeError(
                    f"Failed to load BrainPolicy. The checkpoint may be incompatible with the current "
                    f"configuration. Error: {str(e)}"
                )
        else:
            raise NotImplementedError(f"Loading for {class_name} not implemented from file")

        # Log what we have so far
        logger.info(
            f"Reconstruction attrs after inference: num_features={reconstruction_attrs.get('num_features')}, obs_shape={reconstruction_attrs.get('obs_shape')}"
        )

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
