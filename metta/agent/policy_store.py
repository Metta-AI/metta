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

import hydra
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
            reconstruction_attributes["cfg"] = OmegaConf.to_container(self._cfg.agent, resolve=True)

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
        inner_policy = policy.policy if isinstance(policy, MettaAgent) else policy

        if isinstance(inner_policy, BrainPolicy):
            # Trace the model for revisioning
            agent_attrs = inner_policy.agent_attributes
            obs_shape = agent_attrs["obs_shape"]
            dummy_obs = torch.zeros(1, *obs_shape, device=self._device)
            dummy_lstm_h = torch.zeros(
                agent_attrs["core_num_layers"], 1, agent_attrs["hidden_size"], device=self._device
            )
            dummy_lstm_c = torch.zeros(
                agent_attrs["core_num_layers"], 1, agent_attrs["hidden_size"], device=self._device
            )

            traced_policy = torch.jit.trace(inner_policy, (dummy_obs, dummy_lstm_h, dummy_lstm_c))
            torch.jit.save(traced_policy, path)
            logger.info(f"Saved JIT traced policy to {path}")
        else:
            # Fallback for other policy types
            torch.save(
                {
                    "state_dict": inner_policy.state_dict(),
                    "metadata": metadata,
                    "class_name": inner_policy.__class__.__name__,
                    "reconstruction_attributes": metadata.get("reconstruction_attributes", {}),
                },
                path,
            )

        pr = PolicyRecord(self, path, "file://" + path, metadata)
        pr._policy_store = self
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

        # HACK: we have no way of knowing the metadata without loading the file
        # so we just load them all.
        records = []
        for p in paths:
            try:
                records.append(self._load_from_file(p, metadata_only=True))
            except Exception as e:
                logger.warning(f"Could not load policy record from {p}: {e}")

        return records

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
            pr = self._cached_prs[path]
            if metadata_only or pr._policy is not None:
                return pr

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])
        logger.info(f"Loading policy from {path}")

        self._make_codebase_backwards_compatible()
        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        try:
            # Try loading as a JIT-traced model first
            policy = torch.jit.load(path, map_location=self._device)
            # We can't get metadata from a JIT file, so we have to load it from a sidecar file if it exists
            metadata_path = path + ".meta.pt"
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path)["metadata"]
            else:
                metadata = {}

            name = os.path.basename(path)
            pr = PolicyRecord(self, name, "file://" + path, metadata)
            if not metadata_only:
                pr._policy = MettaAgent(policy)

            self._cached_prs[path] = pr
            return pr
        except (torch.jit.frontend.NotSupportedError, RuntimeError):
            logger.info(f"Failed to load {path} as a JIT model, falling back to legacy state_dict loading.")

        # Fallback to legacy loading
        checkpoint = torch.load(path, map_location=self._device)

        if isinstance(checkpoint, PolicyRecord):
            logger.info("Loading legacy PolicyRecord object.")
            pr = checkpoint
            pr._policy_store = self
            pr._local_path = path
            self._cached_prs[path] = pr
            if metadata_only:
                pr._policy = None
            return pr

        metadata = checkpoint.get("metadata", {})
        name = os.path.basename(path)
        pr = PolicyRecord(self, name, "file://" + path, metadata)
        pr._local_path = path

        if not metadata_only:
            class_name = checkpoint.get("class_name", "BrainPolicy")
            if class_name == "BrainPolicy":
                recon_attrs = checkpoint["reconstruction_attributes"]
                # Use hydra to build the policy from config
                cfg = OmegaConf.create(recon_attrs.pop("cfg"))
                brain = hydra.utils.instantiate(cfg, **recon_attrs)
                brain.load_state_dict(checkpoint["state_dict"])
                policy = MettaAgent(brain)
                pr._policy = policy
            else:
                raise NotImplementedError(f"Legacy loading for {class_name} not implemented")

        self._cached_prs[path] = pr
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
