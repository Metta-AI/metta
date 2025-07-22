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
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import wandb
from omegaconf import DictConfig

from metta.agent.metta_agent import make_policy
from metta.agent.policy_cache import PolicyCache
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.common.wandb.wandb_context import WandbRun
from metta.rl.policy import load_pytorch_policy
from metta.rl.trainer_config import TrainerConfig, create_trainer_config

logger = logging.getLogger("policy_store")


class PolicySelectorConfig:
    """Simple config class for policy selection without pydantic dependency."""

    def __init__(self, type: str = "top", metric: str = "score"):
        self.type = type
        self.metric = metric


class PolicyStore:
    def __init__(self, cfg: DictConfig, wandb_run: WandbRun | None):
        self._cfg = cfg
        self._device = cfg.device
        self._wandb_run: WandbRun | None = wandb_run
        cache_size = cfg.get("policy_cache_size", 10)  # Default to 10 if not specified
        self._cached_prs = PolicyCache(max_size=cache_size)
        self._made_codebase_backwards_compatible = False

    def policy_record(
        self, uri_or_config: Union[str, DictConfig], selector_type: str = "top", metric="score"
    ) -> PolicyRecord:
        prs = self.policy_records(uri_or_config, selector_type, 1, metric)
        assert len(prs) == 1, f"Expected 1 policy record, got {len(prs)} policy records!"
        return prs[0]

    def policy_records(
        self, uri_or_config: Union[str, DictConfig], selector_type: str = "top", n=1, metric="score"
    ) -> List[PolicyRecord]:
        uri = uri_or_config if isinstance(uri_or_config, str) else uri_or_config.uri
        return self._select_policy_records(uri, selector_type, n, metric)

    def _select_policy_records(
        self, uri: str, selector_type: str = "top", n: int = 1, metric: str = "score"
    ) -> List[PolicyRecord]:
        """
        Select policy records based on URI and selection criteria.

        Args:
            uri: Resource identifier (wandb://, file://, pytorch://, or path)
            selector_type: Selection strategy ('all', 'latest', 'rand', 'top')
            n: Number of policy records to select (for 'top' selector)
            metric: Metric to use for 'top' selection

        Returns:
            List of selected PolicyRecord objects
        """
        # Load policy records from URI
        prs = self._load_policy_records_from_uri(uri)

        if not prs:
            raise ValueError(f"No policy records found at {uri}")

        logger.info(f"Found {len(prs)} policy records at {uri}")

        # Apply selector
        if selector_type == "all":
            logger.info(f"Returning all {len(prs)} policy records")
            return prs

        elif selector_type == "latest":
            logger.info(f"Selected latest policy: {prs[0].run_name}")
            return [prs[0]]

        elif selector_type == "rand":
            selected = random.choice(prs)
            logger.info(f"Selected random policy: {selected.run_name}")
            return [selected]

        elif selector_type == "top":
            return self._select_top_prs_by_metric(prs, n, metric)

        else:
            raise ValueError(f"Invalid selector type: {selector_type}")

    def _load_policy_records_from_uri(self, uri: str) -> List[PolicyRecord]:
        """Load policy records from various URI schemes."""
        if uri.startswith("wandb://"):
            wandb_uri = uri[8:]
            version = None

            if ":" in wandb_uri:
                wandb_uri, version = wandb_uri.split(":", 1)

            if wandb_uri.startswith("run/"):
                return self._prs_from_wandb_run(wandb_uri[4:], version)
            elif wandb_uri.startswith("sweep/"):
                return self._prs_from_wandb_sweep(wandb_uri[6:], version)
            else:
                return self._prs_from_wandb_artifact(wandb_uri, version)

        elif uri.startswith("file://"):
            return self._prs_from_path(uri[7:])

        elif uri.startswith("pytorch://"):
            return self._prs_from_pytorch(uri[10:])

        else:
            return self._prs_from_path(uri)

    def _select_top_prs_by_metric(self, prs: List[PolicyRecord], n: int, metric: str) -> List[PolicyRecord]:
        """Select top N policy records based on metric score."""
        # Extract scores
        policy_scores = self._get_pr_scores(prs, metric)

        # Filter policy records with valid scores
        valid_policies = [(p, score) for p, score in policy_scores.items() if score is not None]

        if not valid_policies:
            logger.warning(f"No valid scores found for metric '{metric}', returning latest policy")
            return [prs[0]]

        # Check if we have enough valid scores (80% threshold)
        if len(valid_policies) < len(prs) * 0.8:
            logger.warning("Too many invalid scores (>20%), returning latest policy")
            return [prs[0]]

        # Sort by score (highest first) and take top n
        sorted_policies = sorted(valid_policies, key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in sorted_policies[:n]]

        # Log results
        if len(selected) < n:
            logger.warning(f"Only found {len(selected)} policy records matching criteria, requested {n}")

        logger.info(f"Top {len(selected)} policy records by {metric}:")
        logger.info(f"{'Policy':<40} | {metric:<20}")
        logger.info("-" * 62)

        for policy in selected:
            score = policy_scores[policy]
            logger.info(f"{policy.run_name:<40} | {score:<20.4f}")

        return selected

    def _get_pr_scores(self, prs: List[PolicyRecord], metric: str) -> dict[PolicyRecord, Optional[float]]:
        """Extract metric scores from policy metadata."""
        if not prs:
            return {}

        # Check where the metric is stored in the first policy
        sample = prs[0]

        # Check in eval_scores first
        sample_eval_scores = sample.metadata.get("eval_scores", {})
        candidate_metric_keys = [metric]
        if metric.endswith("_score"):
            candidate_metric_keys.append(metric[:-6])

        for candidate_metric_key in candidate_metric_keys:
            if candidate_metric_key in sample_eval_scores:
                logger.info(f"Found metric '{candidate_metric_key}' in metadata['eval_scores']")
                return {p: p.metadata.get("eval_scores", {}).get(candidate_metric_key) for p in prs}

        # Check directly in metadata
        if metric in sample.metadata:
            logger.info(f"Found metric '{metric}' directly in metadata")
            return {p: p.metadata.get(metric) for p in prs}

        # Metric not found
        else:
            logger.warning(f"Metric '{metric}' not found in policy metadata")
            return {p: None for p in prs}

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create_empty_policy_record(self, name: str, override_path: str | None = None) -> PolicyRecord:
        if "trainer" not in self._cfg:
            raise AttributeError("New policies can't be created by a PolicyStore with no 'cfg.trainer' attribute.")

        trainer_cfg: TrainerConfig = create_trainer_config(self._cfg)

        path = override_path if override_path is not None else os.path.join(trainer_cfg.checkpoint.checkpoint_dir, name)
        metadata = PolicyMetadata()
        return PolicyRecord(self, name, f"file://{path}", metadata)

    def save(self, pr: PolicyRecord, path: str | None = None) -> PolicyRecord:
        """Save a policy record using the simple torch.save approach with atomic file operations."""
        if path is None:
            if hasattr(pr, "file_path"):
                path = pr.file_path
            else:
                path = pr.uri[7:] if pr.uri.startswith("file://") else pr.uri

        logger.info(f"Saving policy to {path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save to a temporary file first to ensure atomic writes
        temp_path = path + ".tmp"

        # Temporarily remove the policy store reference to avoid pickling issues
        pr._policy_store = None
        try:
            torch.save(pr, temp_path)
            # Atomically replace the file (works even if target exists)
            # os.replace is atomic on POSIX systems and handles existing files
            os.replace(temp_path, path)
        finally:
            pr._policy_store = self
            # Clean up temp file if it still exists (in case of error)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        # Don't cache the policy that we just saved,
        # since it might be updated later. We always
        # load the policy from the file when needed.
        pr._cached_policy = None
        self._cached_prs.put(path, pr)
        return pr

    def add_to_wandb_run(self, run_id: str, pr: PolicyRecord, additional_files: list[str] | None = None) -> str:
        return self.add_to_wandb_artifact(run_id, "model", pr.metadata, pr.file_path, additional_files)

    def add_to_wandb_sweep(self, sweep_name: str, pr: PolicyRecord, additional_files: list[str] | None = None) -> str:
        return self.add_to_wandb_artifact(sweep_name, "sweep_model", pr.metadata, pr.file_path, additional_files)

    def add_to_wandb_artifact(
        self, name: str, type: str, metadata: dict[str, Any], file_path: str, additional_files: list[str] | None = None
    ) -> str:
        if self._wandb_run is None:
            raise ValueError("PolicyStore was not initialized with a wandb run")

        additional_files = additional_files or []

        artifact = wandb.Artifact(name, type=type, metadata=metadata)
        artifact.add_file(file_path, name="model.pt")
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
            PolicyRecord(
                self, run_name=a.name, uri="wandb://" + a.qualified_name, metadata=PolicyMetadata.from_dict(a.metadata)
            )
            for a in artifacts
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

    def _load_from_pytorch(self, path: str) -> PolicyRecord:
        name = os.path.basename(path)
        # PolicyMetadata only requires: agent_step, epoch, generation, train_time
        # action_names is optional and not used by pytorch:// checkpoints
        metadata = PolicyMetadata()
        pr = PolicyRecord(self, name, "pytorch://" + name, metadata)
        pr._cached_policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._cfg.get("pytorch"))
        return pr

    def _make_codebase_backwards_compatible(self):
        """
        torch.load expects the codebase to be in the same structure as when the model was saved.
        We can use this function to alias old layout structures. For now we are supporting:
        - agent --> metta.agent
        """
        # Memoize
        if self._made_codebase_backwards_compatible:
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

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from a file using simple torch.load."""
        cached_pr = self._cached_prs.get(path)
        if cached_pr is not None:
            if metadata_only or cached_pr._cached_policy is not None:
                return cached_pr

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])

        logger.info(f"Loading policy from {path}")

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        # Make codebase backwards compatible before loading
        self._make_codebase_backwards_compatible()

        # Load checkpoint - could be PolicyRecord or legacy format
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        if isinstance(checkpoint, PolicyRecord):
            # New format - PolicyRecord object
            pr = checkpoint
            pr._policy_store = self

            # Ensure _cached_policy attribute exists
            if not hasattr(pr, "_cached_policy"):
                pr._cached_policy = None

            # Check if this is a legacy PolicyRecord with metadata under old names
            if not hasattr(pr, "_metadata"):
                # Access metadata property to trigger backwards compatibility
                try:
                    _ = pr.metadata  # This will convert old attributes to new format
                    logger.info("Converted legacy PolicyRecord metadata to new format")
                except AttributeError:
                    logger.warning("PolicyRecord has no metadata - creating default metadata")
                    pr._metadata = PolicyMetadata()

            # Also check for policy under old attribute names
            if not metadata_only and pr._cached_policy is None:
                policy_cache_attributes = ["_cached_policy", "_policy", "policy_cache"]
                for attr in policy_cache_attributes:
                    if hasattr(pr, attr):
                        policy = getattr(pr, attr)
                        if policy is not None:
                            pr._cached_policy = policy
                            if attr != "_cached_policy":
                                logger.info(f"Found policy under legacy attribute '{attr}'")
                            break

            self._cached_prs.put(path, pr)

            if metadata_only:
                pr._cached_policy = None

            return pr

        # Legacy format - try to load as old checkpoint
        return self._load_legacy_checkpoint(path, checkpoint, metadata_only)

    def _load_legacy_checkpoint(self, path: str, checkpoint: Any, metadata_only: bool = False) -> PolicyRecord:
        """Load a legacy checkpoint format (dict or old PolicyRecord)."""
        logger.info(f"Loading legacy checkpoint from {path}")

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Create PolicyRecord with metadata from checkpoint
        metadata_dict = {
            k: checkpoint.get(k, 0 if k != "action_names" else [])
            for k in ["action_names", "agent_step", "epoch", "generation", "train_time"]
        }

        pr = PolicyRecord(self, os.path.basename(path), f"file://{path}", PolicyMetadata(**metadata_dict))

        if not metadata_only:
            try:
                # Create mock environment for policy creation
                obs_shape = checkpoint.get("obs_shape", [34, 11, 11])
                env = SimpleNamespace(
                    single_observation_space=gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                    obs_width=obs_shape[1],
                    obs_height=obs_shape[2],
                    single_action_space=gym.spaces.MultiDiscrete(checkpoint.get("action_space_nvec", [9, 10])),
                    feature_normalizations=checkpoint.get("feature_normalizations", {}),
                    global_features=[],
                )

                policy = make_policy(env, self._cfg)  # type: ignore

                # Load state dict from checkpoint
                state_key = next((k for k in ["model_state_dict", "state_dict"] if k in checkpoint), None)
                if state_key:
                    policy.load_state_dict(checkpoint[state_key])
                else:
                    # If no state dict key found, assume the checkpoint itself is the state dict
                    policy.load_state_dict(checkpoint)

                pr._cached_policy = policy
                logger.info("Successfully loaded legacy checkpoint as MettaAgent")
            except Exception as e:
                raise ValueError(f"Cannot load legacy checkpoint as MettaAgent: {e}") from e

        self._cached_prs.put(path, pr)
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
