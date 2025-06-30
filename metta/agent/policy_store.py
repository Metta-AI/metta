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
from typing import Any, List, Optional, Union

import torch
import wandb
from omegaconf import DictConfig, ListConfig

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.rl.policy import load_pytorch_policy
from metta.rl.trainer_config import TrainerConfig, parse_trainer_config

logger = logging.getLogger("policy_store")


class PolicySelectorConfig:
    """Simple config class for policy selection without pydantic dependency."""

    def __init__(self, type: str = "top", metric: str = "score"):
        self.type = type
        self.metric = metric


class PolicyStore:
    def __init__(self, cfg: ListConfig | DictConfig, wandb_run):
        self._cfg = cfg
        self._trainer_cfg: TrainerConfig | None = parse_trainer_config(cfg) if "trainer" in cfg else None
        self._device = cfg.device
        self._wandb_run = wandb_run
        self._cached_prs = {}

    def policy_record(
        self, uri_or_config: Union[str, ListConfig | DictConfig], selector_type: str = "top", metric="score"
    ) -> PolicyRecord:
        prs = self.policy_records(uri_or_config, selector_type, 1, metric)
        assert len(prs) == 1, f"Expected 1 policy record, got {len(prs)} policy records!"
        return prs[0]

    def policy_records(
        self, uri_or_config: Union[str, ListConfig | DictConfig], selector_type: str = "top", n=1, metric="score"
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
            logger.info(f"Selected latest policy: {prs[0].name}")
            return [prs[0]]

        elif selector_type == "rand":
            selected = random.choice(prs)
            logger.info(f"Selected random policy: {selected.name}")
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
            logger.info(f"{policy.name:<40} | {score:<20.4f}")

        return selected

    def _get_pr_scores(self, prs: List[PolicyRecord], metric: str) -> dict[PolicyRecord, Optional[float]]:
        """Extract metric scores from policy metadata."""
        if not prs:
            return {}

        # Check where the metric is stored in the first policy
        sample = prs[0]

        # Check in eval_scores first
        if (
            "eval_scores" in sample.metadata
            and sample.metadata["eval_scores"] is not None
            and metric in sample.metadata["eval_scores"]
        ):
            logger.info(f"Found metric '{metric}' in metadata['eval_scores']")
            return {p: p.metadata.get("eval_scores", {}).get(metric) for p in prs}

        # Check directly in metadata
        elif metric in sample.metadata:
            logger.info(f"Found metric '{metric}' directly in metadata")
            return {p: p.metadata.get(metric) for p in prs}

        # Metric not found
        else:
            logger.warning(f"Metric '{metric}' not found in policy metadata")
            return {p: None for p in prs}

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create_empty_policy_record(self, name: str, override_path: str | None = None) -> PolicyRecord:
        path = (
            override_path
            if override_path is not None
            else os.path.join(self._trainer_cfg.checkpoint.checkpoint_dir, name)
        )
        metadata = PolicyMetadata()
        return PolicyRecord(self, name, f"file://{path}", metadata)

    def save(self, pr: PolicyRecord, path: str | None = None) -> PolicyRecord:
        """Save a policy record using the simple torch.save approach."""
        if path is None:
            if hasattr(pr, "file_path"):
                path = pr.file_path
            else:
                path = pr.uri[7:] if pr.uri.startswith("file://") else pr.uri

        logger.info(f"Saving policy to {path}")

        # Temporarily remove the policy store reference to avoid pickling issues
        pr._policy_store = None
        torch.save(pr, path)
        pr._policy_store = self

        # Don't cache the policy that we just saved,
        # since it might be updated later. We always
        # load the policy from the file when needed.
        pr._cached_policy = None
        self._cached_prs[path] = pr
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

    def _load_from_pytorch(self, path: str) -> PolicyRecord:
        name = os.path.basename(path)
        # PolicyMetadata only requires: agent_step, epoch, generation, train_time
        # action_names is optional and not used by pytorch:// checkpoints
        metadata = PolicyMetadata()
        pr = PolicyRecord(self, name, "pytorch://" + name, metadata)
        pr._cached_policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._cfg.get("pytorch"))
        return pr

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from a file using simple torch.load."""
        if path in self._cached_prs:
            cached_pr = self._cached_prs[path]
            if metadata_only or cached_pr._cached_policy is not None:
                return cached_pr

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])

        logger.info(f"Loading policy from {path}")

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        # Simple torch.load approach - expects a PolicyRecord object
        pr = torch.load(path, map_location=self._device, weights_only=False)

        if not isinstance(pr, PolicyRecord):
            raise ValueError(
                f"Expected PolicyRecord object in {path}, got {type(pr).__name__}. "
                "This codebase only supports checkpoints saved with the current format."
            )

        pr._policy_store = self
        self._cached_prs[path] = pr

        if metadata_only:
            pr._cached_policy = None

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
