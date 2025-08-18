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
from typing import Any, Literal

import wandb
from omegaconf import DictConfig

from metta.agent.policy_loader import PolicyLoader
from metta.agent.policy_record import PolicyRecord
from metta.common.wandb.wandb_context import WandbRun
from metta.rl.trainer_config import CheckpointFileType

logger = logging.getLogger("policy_store")


PolicySelectorType = Literal["all", "top", "latest", "rand"]


class PolicySelectorConfig:
    """Simple config class for policy selection without pydantic dependency."""

    def __init__(self, type: PolicySelectorType = "top", metric: str = "score"):
        self.type = type
        self.metric = metric


class PolicyMissingError(ValueError):
    pass


class PolicyStore:
    def __init__(
        self,
        device: str | None = None,  # for loading policies from checkpoints
        wandb_run: WandbRun | None = None,  # for saving artifacts to wandb
        data_dir: str | None = None,  # for storing policy artifacts locally for cached access
        wandb_entity: str | None = None,  # for loading policies from wandb
        wandb_project: str | None = None,  # for loading policies from wandb
        pytorch_cfg: DictConfig | None = None,  # for loading pytorch policies
        policy_cache_size: int = 10,  # num policies to keep in memory
    ) -> None:
        self._wandb_run: WandbRun | None = wandb_run

        # Initialize the policy loader
        self._policy_loader = PolicyLoader(
            device=device or "cpu",
            data_dir=data_dir or "./train_dir",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            pytorch_cfg=pytorch_cfg,
            policy_cache_size=policy_cache_size,
            agent_factory=None,
        )

    def policy_record(
        self,
        uri_or_config: str | DictConfig,
        selector_type: PolicySelectorType = "top",
        metric: str = "score",
    ) -> PolicyRecord:
        prs = self.policy_records(uri_or_config, selector_type, 1, metric)
        assert len(prs) == 1, f"Expected 1 policy record, got {len(prs)} policy records!"
        return prs[0]

    def policy_records(
        self,
        uri_or_config: str | DictConfig,
        selector_type: PolicySelectorType = "top",
        n: int = 1,
        metric: str = "score",
    ) -> list[PolicyRecord]:
        uri = uri_or_config if isinstance(uri_or_config, str) else uri_or_config.uri
        return self._select_policy_records(uri, selector_type, n, metric)

    def _select_policy_records(
        self, uri: str, selector_type: PolicySelectorType = "top", n: int = 1, metric: str = "score"
    ) -> list[PolicyRecord]:
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
        prs = self._policy_loader._load_policy_records_from_uri(uri)

        if not prs:
            raise PolicyMissingError(f"No policy records found at {uri}")

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

    def _select_top_prs_by_metric(self, prs: list[PolicyRecord], n: int, metric: str) -> list[PolicyRecord]:
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

    def _get_pr_scores(self, prs: list[PolicyRecord], metric: str) -> dict[PolicyRecord, float | None]:
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

    # ?? move this
    @classmethod
    def make_model_name(cls, epoch: int, model_suffix: str) -> str:
        return f"model_{epoch:04d}{model_suffix}"

    # ?? move this
    @classmethod
    def make_model_path(cls, checkpoint_dir: str, name: str) -> str:
        return os.path.join(checkpoint_dir, name)

    # ?? rename to save_policy_record
    # ?? does checkpoint_file_type need to be here - isn't it part of the pr?
    def save(
        self, pr: PolicyRecord, checkpoint_file_type: CheckpointFileType = "pt", path: str | None = None
    ) -> PolicyRecord:
        # Save the policy record (includes caching)
        self._policy_loader.save_policy(pr, checkpoint_file_type, path)
        return pr

    # ?? i don't think wandb needs to be in PolicyStore
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

    def load_from_uri(self, uri: str) -> PolicyRecord:
        """Load a single policy record from URI using the policy loader."""
        pr = self._policy_loader.load_from_uri(uri)
        return pr
