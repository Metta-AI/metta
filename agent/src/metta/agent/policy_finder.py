"""
This file implements a PolicyFinder class that finds policy records from various sources.

This class provides functionality to discover and load policy metadata from various URIs,
similar to PolicyStore but focused on finding policies rather than managing them.
"""

import logging
import os
import random
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import wandb
from omegaconf import DictConfig

from metta.agent.policy_handle import PolicyHandle
from metta.agent.policy_metadata import PolicyMetadata
from metta.app_backend.clients.stats_client import StatsClient
from metta.sim.utils import get_pr_scores_from_stats_server

if TYPE_CHECKING:
    from typing import Literal

    PolicySelectorType = Literal["all", "top", "latest", "rand"]

logger = logging.getLogger("policy_finder")


class PolicyMissingError(ValueError):
    pass


class PolicyFinder:
    """A class for finding policy records from various sources, always loading metadata only.

    This class provides functionality to discover and load policy metadata from various URIs,
    similar to PolicyStore but focused on finding policies rather than managing them.
    """

    def __init__(
        self,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ) -> None:
        """Initialize PolicyFinder with a PolicyLoader.

        Args:
            policy_loader: The PolicyLoader instance to use for loading policy records
            wandb_entity: Wandb entity for loading policies from wandb
            wandb_project: Wandb project for loading policies from wandb
        """
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project

    @classmethod
    def create(cls, wandb_entity: str | None = None, wandb_project: str | None = None) -> "PolicyFinder":
        """Create a PolicyFinder with default settings.

        Args:
            wandb_entity: Wandb entity for loading policies from wandb
            wandb_project: Wandb project for loading policies from wandb
            device: Device to load policies on (e.g., "cpu", "cuda")
            data_dir: Directory for storing policy artifacts
            pytorch_cfg: Optional pytorch configuration
            policy_cache_size: Number of policies to keep in memory

        Returns:
            Configured PolicyFinder instance
        """

        return cls(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )

    def policy_records(
        self,
        uri_or_config: str | DictConfig,
        selector_type: "PolicySelectorType" = "top",
        n: int = 1,
        metric: str = "score",
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
    ) -> list[PolicyHandle]:
        """Find and select policy handles based on URI and selection criteria.

        Args:
            uri_or_config: URI or config containing URI to load policies from
            selector_type: Selection strategy ('all', 'latest', 'rand', 'top')
            n: Number of policy handles to select (for 'top' selector)
            metric: Metric to use for 'top' selection
            stats_client: Optional stats client for fetching scores
            eval_name: Optional eval name for fetching scores from stats server

        Returns:
            List of selected PolicyHandle objects
        """
        uri = uri_or_config if isinstance(uri_or_config, str) else uri_or_config.uri
        return self._select_policy_handles(uri, selector_type, n, metric, stats_client, eval_name)

    def _select_policy_handles(
        self,
        uri: str,
        selector_type: "PolicySelectorType" = "top",
        n: int = 1,
        metric: str = "score",
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
    ) -> list[PolicyHandle]:
        """
        Select policy handles based on URI and selection criteria.

        Args:
            uri: Resource identifier (wandb://, file://, pytorch://, or path)
            selector_type: Selection strategy ('all', 'latest', 'rand', 'top')
            n: Number of policy handles to select (for 'top' selector)
            metric: Metric to use for 'top' selection
            stats_client: Optional stats client for fetching scores
            eval_name: Optional eval name for fetching scores from stats server

        Returns:
            List of selected PolicyHandle objects
        """
        # Load policy handles from URI
        handles = self._load_from_uri(uri)

        if not handles:
            raise PolicyMissingError(f"No policy records found at {uri}")

        logger.info(f"Found {len(handles)} policy records at {uri}")

        # Apply selector
        if selector_type == "all":
            logger.info(f"Returning all {len(handles)} policy handles")
            return handles

        elif selector_type == "latest":
            logger.info(f"Selected latest policy: {handles[0].run_name}")
            return [handles[0]]

        elif selector_type == "rand":
            selected = random.choice(handles)
            logger.info(f"Selected random policy: {selected.run_name}")
            return [selected]

        elif selector_type == "top":
            return self._select_top_handles_by_metric(handles, n, metric, stats_client, eval_name)

        else:
            raise ValueError(f"Invalid selector type: {selector_type}")

    def _create_policy_handle(self, uri: str, run_name: str, metadata: PolicyMetadata | None) -> PolicyHandle:
        """Create a PolicyHandle from a PolicyRecord."""
        return PolicyHandle(
            uri=uri or "",
            factory=lambda pl: pl.load_from_uri(uri),
            run_name=run_name,
            metadata=metadata,
        )

    def _load_from_uri(self, uri: str) -> list[PolicyHandle]:
        """Load policy handles from various URI types."""
        if not uri:
            return []

        parsed = urlparse(uri)
        scheme = parsed.scheme

        if scheme == "wandb":
            return self._prs_from_wandb(uri)
        if scheme == "file":
            return self._prs_from_path(parsed.path)
        if scheme == "pytorch":
            return self._prs_from_pytorch(parsed.path)

        # Treat as direct filesystem path by default
        return self._prs_from_path(uri)

    def _prs_from_wandb(self, uri: str) -> list[PolicyHandle]:
        """
        Supported formats:
        - wandb://run/<run_name>[:<version>]
        - wandb://sweep/<sweep_name>[:<version>]
        - wandb://<entity>/<project>/<artifact_type>/<name>[:<version>]
        """
        wandb_uri = uri[len("wandb://") :]
        version = None

        if ":" in wandb_uri:
            wandb_uri, version = wandb_uri.split(":", 1)

        for prefix, artifact_type in [("run/", "model"), ("sweep/", "sweep_model")]:
            if wandb_uri.startswith(prefix):
                if not self._wandb_entity or not self._wandb_project:
                    raise ValueError("Wandb entity and project must be specified to use short policy uris")
                name = wandb_uri[len(prefix) :]
                return self._prs_from_wandb_artifact(
                    f"{self._wandb_entity}/{self._wandb_project}/{artifact_type}/{name}",
                    version,
                )
        else:
            return self._prs_from_wandb_artifact(wandb_uri, version)

    def _prs_from_wandb_artifact(self, uri: str, version: str | None = None) -> list[PolicyHandle]:
        """
        Expected uri format: <entity>/<project>/<artifact_type>/<name>
        """
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
            PolicyHandle(
                uri=uri,
                factory=lambda pl: pl.load_from_uri(uri),
                run_name=a.name,
                metadata=PolicyMetadata.from_dict(a.metadata),
            )
            for a in artifacts
        ]

    def _prs_from_path(self, path: str) -> list[PolicyHandle]:
        """Find policy records from a file path or directory."""
        paths = []

        if path.endswith(".pt"):
            paths.append(path)
        elif path.endswith(".safetensors"):
            paths.append(path)
        else:
            checkpoint_files = [p for p in os.listdir(path) if p.endswith(".pt") or p.endswith(".safetensors")]
            checkpoint_files.sort(
                key=lambda f: int(os.path.splitext(f)[0][6:])
                if f.startswith("model_") and f.endswith(".pt") and os.path.splitext(f)[0][6:].isdigit()
                else -1,
                reverse=True,
            )
            paths.extend([os.path.join(path, p) for p in checkpoint_files])

        # Flatten the list of lists into a single list of PolicyHandle objects
        all_handles = []
        for path in paths:
            handles = self._load_from_uri(path)
            all_handles.extend(handles)
        return all_handles

    def _prs_from_pytorch(self, path: str) -> list[PolicyHandle]:
        """Load policy record from pytorch path."""
        return [PolicyHandle(uri="file://" + path, factory=lambda pl: pl.load_from_file(path))]

    def _select_top_handles_by_metric(
        self,
        handles: list[PolicyHandle],
        n: int,
        metric: str,
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
    ) -> list[PolicyHandle]:
        """Select top N policy handles based on metric score."""
        # Extract scores from metadata
        handle_scores = self._get_handle_scores(handles, metric)

        # If needed, fetch missing scores from stats server (assume reward metric for eval lookups)
        if eval_name and stats_client:
            missing = [h for h, s in handle_scores.items() if s is None]
            if missing:
                pr_scores = get_pr_scores_from_stats_server(stats_client, handles, eval_name, metric="reward")
                for h in missing:
                    if h in pr_scores:
                        handle_scores[h] = pr_scores[h]

        # Keep only handles with valid scores
        scored: list[tuple[PolicyHandle, float]] = [(h, s) for h, s in handle_scores.items() if s is not None]  # type: ignore[misc]

        if not scored:
            logger.warning(f"No valid scores found for metric '{metric}', returning latest policy")
            return [handles[0]]

        # Require at least 80% of handles to have valid scores
        if len(scored) < len(handles) * 0.8:
            logger.warning("Too many invalid scores (>20%), returning latest policy")
            return [handles[0]]

        # Sort by score (descending) and select top n
        selected_pairs = sorted(scored, key=lambda pair: pair[1], reverse=True)[:n]
        selected = [h for h, _ in selected_pairs]

        if len(selected) < n:
            logger.warning(f"Only found {len(selected)} policy handles matching criteria, requested {n}")

        logger.info(f"Top {len(selected)} policy handles by {metric}:")
        logger.info(f"{'Policy':<40} | {metric:<20}")
        logger.info("-" * 62)
        for h, score in selected_pairs:
            logger.info(f"{h.run_name:<40} | {score:<20.4f}")

        return selected

    def _get_handle_scores(self, handles: list[PolicyHandle], metric: str) -> dict[PolicyHandle, float | None]:
        """Extract metric scores from policy handle metadata."""
        if not handles:
            return {}

        # Check where the metric is stored in the first policy handle
        sample = handles[0]
        if not sample.metadata:
            return {h: None for h in handles}

        # Check in eval_scores first
        sample_eval_scores = sample.metadata.get("eval_scores", {})
        candidate_metric_keys = [metric]
        if metric.endswith("_score"):
            candidate_metric_keys.append(metric[:-6])

        for candidate_metric_key in candidate_metric_keys:
            if candidate_metric_key in sample_eval_scores:
                logger.info(f"Found metric '{candidate_metric_key}' in metadata['eval_scores']")
                return {
                    h: h.metadata.get("eval_scores", {}).get(candidate_metric_key) if h.metadata else None
                    for h in handles
                }

        # Check directly in metadata
        if metric in sample.metadata:
            logger.info(f"Found metric '{metric}' directly in metadata")
            return {h: h.metadata.get(metric) if h.metadata else None for h in handles}

        # Metric not found
        else:
            logger.warning(f"Metric '{metric}' not found in policy metadata")
            return {h: None for h in handles}
