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

from metta.agent.agent_config import AgentConfig
from metta.agent.policy_loader import EmptyPolicyInitializer, PolicyLoader
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config import Config
from metta.common.wandb.wandb_context import WandbConfig, WandbRun
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import CheckpointFileType
from metta.sim.utils import get_pr_scores_from_stats_server

logger = logging.getLogger("policy_store")


PolicySelectorType = Literal["all", "top", "latest", "rand"]


class PolicySelectorConfig(Config):
    type: PolicySelectorType = "top"
    metric: str = "score"


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
        self._device = device or "cpu"
        self._data_dir = data_dir or "./train_dir"
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._pytorch_cfg = pytorch_cfg
        self._wandb_run: WandbRun | None = wandb_run
        self._policy_loader = PolicyLoader(
            device=device,
            data_dir=data_dir,
            pytorch_cfg=pytorch_cfg,
            policy_cache_size=policy_cache_size,
        )
        self._initialize_empty_policy: EmptyPolicyInitializer | None = None

    @property
    def initialize_empty_policy(self) -> EmptyPolicyInitializer | None:
        return self._initialize_empty_policy

    @initialize_empty_policy.setter
    def initialize_empty_policy(self, value: EmptyPolicyInitializer | None) -> None:
        self._initialize_empty_policy = value
        if self._policy_loader is not None:
            self._policy_loader.initialize_empty_policy = value

    def policy_record(
        self,
        uri_or_config: str | DictConfig,
        selector_type: PolicySelectorType = "top",
        metric: str = "score",
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
    ) -> PolicyRecord:
        prs = self.policy_records(uri_or_config, selector_type, 1, metric, stats_client, eval_name)
        assert len(prs) == 1, f"Expected 1 policy record, got {len(prs)} policy records!"
        return prs[0]

    def policy_records(
        self,
        uri_or_config: str | DictConfig,
        selector_type: PolicySelectorType = "top",
        n: int = 1,
        metric: str = "score",
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
    ) -> list[PolicyRecord]:
        uri = uri_or_config if isinstance(uri_or_config, str) else uri_or_config.uri
        return self._select_policy_records(uri, selector_type, n, metric, stats_client, eval_name)

    def _select_policy_records(
        self,
        uri: str,
        selector_type: PolicySelectorType = "top",
        n: int = 1,
        metric: str = "score",
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
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
        prs = self._load_policy_records_from_uri(uri)

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
            return self._select_top_prs_by_metric(prs, n, metric, stats_client, eval_name)

        else:
            raise ValueError(f"Invalid selector type: {selector_type}")

    def _prs_from_wandb(self, uri: str) -> list[PolicyRecord]:
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

    def _load_policy_records_from_uri(self, uri: str) -> list[PolicyRecord]:
        if uri.startswith("wandb://"):
            return self._prs_from_wandb(uri)

        elif uri.startswith("file://"):
            return self._prs_from_path(uri[len("file://") :])

        elif uri.startswith("pytorch://"):
            return self._prs_from_pytorch(uri[len("pytorch://") :])

        else:
            return self._prs_from_path(uri)

    def _select_top_prs_by_metric(
        self,
        prs: list[PolicyRecord],
        n: int,
        metric: str,
        stats_client: StatsClient | None = None,
        eval_name: str | None = None,
    ) -> list[PolicyRecord]:
        """Select top N policy records based on metric score."""
        # Extract scores
        policy_scores = self._get_pr_scores(prs, metric)
        if eval_name and stats_client and any([s is None for s in policy_scores.values()]):
            # Because an eval_name is provided, assume that the metric is reward
            policy_scores.update(get_pr_scores_from_stats_server(stats_client, prs, eval_name, metric="reward"))

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

    def make_model_name(self, epoch: int, model_suffix: str) -> str:
        return f"model_{epoch:04d}{model_suffix}"

    def create_policy_record(
        self, name: str, checkpoint_dir: str, metadata: PolicyMetadata | dict | None = None, policy: Any | None = None
    ) -> PolicyRecord:
        pr = self._policy_loader.create_empty_policy_record(name, checkpoint_dir)
        if metadata is not None:
            pr.metadata = metadata
        if policy is not None:
            pr.policy = policy
        return pr

    def save_to_pt_file(self, pr: PolicyRecord, path: str | None) -> str:
        return self._policy_loader.save_to_pt_file(pr, path)

    def save_to_safetensors_file(self, pr: PolicyRecord, path: str | None) -> str:
        return self._policy_loader.save_to_safetensors_file(pr, path)

    def save(
        self, pr: PolicyRecord, checkpoint_file_type: CheckpointFileType = "pt", path: str | None = None
    ) -> PolicyRecord:
        # if saving both, take path from safetensors
        if checkpoint_file_type in ["safetensors", "pt_also_emit_safetensors"]:
            path = self.save_to_safetensors_file(pr, path)
        if checkpoint_file_type in ["pt", "pt_also_emit_safetensors"]:
            path = self.save_to_pt_file(pr, path)

        if path is not None:
            self._policy_loader._cached_prs.put(path, pr)
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

    def _prs_from_path(self, path: str) -> list[PolicyRecord]:
        paths = []

        if path.endswith(".pt"):
            paths.append(path)
        elif path.endswith(".safetensors"):
            paths.append(path)
        else:
            checkpoint_files = [p for p in os.listdir(path) if p.endswith(".pt") or p.endswith(".safetensors")]
            checkpoint_files.sort(
                key=lambda f: int(os.path.splitext(f)[0][6:])
                if f.startswith("model_") and os.path.splitext(f)[0][6:].isdigit()
                else -1,
                reverse=True,
            )
            paths.extend([os.path.join(path, p) for p in checkpoint_files])

        return list([self._policy_loader.load_from_file(path, metadata_only=True) for path in paths])

    def _prs_from_wandb_artifact(self, uri: str, version: str | None = None) -> list[PolicyRecord]:
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
            PolicyRecord(
                self, run_name=a.name, uri="wandb://" + a.qualified_name, metadata=PolicyMetadata.from_dict(a.metadata)
            )
            for a in artifacts
        ]

    def _prs_from_pytorch(self, path: str) -> list[PolicyRecord]:
        return [self._policy_loader._load_from_pytorch(path)]

    def load_from_uri(self, uri: str) -> PolicyRecord:
        if uri.startswith("wandb://"):
            return self._policy_loader._load_wandb_artifact(uri[len("wandb://") :])
        if uri.startswith("file://"):
            file_path = uri[len("file://") :]
            return self._policy_loader.load_from_file(file_path)
        if uri.startswith("pytorch://"):
            return self._policy_loader._load_from_pytorch(uri[len("pytorch://") :])
        if "://" not in uri:
            return self._policy_loader.load_from_file(uri)

        raise ValueError(f"Invalid URI: {uri}")

    @classmethod
    def create(
        cls,
        device: str,
        data_dir: str,
        wandb_config: WandbConfig,
        wandb_run: WandbRun | None = None,
        system_cfg: SystemConfig | None = None,
        agent_cfg: AgentConfig | None = None,
        env_cfg: EnvConfig | None = None,
    ) -> "PolicyStore":
        """Create a PolicyStore from a WandbConfig.

        Args:
            device: Device to load policies on (e.g., "cpu", "cuda")
            wandb_config: WandbConfig object containing entity and project info
            replay_dir: Directory for storing policy artifacts
            wandb_run: Optional existing wandb run

        Returns:
            Configured PolicyStore instance
        """
        return cls(
            device=device,
            wandb_run=wandb_run,
            data_dir=data_dir,
            wandb_entity=wandb_config.entity if wandb_config.enabled else None,
            wandb_project=wandb_config.project if wandb_config.enabled else None,
        )

    def policy_record_or_mock(
        self,
        policy_uri: str | None,
        run_name: str = "mock_run",
    ) -> PolicyRecord:
        """Get a policy record or create a mock if no URI provided.

        Args:
            policy_uri: Optional policy URI to load
            run_name: Name for the mock run if no URI provided

        Returns:
            PolicyRecord from URI or MockPolicyRecord
        """
        if policy_uri is not None:
            return self.policy_record(policy_uri)
        else:
            # Import here to avoid circular dependency
            from metta.agent.mocks import MockPolicyRecord

            return MockPolicyRecord(run_name=run_name, uri=None)
