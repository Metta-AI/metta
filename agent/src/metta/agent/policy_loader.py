"""
This file implements a PolicyLoader class that handles one-file operations for loading and saving policy records.
"""

import collections
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict
from urllib.parse import urlparse

import torch
import wandb
from omegaconf import DictConfig
from safetensors.torch import load_file

from metta.agent import policy_metadata_yaml_helper
from metta.agent.policy_cache import PolicyCache
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyAgent, PolicyRecord
from metta.rl.puffer_policy import load_pytorch_policy

if TYPE_CHECKING:
    from metta.common.wandb.wandb_context import WandbRun
    from metta.rl.system_config import SystemConfig

logger = logging.getLogger("policy_loader")


def make_codebase_backwards_compatible():
    """
    torch.load expects the codebase to be in the same structure as when the model was saved.
    We can use this function to alias old layout structures. For now we are supporting:
    - agent --> metta.agent
    """
    # Use a module-level flag to memoize
    if hasattr(make_codebase_backwards_compatible, "_made_compatible"):
        return
    make_codebase_backwards_compatible._made_compatible = False

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


class AgentBuilder(ABC):
    """Abstract class for building policy agents."""

    @abstractmethod
    def initialize_agent(self, policy_metadata: PolicyMetadata, weights: Dict[str, torch.Tensor] | None) -> PolicyAgent:
        """Initialize an empty policy with the given parameters."""
        pass


class PolicyLoader:
    """Handles one-file operations for loading and saving policy records."""

    def __init__(
        self,
        device: str | None = None,
        data_dir: str | None = None,
        pytorch_cfg: DictConfig | None = None,
        policy_cache_size: int = 10,
        wandb_run: "WandbRun | None" = None,
    ) -> None:
        self._device = device or "cpu"
        self._data_dir = data_dir or "./train_dir"
        self._pytorch_cfg = pytorch_cfg
        self._cached_prs = PolicyCache(max_size=policy_cache_size)
        self._wandb_run = wandb_run
        self.agent_builder: AgentBuilder | None = None

    @classmethod
    # ?? todo: clean up pytorch_cfg here - then remove SystemConfig arg
    def create(
        cls,
        device: str,
        data_dir: str,
        system_cfg: "SystemConfig | None" = None,
        wandb_run: "WandbRun | None" = None,
        agent_builder: "AgentBuilder | None" = None,
    ) -> "PolicyLoader":
        """Create a PolicyLoader with the specified configuration.

        Args:
            device: Device to load policies on (e.g., "cpu", "cuda")
            data_dir: Directory for storing policy artifacts
            system_cfg: Optional system configuration
            wandb_run: Optional wandb run for uploading artifacts
            agent_builder: Optional agent builder for creating new agents

        Returns:
            Configured PolicyLoader instance
        """
        loader = cls(
            device=device,
            data_dir=data_dir,
            pytorch_cfg=getattr(system_cfg, "pytorch", None) if system_cfg else None,
            wandb_run=wandb_run,
        )
        if agent_builder is not None:
            loader.agent_builder = agent_builder
        return loader

    def load_from_uri(self, uri: str) -> PolicyRecord:
        """Load a PolicyHandle from various URI types.

        Args:
            uri: URI to load from (file://, wandb://, pytorch://, or direct path)

        Returns:
            PolicyHandle with appropriate factory function
        """
        if uri.startswith("wandb://"):
            return self._load_from_wandb_uri(uri)
        elif uri.startswith("file://"):
            return self.load_from_file(uri[len("file://") :])
        elif uri.startswith("pytorch://"):
            return self._load_from_pytorch_uri(uri)
        else:
            return self.load_from_file(uri)

    def load_from_file(self, path: str) -> PolicyRecord:
        """Load a PolicyRecord from a file, automatically detecting format based on extension."""
        if path.endswith(".pt"):
            return self._load_from_pt_file(path)
        elif path.endswith(".safetensors"):
            return self._load_from_safetensors_file(path)
        else:
            raise ValueError(f"Unsupported file format: {path}. Expected .pt or .safetensors")

    def _load_from_pt_file(self, path: str) -> PolicyRecord:
        """Load a PolicyRecord from a file using simple torch.load."""
        cached_pr = self._cached_prs.get(path)
        if cached_pr is not None:
            return cached_pr

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])

        logger.info(f"Loading policy from {path}")

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        # Make codebase backwards compatible before loading
        self._make_codebase_backwards_compatible()

        # Load checkpoint - could be PolicyRecord or legacy format
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        if not isinstance(checkpoint, PolicyRecord):
            raise Exception("Invalid checkpoint, possibly in a legacy format")

        # New format - PolicyRecord object
        pr = checkpoint
        self._cached_prs.put(path, pr)

        return pr

    def _load_from_safetensors_file(self, path: str) -> PolicyRecord:
        """Load a PolicyRecord from safetensors format with YAML metadata sidecar."""
        cached_pr = self._cached_prs.get(path)
        if cached_pr is not None:
            return cached_pr

        if self.agent_builder is None:
            raise ValueError("agent_builder must be set to load from safetensors format")

        weights = load_file(path)
        metadata = policy_metadata_yaml_helper.get_metadata(self.checkpoint_name(path), self.base_path(path))
        policy = self.agent_builder.initialize_agent(metadata, weights)

        pr = PolicyRecord(self.checkpoint_name(path), f"file://{path}", metadata, policy)

        self._cached_prs.put(path, pr)

        return pr

    def _load_from_pytorch_uri(self, path: str) -> PolicyRecord:
        name = os.path.basename(path)
        # PolicyMetadata only requires: agent_step, epoch, generation, train_time
        # action_names is optional and not used by pytorch:// checkpoints
        metadata = PolicyMetadata()
        pr = PolicyRecord(
            name, "pytorch://" + name, metadata, load_pytorch_policy(path, self._device, pytorch_cfg=self._pytorch_cfg)
        )
        return pr

    def _load_from_wandb_uri(self, uri: str) -> PolicyRecord:
        """Load a PolicyRecord from a wandb URI.

        Args:
            uri: Wandb URI to load from (e.g., wandb://entity/project/artifact_type/name)

        Returns:
            PolicyRecord loaded from wandb

        Raises:
            NotImplementedError: Wandb loading is not implemented yet
        """
        raise NotImplementedError("wandb not implemented yet")

    def _load_wandb_artifact(self, qualified_name: str):
        logger.info(f"Loading policy from wandb artifact {qualified_name}")

        artifact = wandb.Api().artifact(qualified_name)

        artifact_path = os.path.join(self._data_dir, "artifacts", artifact.name)

        if not os.path.exists(artifact_path):
            artifact.download(root=artifact_path)

        logger.info(f"Downloaded artifact {artifact.name} to {artifact_path}")

        pr = self.load_from_file(os.path.join(artifact_path, "model.pt"))
        pr.metadata.update(artifact.metadata)
        return pr

    def save_to_pt_file(self, pr: PolicyRecord, path: str | None) -> str:
        """Save a policy record using the simple torch.save approach with atomic file operations."""
        if path is None:
            if hasattr(pr, "file_path"):
                path = pr.file_path
            elif pr.uri is not None:
                path = pr.uri[7:] if pr.uri.startswith("file://") else pr.uri
            else:
                raise ValueError("PolicyRecord has no file_path or uri")

        logger.info(f"Saving policy to {path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save to a temporary file first to ensure atomic writes
        temp_path = path + ".tmp"

        try:
            torch.save(pr, temp_path)
            # Atomically replace the file (works even if target exists)
            # os.replace is atomic on POSIX systems and handles existing files
            os.replace(temp_path, path)
        finally:
            # Clean up temp file if it still exists (in case of error)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        return path

    def save_to_safetensors_file(self, pr: PolicyRecord, path: str | None) -> str:
        """Save a policy record using safetensors format with YAML metadata sidecar."""
        if path is None:
            if hasattr(pr, "file_path"):
                path = pr.file_path
            elif pr.uri is not None:
                path = pr.uri[7:] if pr.uri.startswith("file://") else pr.uri
            else:
                raise ValueError("PolicyRecord has no file_path or uri")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get checkpoint directory and name
        checkpoint_dir = os.path.dirname(path)
        checkpoint_name = os.path.splitext(os.path.basename(path))[0]

        # Save .safetensors file (just the model weights/state dict)
        safetensors_path = policy_metadata_yaml_helper.save_policy(pr, checkpoint_name, checkpoint_dir)
        return safetensors_path

    def _make_codebase_backwards_compatible(self):
        """Call the module-level function to make codebase backwards compatible."""
        make_codebase_backwards_compatible()

    def checkpoint_name(self, url: str) -> str:
        path = urlparse(url).path  # "/path/to/file.txt"
        filename = os.path.basename(path)  # "file.txt"
        name, _ = os.path.splitext(filename)  # ("file", ".txt")
        return name

    def base_path(self, url: str) -> str:
        parsed = urlparse(url)
        # Remove the last segment from the path
        path = parsed.path.rsplit("/", 1)[0]
        return parsed._replace(path=path).geturl()

    def _load_from_pytorch(self, path: str) -> PolicyRecord:
        name = os.path.basename(path)
        # PolicyMetadata only requires: agent_step, epoch, generation, train_time
        # action_names is optional and not used by pytorch:// checkpoints
        metadata = PolicyMetadata()
        cached_policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._pytorch_cfg)
        pr = PolicyRecord(name, "pytorch://" + name, metadata, cached_policy)
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
