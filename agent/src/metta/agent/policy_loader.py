"""
This file implements a PolicyLoader class that handles one-file operations for loading and saving policy records.
"""

import collections
import logging
import os
import sys
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import torch
import wandb
from omegaconf import DictConfig

from metta.agent import policy_metadata_yaml_helper
from metta.agent.policy_cache import PolicyCache
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.rl.puffer_policy import load_pytorch_policy

logger = logging.getLogger("policy_loader")


class EmptyPolicyInitializer(ABC):
    """Abstract class for initializing empty policies."""

    @abstractmethod
    def initialize_empty_policy(
        self, policy_record: "PolicyRecord", base_path: str, checkpoint_name: str
    ) -> "PolicyRecord":
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
    ) -> None:
        self._device = device or "cpu"
        self._data_dir = data_dir or "./train_dir"
        self._pytorch_cfg = pytorch_cfg
        self._cached_prs = PolicyCache(max_size=policy_cache_size)
        self._made_codebase_backwards_compatible = False
        self.initialize_empty_policy: EmptyPolicyInitializer | None = None

    def load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from a file, automatically detecting format based on extension."""
        if path.endswith(".pt"):
            return self._load_from_pt_file(path, metadata_only)
        elif path.endswith(".safetensors"):
            return self._load_from_safetensors_file(path, metadata_only)
        else:
            raise ValueError(f"Unsupported file format: {path}. Expected .pt or .safetensors")

    def _load_from_pt_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from a file using simple torch.load."""
        cached_pr = self._cached_prs.get(path)
        if cached_pr is not None:
            if metadata_only or cached_pr.cached_policy is not None:
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

        if metadata_only:
            pr.set_policy_deferred(lambda: self._load_from_pt_file(path, metadata_only=False).policy)

        return pr

    def _load_from_safetensors_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from safetensors format with YAML metadata sidecar."""
        cached_pr = self._cached_prs.get(path)
        if cached_pr is not None:
            if metadata_only or cached_pr.cached_policy is not None:
                return cached_pr

        pr = self.create_empty_policy_record(self.checkpoint_name(path), self.base_path(path))
        if self.initialize_empty_policy is None:
            raise ValueError("initialize_empty_policy must be set to load from safetensors format")
        pr = self.initialize_empty_policy.initialize_empty_policy(pr, self.base_path(path), self.checkpoint_name(path))

        self._cached_prs.put(path, pr)

        if metadata_only:
            pr.set_policy_deferred(lambda: self._load_from_safetensors_file(path, metadata_only=False).policy)

        return pr

    def _load_from_pytorch(self, path: str) -> PolicyRecord:
        name = os.path.basename(path)
        # PolicyMetadata only requires: agent_step, epoch, generation, train_time
        # action_names is optional and not used by pytorch:// checkpoints
        metadata = PolicyMetadata()
        pr = PolicyRecord(self, name, "pytorch://" + name, metadata)
        pr.cached_policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._pytorch_cfg)
        return pr

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

    # note that you should set policy after this
    def create_empty_policy_record(self, name: str, checkpoint_dir: str) -> PolicyRecord:
        path = os.path.join(checkpoint_dir, name)
        metadata = PolicyMetadata()
        return PolicyRecord(
            name,
            f"file://{path}",
            metadata,
        )
