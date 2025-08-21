"""
This file implements a PolicyLoader class that handles loading of policies from various URI formats.
It provides functionality to:
- Load policies from local files or remote URIs
- Support multiple file formats (.pt, .safetensors)
- Handle wandb artifacts and pytorch checkpoints
- Maintain backwards compatibility for torch.load

The PolicyLoader is used by PolicyStore and can be used independently for loading policies.
"""

import collections
import logging
import os
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import torch
import wandb
from omegaconf import DictConfig

from metta.agent import policy_metadata_yaml_helper
from metta.agent.policy_cache import PolicyCache
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.rl.model_architecture_serializer import save_model_architecture
from metta.rl.puffer_policy import load_pytorch_policy
from metta.rl.rng_state_config import load_and_restore_rng_state, save_current_rng_state
from metta.rl.trainer_config import CheckpointFileType

logger = logging.getLogger("policy_loader")


class PolicyLoader:
    """Handles loading of policies from various URI formats."""

    def __init__(
        self,
        device: str | None = None,
        data_dir: str | None = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        pytorch_cfg: DictConfig | None = None,
        policy_cache_size: int = 10,
        agent_factory: Callable[[str], PolicyRecord] | None = None,
    ) -> None:
        self._device = device or "cpu"
        self._data_dir = data_dir or "./train_dir"
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._pytorch_cfg = pytorch_cfg
        self._cached_prs = PolicyCache(max_size=policy_cache_size)
        self._made_codebase_backwards_compatible = False
        self.agent_factory = agent_factory

    def load_from_uri(self, uri: str) -> PolicyRecord:
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://") :])
        if uri.startswith("file://"):
            file_path = uri[len("file://") :]
            return self._load_from_file(file_path)
        if uri.startswith("pytorch://"):
            return self._load_from_pytorch(uri[len("pytorch://") :])
        if "://" not in uri:
            return self._load_from_file(uri)

        raise ValueError(f"Invalid URI: {uri}")

    def _load_policy_records_from_uri(self, uri: str) -> list[PolicyRecord]:
        if uri.startswith("wandb://"):
            return self._prs_from_wandb(uri)

        elif uri.startswith("file://"):
            return self._prs_from_path(uri[len("file://") :])

        elif uri.startswith("pytorch://"):
            return self._prs_from_pytorch(uri[len("pytorch://") :])

        else:
            return self._prs_from_path(uri)

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
                self,
                run_name=a.name,
                uri="wandb://" + a.qualified_name,
                metadata=PolicyMetadata.from_dict(a.metadata),
            )
            for a in artifacts
        ]

    def _prs_from_path(self, path: str) -> list[PolicyRecord]:
        paths = []

        if path.endswith(".pt"):
            paths.append(path)
        elif path.endswith(".safetensors"):
            paths.append(path)
        else:
            # Look for both .pt and .safetensors files in directory
            pt_files = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".pt")]
            safetensors_files = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".safetensors")]
            paths.extend(pt_files)
            paths.extend(safetensors_files)

        policy_records = []
        for path in paths:
            policy_records.append(self._load_from_file(path, metadata_only=True))

        return policy_records

    def _prs_from_pytorch(self, path: str) -> list[PolicyRecord]:
        return [self._load_from_pytorch(path)]

    def _load_from_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from a file, automatically detecting format based on extension."""
        if path.endswith(".pt"):
            return self._load_from_pt_file(path, metadata_only)
        elif path.endswith(".safetensors"):
            return self._load_from_safetensorsfile(path, metadata_only)
        else:
            raise ValueError(f"Unsupported file format: {path}. Expected .pt or .safetensors")

    def _load_from_pytorch(self, path: str) -> PolicyRecord:
        name = os.path.basename(path)
        # PolicyMetadata only requires: agent_step, epoch, generation, train_time
        # action_names is optional and not used by pytorch:// checkpoints
        metadata = PolicyMetadata()
        pr = PolicyRecord(self, name, "pytorch://" + name, metadata)
        pr.cached_policy = load_pytorch_policy(path, self._device, pytorch_cfg=self._pytorch_cfg)
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

    def _load_from_pt_file(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from a file using simple torch.load."""
        cached_pr = self._cached_prs.get(path) if self._cached_prs else None
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

        # Try to load RNG state if available
        rng_state_path = path.replace(".pt", ".rng")
        try:
            load_and_restore_rng_state(rng_state_path)
            logger.info(f"Restored RNG state from {rng_state_path}")
        except FileNotFoundError:
            logger.debug(f"No RNG state file found at {rng_state_path}")
        except Exception as e:
            logger.warning(f"Failed to load RNG state from {rng_state_path}: {e}")

        if self._cached_prs:
            self._cached_prs.put(path, pr)

        if metadata_only:
            pr.invalidate_cached_policy()

        return pr

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

    def _load_from_safetensorsfile(self, path: str, metadata_only: bool = False) -> PolicyRecord:
        """Load a PolicyRecord from safetensors format with YAML metadata sidecar."""

        # ?? problem with the way i did paths is that i normalized them when putting them in the cache, so i have to
        # ?? normalize them again here. not great
        path = str(Path(path))
        cached_pr = self._cached_prs.get(path) if self._cached_prs else None
        if cached_pr is not None:
            if metadata_only or cached_pr.cached_policy is not None:
                return cached_pr

        if self.agent_factory is None:
            raise ValueError("agent_factory must be set to load safetensors files")

        pr = self.agent_factory(path)
        policy_metadata_yaml_helper.restore_agent(pr.policy, self.checkpoint_name(path), Path(self.base_path(path)))

        # Try to load RNG state if available
        rng_state_path = path.replace(".safetensors", ".rng")
        try:
            load_and_restore_rng_state(rng_state_path)
            logger.info(f"Restored RNG state from {rng_state_path}")
        except FileNotFoundError:
            logger.debug(f"No RNG state file found at {rng_state_path}")
        except Exception as e:
            logger.warning(f"Failed to load RNG state from {rng_state_path}: {e}")

        if self._cached_prs:
            self._cached_prs.put(path, pr)
        return pr

    def _load_wandb_artifact(self, qualified_name: str):
        logger.info(f"Loading policy from wandb artifact {qualified_name}")

        artifact = wandb.Api().artifact(qualified_name)

        artifact_path = os.path.join(self._data_dir, "artifacts", artifact.name)

        if not os.path.exists(artifact_path):
            artifact.download(root=artifact_path)

        logger.info(f"Downloaded artifact {artifact.name} to {artifact_path}")

        pr = self._load_from_file(os.path.join(artifact_path, "model.pt"))
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

        # Save RNG state
        rng_state_path = path.replace(".pt", ".rng")
        try:
            save_current_rng_state(rng_state_path)
            save_model_architecture(path.replace(".pt", ".genericmodel"), pr.policy)
            logger.info(f"Saved RNG state to {rng_state_path}")
        except Exception as e:
            logger.warning(f"Failed to save RNG state to {rng_state_path}: {e}")

        # Temporarily remove the policy loader reference to avoid pickling issues
        pr._policy_loader = None  # type: ignore
        try:
            torch.save(pr, temp_path)
            # Atomically replace the file (works even if target exists)
            # os.replace is atomic on POSIX systems and handles existing files
            os.replace(temp_path, path)
        finally:
            pr._policy_loader = self
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
        safetensors_path = policy_metadata_yaml_helper.save_policy(pr, checkpoint_name, Path(checkpoint_dir))

        # Save RNG state
        rng_state_path = str(safetensors_path).replace(".safetensors", ".rng")
        try:
            save_current_rng_state(rng_state_path)
            logger.info(f"Saved RNG state to {rng_state_path}")
            save_model_architecture(pr.policy, path.replace(".safetensors", ".genericmodel"))

        except Exception as e:
            logger.warning(f"Failed to save RNG state to {rng_state_path}: {e}")

        return str(safetensors_path)

    # ?? this just returns the pr that is passed in. is that correct?
    def save_policy(
        self, pr: PolicyRecord, checkpoint_file_type: CheckpointFileType = "pt", path: str | None = None
    ) -> PolicyRecord:
        """Save a policy record using the specified checkpoint file type."""
        if path is None:
            if hasattr(pr, "file_path"):
                path = pr.file_path
            elif pr.uri is not None:
                path = pr.uri[7:] if pr.uri.startswith("file://") else pr.uri
            else:
                raise ValueError("PolicyRecord has no file_path or uri")

        # if saving both, take path from safetensors
        if checkpoint_file_type in ["safetensors", "pt_also_emit_safetensors"]:
            path = self.save_to_safetensors_file(pr, path)
        if checkpoint_file_type in ["pt", "pt_also_emit_safetensors"]:
            path = self.save_to_pt_file(pr, path)

        # Cache the policy record
        # ?? encapsulation
        if path is not None and self._cached_prs is not None:
            self._cached_prs.put(path, pr)

        return pr
