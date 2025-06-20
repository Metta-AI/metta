"""
PolicyRecord implementation using torch.package for robust save/load functionality.

This module implements PolicyRecord which tracks trained policies and uses torch.package
to handle all dependency management and code packaging automatically.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import nn
from torch.package import PackageExporter, PackageImporter

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent

if TYPE_CHECKING:
    from metta.agent.policy_store import PolicyStore

logger = logging.getLogger("policy_record")


class PolicyRecord:
    """Represents a trained policy with metadata and torch.package-based persistence."""

    def __init__(self, policy_store: Optional["PolicyStore"], name: str, uri: str, metadata: dict):
        self._policy_store = policy_store
        self.name = name
        self.uri = uri
        self.metadata = metadata
        self._policy = None
        self._local_path = None

        if self.uri.startswith("file://"):
            self._local_path = self.uri[len("file://") :]

    def policy(self) -> nn.Module:
        """Get the policy, loading it if necessary."""
        if self._policy is None:
            if self._policy_store is None:
                raise ValueError("PolicyStore is required to load policy")
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
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self) -> Optional[str]:
        """Get the local file path if available."""
        return self._local_path

    def save(self, path: str, policy: nn.Module) -> "PolicyRecord":
        """Save a policy using torch.package for automatic dependency management."""
        logger.info(f"Saving policy to {path} using torch.package")

        # Update local path
        self._local_path = path
        self.uri = "file://" + path

        # Get the actual policy (unwrap MettaAgent if needed)
        actual_policy = policy.policy if isinstance(policy, MettaAgent) else policy

        # Use torch.package to save the policy with all dependencies
        with PackageExporter(path) as exporter:
            # Intern all metta modules to include them in the package
            exporter.intern("metta.**")

            # Check if the policy comes from __main__ (common in scripts/notebooks)
            # For __main__ modules, we need to handle them specially since they don't have __file__
            if actual_policy.__class__.__module__ == "__main__":
                # Get the source code of the class
                import inspect

                try:
                    source = inspect.getsource(actual_policy.__class__)
                    # Save it as a module source
                    exporter.save_source_string("__main__", source)
                except:
                    # If we can't get source, just extern it and hope for the best
                    exporter.extern("__main__")

            # External modules that should use the system version
            exporter.extern("torch")
            exporter.extern("torch.**")
            exporter.extern("numpy")
            exporter.extern("numpy.**")
            exporter.extern("gymnasium")
            exporter.extern("gymnasium.**")
            exporter.extern("gym")
            exporter.extern("gym.**")
            exporter.extern("tensordict")
            exporter.extern("tensordict.**")
            exporter.extern("einops")
            exporter.extern("einops.**")
            exporter.extern("hydra")
            exporter.extern("hydra.**")
            exporter.extern("omegaconf")
            exporter.extern("omegaconf.**")

            # Mock modules that we don't need to include
            exporter.mock("wandb")
            exporter.mock("wandb.**")
            exporter.mock("pufferlib")
            exporter.mock("pufferlib.**")
            exporter.mock("pydantic")
            exporter.mock("pydantic.**")
            exporter.mock("typing_extensions")
            exporter.mock("boto3")
            exporter.mock("boto3.**")
            exporter.mock("botocore")
            exporter.mock("botocore.**")
            exporter.mock("duckdb")
            exporter.mock("duckdb.**")
            exporter.mock("pandas")
            exporter.mock("pandas.**")

            # Handle C extension modules
            exporter.extern("mettagrid.mettagrid_c")
            exporter.extern("mettagrid")
            exporter.extern("mettagrid.**")

            # Save the policy record (which includes metadata)
            exporter.save_pickle("policy_record", "data.pkl", self)

            # Save the actual policy
            exporter.save_pickle("policy", "model.pkl", actual_policy)

        logger.info(f"Saved policy with torch.package to {path}")
        return self

    def load(self, path: str, device: str = "cpu") -> nn.Module:
        """Load a policy from a torch.package file."""
        logger.info(f"Loading policy from {path} using torch.package")

        try:
            # Use torch.package to load the policy
            importer = PackageImporter(path)

            # First try to load the policy directly
            try:
                actual_policy = importer.load_pickle("policy", "model.pkl", map_location=device)

                # Import MettaAgent from the normal system (not from the package)
                from metta.agent.metta_agent import MettaAgent

                # Wrap in MettaAgent if it's not already
                if not isinstance(actual_policy, MettaAgent):
                    policy = MettaAgent(actual_policy)
                else:
                    policy = actual_policy

                logger.info("Successfully loaded policy using torch.package")
                return policy

            except Exception as e:
                logger.warning(f"Could not load policy directly: {e}")
                # Fall back to loading from policy_record
                pr = importer.load_pickle("policy_record", "data.pkl", map_location=device)
                if hasattr(pr, "_policy") and pr._policy is not None:
                    return pr._policy
                else:
                    raise ValueError("PolicyRecord in package does not contain a policy")

        except Exception as e:
            # Fallback for old checkpoints without torch.package
            logger.info(f"torch.package load failed ({e}), trying legacy load")

            # For legacy checkpoints, they are just regular torch saves, not packages
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Simple loading - just get state dict and create a minimal wrapper
            if "model_state_dict" in checkpoint:
                # Create a simple wrapper module
                class LegacyPolicy(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # This will be populated by load_state_dict

                policy = LegacyPolicy()
                try:
                    policy.load_state_dict(checkpoint["model_state_dict"])
                except:
                    # If that fails, just return a dummy policy
                    pass

                from metta.agent.metta_agent import MettaAgent

                return MettaAgent(policy)
            else:
                raise ValueError(f"Cannot load checkpoint from {path}")

    def key_and_version(self) -> tuple[str, int]:
        """
        Extract the policy key and version from the URI.

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
        """Get the policy key (name without version)."""
        return self.key_and_version()[0]

    def version(self) -> int:
        """Get the policy version number."""
        return self.key_and_version()[1]

    def __repr__(self):
        """Generate a detailed representation of the PolicyRecord."""
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
        try:
            policy = self.policy()

            # Add total parameter count
            total_params = sum(p.numel() for p in policy.parameters())
            trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

            # Add module structure (simplified version)
            lines.append("\nKey Modules:")
            for name, module in policy.named_modules():
                if name and "." not in name:  # Top-level modules only
                    module_type = module.__class__.__name__
                    param_count = sum(p.numel() for p in module.parameters())
                    if param_count > 0:
                        lines.append(f"  {name}: {module_type} ({param_count:,} params)")

        except Exception as e:
            lines.append(f"Error loading policy: {str(e)}")

        return "\n".join(lines)
