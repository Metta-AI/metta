"""
PolicyRecord implementation using torch.package for robust save/load functionality.

This module implements PolicyRecord which tracks trained policies and uses torch.package
to handle all dependency management and code packaging automatically.

Key design decisions:
1. We use torch.package for all saves to get automatic dependency management
2. Models loaded from torch.package cannot be directly re-saved (due to package namespace issues)
3. The trainer handles this by creating fresh model instances before saving when needed
4. Once a fresh instance is created, it can be saved and loaded normally

This approach gives us the benefits of torch.package (no dependency issues when loading)
while the trainer handles the constraint that packaged models can't be re-packaged.
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

        # Use duck typing instead of isinstance to handle torch.package loaded classes
        required_attrs = ["components", "activate_actions", "policy"]
        for attr in required_attrs:
            if not hasattr(policy, attr):
                raise TypeError(
                    f"Expected MettaAgent or DistributedMettaAgent interface, "
                    f"but policy is missing attribute '{attr}'. Got {type(policy).__name__}"
                )

        return policy

    def num_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self) -> Optional[str]:
        """Get the local file path if available."""
        return self._local_path

    def _clean_metadata_for_packaging(self, metadata: dict) -> dict:
        """Clean metadata to remove any objects that can't be packaged."""
        import copy

        def clean_value(v):
            # Check if it's a wandb object
            if hasattr(v, "__module__") and v.__module__ and "wandb" in v.__module__:
                return None  # Remove wandb objects
            elif isinstance(v, dict):
                return {k: clean_value(val) for k, val in v.items() if clean_value(val) is not None}
            elif isinstance(v, list):
                return [clean_value(item) for item in v if clean_value(item) is not None]
            elif isinstance(v, (str, int, float, bool, type(None))):
                return v
            elif hasattr(v, "__dict__"):
                # For other objects, try to convert to a simple representation
                try:
                    return str(v)
                except:
                    return None
            else:
                return v

        return clean_value(copy.deepcopy(metadata))

    def save(self, path: str, policy: nn.Module) -> "PolicyRecord":
        """Save a policy using torch.package for automatic dependency management."""
        logger.info(f"Saving policy to {path} using torch.package")

        # Update local path
        self._local_path = path
        self.uri = "file://" + path

        # Get the actual policy (unwrap MettaAgent if needed)
        # Check for .policy attribute instead of isinstance to handle torch.package loaded classes
        if (
            hasattr(policy, "policy")
            and hasattr(policy.__class__, "__name__")
            and "MettaAgent" in policy.__class__.__name__
        ):
            actual_policy = policy.policy
        else:
            actual_policy = policy

        try:
            # Use torch.package to save the policy with all dependencies
            with PackageExporter(path, debug=False) as exporter:
                # Intern all metta modules to include them in the package
                exporter.intern("metta.**")

                # Check if the policy comes from __main__ (common in scripts/notebooks)
                if actual_policy.__class__.__module__ == "__main__":
                    # Get the source code of the class
                    import inspect

                    try:
                        source = inspect.getsource(actual_policy.__class__)
                        # Prepend necessary imports to the source
                        full_source = "import torch\nimport torch.nn as nn\n\n" + source
                        # Save it as a module source
                        exporter.save_source_string("__main__", full_source)
                    except:
                        # If we can't get source, just extern it and hope for the best
                        exporter.extern("__main__")

                # External modules that should use the system version
                exporter.extern("torch")
                exporter.extern("torch.**")
                exporter.extern("numpy")
                exporter.extern("numpy.**")
                exporter.extern("scipy")
                exporter.extern("scipy.**")
                exporter.extern("sklearn")
                exporter.extern("sklearn.**")
                exporter.extern("matplotlib")
                exporter.extern("matplotlib.**")
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
                exporter.extern("torch_scatter")
                exporter.extern("torch_geometric")
                exporter.extern("torch_sparse")

                # Mock ALL wandb modules more comprehensively
                exporter.mock("wandb")
                exporter.mock("wandb.**")
                exporter.mock("wandb.*")
                exporter.mock("wandb.sdk")
                exporter.mock("wandb.sdk.**")
                exporter.mock("wandb.sdk.wandb_run")

                # Mock other modules
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
                exporter.mock("seaborn")
                exporter.mock("plotly")

                # Handle C extension modules
                exporter.extern("mettagrid.mettagrid_c")
                exporter.extern("mettagrid")
                exporter.extern("mettagrid.**")

                # Create a clean copy of metadata without wandb references
                clean_metadata = self._clean_metadata_for_packaging(self.metadata)

                # Save a minimal PolicyRecord with clean metadata
                minimal_pr = PolicyRecord(None, self.name, self.uri, clean_metadata)
                exporter.save_pickle("policy_record", "data.pkl", minimal_pr)

                # Save the actual policy
                exporter.save_pickle("policy", "model.pkl", actual_policy)

            logger.info(f"Saved policy with torch.package to {path}")

        except Exception as e:
            logger.warning(f"torch.package save failed: {e}")
            logger.info("Falling back to state_dict only save")

            # Fallback to regular torch save with state_dict only
            checkpoint = {
                "model_state_dict": actual_policy.state_dict(),
                "metadata": self._clean_metadata_for_packaging(self.metadata),
                "model_class_name": actual_policy.__class__.__name__,
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved policy state_dict to {path}")

        return self

    def load(self, path: str, device: str = "cpu") -> nn.Module:
        """Load a policy from a torch.package file."""
        logger.info(f"Loading policy from {path}")

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
            # Not a torch.package file
            logger.info(f"Not a torch.package file ({e})")

            # We don't support non-torch.package files
            raise ValueError(
                f"Cannot load policy from {path}: This file is not a valid torch.package file. "
                "All policies must be saved using torch.package."
            )

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
