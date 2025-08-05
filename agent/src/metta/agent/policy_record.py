"""
PolicyRecord: A lightweight data structure for storing policy metadata and references.
This is separated from PolicyStore to enable cleaner packaging of saved policies.
"""

import logging
from typing import TYPE_CHECKING

import torch

from metta.agent.policy_metadata import PolicyMetadata

if TYPE_CHECKING:
    from metta.agent.metta_agent import PolicyAgent
    from metta.agent.policy_store import PolicyStore

logger = logging.getLogger(__name__)


class PolicyRecord:
    """A record containing a policy and its metadata."""

    def __init__(self, policy_store: "PolicyStore", run_name: str, uri: str | None, metadata: PolicyMetadata):
        self._policy_store = policy_store
        self.run_name = run_name  # Human-readable identifier (e.g., from wandb). Can include version
        self.uri: str | None = uri
        # Use the setter to ensure proper type
        self.metadata = metadata
        self._cached_policy: "PolicyAgent | None" = None

    @property
    def metadata(self) -> PolicyMetadata:
        """Get the metadata."""
        if not hasattr(self, "_metadata"):
            # Try backwards compatibility names
            old_metadata_names = ["checkpoint"]
            for name in old_metadata_names:
                if hasattr(self, name):
                    logger.warning(
                        f"Found metadata under old attribute name '{name}'. "
                        f"This PolicyRecord was saved with an older version of the code. "
                        f"Converting to new format."
                    )
                    # Set using the property setter for proper conversion
                    self.metadata = getattr(self, name)
                    return self._metadata

            # If no old names found, collect available attributes
            available_attrs = {}
            for attr in dir(self):
                if attr == "metadata":  # Skip this property to avoid recursion
                    continue
                if not attr.startswith("_"):
                    try:
                        value = getattr(self, attr)
                        if not callable(value):
                            available_attrs[attr] = type(value).__name__
                    except Exception as e:
                        available_attrs[attr] = f"<Error accessing: {e}>"

            raise AttributeError(
                f"No metadata found under any known attribute names. "
                f"Available attributes: {available_attrs}. "
                f"This PolicyRecord may be corrupted or from an incompatible version."
            )
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> None:
        """Set metadata, ensuring it's a PolicyMetadata instance."""
        if isinstance(value, PolicyMetadata):
            self._metadata = value
        elif isinstance(value, dict):
            # Automatically convert dict to PolicyMetadata
            self._metadata = PolicyMetadata(**value)
        else:
            raise TypeError(f"metadata must be PolicyMetadata or dict, got {type(value).__name__}")

    @property
    def file_path(self) -> str:
        """Extract the file_path from the URI"""
        file_uri_prefix = "file://"
        if self.uri is None:
            raise ValueError("Cannot get file_path without a valid URI.")
        if not self.uri.startswith(file_uri_prefix):
            raise ValueError(f"file_path() only applies to {file_uri_prefix} URIs, but got: {self.uri}.")

        return self.uri[len(file_uri_prefix) :]

    @property
    def policy(self) -> "PolicyAgent":
        """Load and return the policy, using cache if available."""
        if self._cached_policy is None:
            if self._policy_store is None:
                # Standalone loading is not supported
                raise ValueError(
                    "Cannot load policy without a PolicyStore. "
                    "PolicyRecord must be created through PolicyStore for loading functionality."
                )
            else:
                if self.uri is None:
                    raise ValueError("Cannot load policy without a valid URI.")
                pr = self._policy_store.load_from_uri(self.uri)
                if pr._cached_policy is None:
                    raise ValueError(f"Policy loaded from {self.uri} has no cached policy!")
                # access _cached_policy directly to avoid recursion
                self._cached_policy = pr._cached_policy

        return self._cached_policy

    @policy.setter
    def policy(self, policy: "PolicyAgent") -> None:
        """Set or overwrite the policy.

        Args:
            policy: The PyTorch module to set as the policy.

        Raises:
            TypeError: If policy is not a nn.Module.
        """
        self._cached_policy = policy
        logger.info(f"Policy overwritten for {self.run_name}")

    def policy_as_metta_agent(self):
        """Return the policy, ensuring it's a MettaAgent type."""
        policy = self.policy
        if type(policy).__name__ not in {"MettaAgent", "DistributedMettaAgent", "PytorchAgent"}:
            raise TypeError(f"Expected MettaAgent, DistributedMettaAgent, or PytorchAgent, got {type(policy).__name__}")
        return policy

    def num_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

    def __repr__(self):
        """Generate a detailed representation of the PolicyRecord."""
        # Basic policy record info
        lines = [f"PolicyRecord(name={self.run_name}, uri={self.uri})"]

        # Add key metadata if available
        important_keys = ["epoch", "agent_step", "generation", "score"]
        metadata_items = []
        for k in important_keys:
            if k in self.metadata:
                metadata_items.append(f"{k}={self.metadata[k]}")

        if metadata_items:
            lines.append(f"Metadata: {', '.join(metadata_items)}")

        # Load policy if not already loaded
        policy = None
        if self._cached_policy is None:
            try:
                policy = self.policy
            except Exception as e:
                lines.append(f"Error loading policy: {str(e)}")
                return "\n".join(lines)
        else:
            policy = self._cached_policy

        # Add total parameter count
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Add module structure with detailed weight shapes
        lines.append("\nModule Structure with Weight Shapes:")

        for name, module in policy.named_modules():
            # Skip top-level module
            if name == "":
                continue

            # Create indentation based on module hierarchy
            indent = "  " * name.count(".")

            # Get module type
            module_type = module.__class__.__name__

            # Start building the module info line
            module_info = f"{indent}{name}: {module_type}"

            # Get parameters for this module (non-recursive)
            params = list(module.named_parameters(recurse=False))

            # Add detailed parameter information
            if params:
                # For common layer types, add specialized shape information
                if isinstance(module, torch.nn.Conv2d):
                    weight = next((p for pname, p in params if pname == "weight"), None)
                    if weight is not None:
                        out_channels, in_channels, kernel_h, kernel_w = weight.shape
                        module_info += " ["
                        module_info += f"out_channels={out_channels}, "
                        module_info += f"in_channels={in_channels}, "
                        module_info += f"kernel=({kernel_h}, {kernel_w})"
                        module_info += "]"

                elif isinstance(module, torch.nn.Linear):
                    weight = next((p for pname, p in params if pname == "weight"), None)
                    if weight is not None:
                        out_features, in_features = weight.shape
                        module_info += f" [in_features={in_features}, out_features={out_features}]"

                elif isinstance(module, torch.nn.LSTM):
                    module_info += " ["
                    module_info += f"input_size={module.input_size}, "
                    module_info += f"hidden_size={module.hidden_size}, "
                    module_info += f"num_layers={module.num_layers}"
                    module_info += "]"

                elif isinstance(module, torch.nn.Embedding):
                    weight = next((p for pname, p in params if pname == "weight"), None)
                    if weight is not None:
                        num_embeddings, embedding_dim = weight.shape
                        module_info += f" [num_embeddings={num_embeddings}, embedding_dim={embedding_dim}]"

                # Add all parameter shapes
                param_shapes = []
                for param_name, param in params:
                    param_shapes.append(f"{param_name}={list(param.shape)}")

                if param_shapes and not any(
                    x in module_info for x in ["out_channels", "in_features", "hidden_size", "num_embeddings"]
                ):
                    module_info += f" ({', '.join(param_shapes)})"

            # Add formatted module info to output
            lines.append(module_info)

        # Add section for buffer shapes (non-parameter tensors like running_mean in BatchNorm)
        buffers = list(policy.named_buffers())
        if buffers:
            lines.append("\nBuffer Shapes:")
            for name, buffer in buffers:
                lines.append(f"  {name}: {list(buffer.shape)}")

        return "\n".join(lines)
