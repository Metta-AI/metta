"""
PolicyRecord: A lightweight data structure for storing policy metadata and references.
"""

import logging

import torch

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_metadata import PolicyMetadata

logger = logging.getLogger(__name__)


class PolicyRecord:
    """A record containing a policy and its metadata."""

    def __init__(
        self,
        run_name: str,
        uri: str | None,
        metadata: PolicyMetadata | dict,
        policy: "PolicyAgent",
        wandb_entity: str | None = None,  # for loading policies from wandb
        wandb_project: str | None = None,  # for loading policies from wandb
    ):
        self.run_name = run_name  # Human-readable identifier (e.g., from wandb). Can include version
        self.uri: str | None = uri
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Set metadata directly - must be PolicyMetadata or dict
        if isinstance(metadata, PolicyMetadata):
            self._metadata = metadata
        elif isinstance(metadata, dict):
            # Automatically convert dict to PolicyMetadata
            self._metadata = PolicyMetadata(**metadata)
        else:
            raise TypeError(f"metadata must be PolicyMetadata or dict, got {type(metadata).__name__}")

        # Set policy directly - must be a PolicyAgent
        if not isinstance(policy, PolicyAgent):
            raise TypeError(f"policy must be a PolicyAgent, got {type(policy).__name__}")
        self._cached_policy: "PolicyAgent" = policy

    def extract_wandb_run_info(self) -> tuple[str, str, str, str | None]:
        if self.uri is None or not self.uri.startswith("wandb://"):
            raise ValueError("Cannot get wandb info without a valid URI.")
        try:
            entity, project, name = self.uri[len("wandb://") :].split("/")
            version: str | None = None
            if ":" in name:
                name, version = name.split(":")
            return entity, project, name, version
        except ValueError as e:
            raise ValueError(
                f"Failed to parse wandb URI: {self.uri}. Expected format: wandb://<entity>/<project>/<name>"
            ) from e

    @property
    def metadata(self) -> PolicyMetadata:
        """Get the metadata."""
        if not hasattr(self, "_metadata"):
            # Try backwards compatibility names
            old_metadata_names = ["checkpoint"]
            for name in old_metadata_names:
                # ?? this should be removed
                if hasattr(self, name):
                    logger.warning(
                        f"Found metadata under old attribute name '{name}'. "
                        f"This PolicyRecord was saved with an older version of the code. "
                        f"Converting to new format."
                    )
                    # Convert old metadata to new format
                    old_metadata = getattr(self, name)
                    if isinstance(old_metadata, PolicyMetadata):
                        self._metadata = old_metadata
                    elif isinstance(old_metadata, dict):
                        self._metadata = PolicyMetadata(**old_metadata)
                    else:
                        raise TypeError(
                            f"Old metadata must be PolicyMetadata or dict, got {type(old_metadata).__name__}"
                        )
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
        """Get the policy."""
        return self._cached_policy

    @property
    def cached_policy(self) -> "PolicyAgent":
        """Get the cached policy."""
        return self._cached_policy

    @cached_policy.setter
    def cached_policy(self, policy: "PolicyAgent") -> None:
        """Set the cached policy directly."""
        if not isinstance(policy, PolicyAgent):
            raise TypeError(f"cached_policy must be a PolicyAgent, got {type(policy).__name__}")
        self._cached_policy = policy

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

        policy = self._cached_policy

        # Add total parameter count
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Check if this is a legacy checkpoint wrapped in adapter
        try:
            from metta.agent.legacy_adapter import LegacyMettaAgentAdapter

            if hasattr(policy, "policy") and isinstance(policy.policy, LegacyMettaAgentAdapter):
                lines.append("\nNOTE: Legacy checkpoint loaded via LegacyMettaAgentAdapter for backwards compatibility")
        except ImportError:
            # Legacy adapter not available, skip this check
            pass

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
