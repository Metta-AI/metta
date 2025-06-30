"""
PolicyRecord: A lightweight data structure for storing policy metadata and references.
This is separated from PolicyStore to enable cleaner packaging of saved policies.
"""

import logging
import os
from typing import Optional, Union

import torch
from torch import nn

from metta.agent.policy_metadata import PolicyMetadata

logger = logging.getLogger(__name__)


class PolicyRecord:
    """A record containing a policy and its metadata."""

    def __init__(self, policy_store, name: str, uri: str, metadata: Union[PolicyMetadata, dict]):
        self._policy_store = policy_store
        self.name = name  # Human-readable identifier (e.g., from wandb)
        self.uri: str = uri
        # Use the setter to ensure proper type
        self.metadata = metadata
        self._cached_policy = None

    @property
    def metadata(self) -> PolicyMetadata:
        """Get the metadata."""
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
        if not self.uri.startswith(file_uri_prefix):
            raise ValueError(f"file_path() only applies to {file_uri_prefix} URIs, but got: {self.uri}.")

        return self.uri[len(file_uri_prefix) :]

    @property
    def policy(self) -> nn.Module:
        """Load and return the policy, using cache if available."""
        if self._cached_policy is None:
            if self._policy_store is None:
                # If no policy store, try to load directly (for packaged policies)
                path = self.file_path
                if path is not None:
                    self._cached_policy = self.load_from_file(path)
                else:
                    raise ValueError("Cannot load policy without policy_store or a file:// local path uri")
            else:
                pr = self._policy_store.load_from_uri(self.uri)
                # FIX: Access _cached_policy directly to avoid recursion
                self._cached_policy = pr._cached_policy

        return self._cached_policy

    @policy.setter
    def policy(self, policy: nn.Module) -> None:
        """Set or overwrite the policy.

        Args:
            policy: The PyTorch module to set as the policy.

        Raises:
            TypeError: If policy is not a nn.Module.
        """
        if not isinstance(policy, nn.Module):
            raise TypeError(f"Policy must be a torch.nn.Module, got {type(policy).__name__}")
        self._cached_policy = policy
        logger.info(f"Policy overwritten for {self.name}")

    def policy_as_metta_agent(self):
        """Return the policy, ensuring it's a MettaAgent type."""
        policy = self.policy
        if type(policy).__name__ not in {"MettaAgent", "DistributedMettaAgent", "PytorchAgent"}:
            raise TypeError(f"Expected MettaAgent, DistributedMettaAgent, or PytorchAgent, got {type(policy).__name__}")
        return policy

    def num_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

    def load_from_file(self, path: str, device: str = "cpu") -> nn.Module:
        """Load a policy from a file using standard torch.load."""
        logger.info(f"Loading policy from {path}")

        # Standard torch checkpoint loading
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # New checkpoint format (v2.0)
            if checkpoint.get("checkpoint_version") == "2.0" and "model" in checkpoint:
                return checkpoint["model"].to(device)

            # Legacy dict format with state dict
            elif "model_state_dict" in checkpoint:
                # This is a state dict only - need to reconstruct the model
                # For now, we'll need the environment and config from the policy store
                if self._policy_store is None:
                    raise ValueError("Cannot reconstruct model from state dict without policy_store context")

                # TODO: This is temporary - we'll need to save env config with the checkpoint
                raise NotImplementedError("Loading from state dict only not yet implemented")

            else:
                raise ValueError(f"Unknown checkpoint dictionary format with keys: {list(checkpoint.keys())}")

        elif isinstance(checkpoint, nn.Module):
            # Direct model save
            return checkpoint.to(device)

        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    def save_to_file(self, path: Optional[str] = None, packaging_rules_callback=None) -> "PolicyRecord":
        """Save a policy and its metadata using standard torch.save.

        Args:
            path: Optional path to save to. If None, uses the path from the URI.
            packaging_rules_callback: Ignored - kept for compatibility.

        Returns:
            Self, with updated URI if a new path was provided.
        """
        if path is None:
            if not self.uri.startswith("file://"):
                raise ValueError("Can only save to file:// URIs without explicit path")
            path = self.file_path

        if os.path.exists(path):
            logger.warning(f"Overwriting existing policy at {path}")
        else:
            logger.info(f"Saving policy to {path}")

        try:
            # Create checkpoint dictionary with all necessary information
            checkpoint = {
                "model_state_dict": self.policy.state_dict(),
                "model_class": self.policy.__class__.__name__,
                "model_module": self.policy.__class__.__module__,
                "metadata": self.metadata.sanitized(),
                "name": self.name,
                "uri": f"file://{path}",
                # Include architecture information for reconstruction
                "agent_attributes": getattr(self.policy, "agent_attributes", {}),
                # Version info for compatibility checking
                "checkpoint_version": "2.0",  # New simplified format
            }

            # Save the complete model as well for easier loading
            checkpoint["model"] = self.policy

            torch.save(checkpoint, path)
            logger.info(f"Successfully saved checkpoint to {path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Failed to save policy: {e}") from e

        self.uri = f"file://{path}"
        return self

    def key_and_version(self) -> tuple[str, int]:
        """Extract key and version from the URI, handling both wandb:// and file:// URIs.

        Returns:
            Tuple of (key, version_number)
            - For wandb:// URIs: (artifact_name, wandb_version)
            - For file:// URIs: (filename_without_extension, epoch_from_metadata)
        """
        if self.uri.startswith("wandb://"):
            # Remove wandb:// prefix and get the last part
            artifact_path = self.uri[8:]
            base_name = artifact_path.split("/")[-1]

            # Check if it has a version number in format ":vNUM"
            if ":" in base_name and ":v" in base_name:
                parts = base_name.split(":v")
                try:
                    version = int(parts[1])
                    key = parts[0]
                except ValueError:
                    key = base_name
                    version = 0
            else:
                # No version, use the whole thing as key and version = 0
                key = base_name
                version = 0

            return key, version

        elif self.uri.startswith("file://"):
            # For file URIs, use filename as key and epoch as version
            path = self.file_path
            filename = os.path.basename(path)
            # Remove extension to get key
            key = os.path.splitext(filename)[0]
            # Use epoch from metadata as version, defaulting to 0
            version = self.metadata.get("epoch", 0)
            return key, version

        else:
            # For other URIs, return a sensible default
            return self.name, self.metadata.get("epoch", 0)

    def wandb_key_and_version(self) -> tuple[str, int]:
        """Extract the wandb artifact key and version from the URI.

        Returns:
            Tuple of (artifact_key, version_number)

        Raises:
            ValueError: If the URI is not a wandb:// URI
        """
        if not self.uri.startswith("wandb://"):
            raise ValueError(
                f"wandb_key_and_version() only applies to wandb:// URIs, "
                f"but got: {self.uri}. Use key_and_version() for a general method that handles all URI types."
            )

        return self.key_and_version()  # Delegate to general method

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
