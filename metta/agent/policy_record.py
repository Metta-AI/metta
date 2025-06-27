"""
PolicyRecord: A lightweight data structure for storing policy metadata and references.
This is separated from PolicyStore to enable cleaner packaging of saved policies.
"""

import logging
from typing import Optional

import torch
from torch import nn
from torch.package.package_importer import PackageImporter

logger = logging.getLogger("policy_record")


class PolicyRecord:
    """A record containing a policy and its metadata."""

    def __init__(self, policy_store, name: str, uri: str, metadata: dict):
        self._policy_store = policy_store
        self.name = name
        self.uri: str = uri
        self.metadata = metadata
        self._cached_policy = None

    @property
    def policy(self) -> nn.Module:
        """Load and return the policy, using cache if available."""
        if self._cached_policy is None:
            if self._policy_store is None:
                # If no policy store, try to load directly (for packaged policies)
                local_path = self.local_path()
                if local_path is not None:
                    self._cached_policy = self.load(local_path)
                else:
                    raise ValueError("Cannot load policy without policy_store or a file:// local path uri")
            else:
                pr = self._policy_store.load_from_uri(self.uri)
                # FIX: Access _policy directly to avoid recursion
                self._cached_policy = pr._policy

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

    def local_path(self) -> Optional[str]:
        """Return the local file path if available."""
        if self.uri.startswith("file://"):
            return self.uri[len("file://") :]
        else:
            return None

    def load(self, path: str, device: str = "cpu") -> nn.Module:
        """Load a policy from a torch package file."""
        logger.info(f"Loading policy from {path}")
        try:
            importer = PackageImporter(path)
            try:
                return importer.load_pickle("policy", "model.pkl", map_location=device)
            except Exception as e:
                logger.warning(f"Could not load policy directly: {e}")
                pr = importer.load_pickle("policy_record", "data.pkl", map_location=device)
                if hasattr(pr, "_policy") and pr._cached_policy is not None:
                    return pr._cached_policy
                raise ValueError("PolicyRecord in package does not contain a policy") from e
        except Exception as e:
            logger.info(f"Not a torch.package file ({e})")
            raise ValueError(f"Cannot load policy from {path}: This file is not a valid torch.package file.") from e

    def key_and_version(self) -> tuple[str, int]:
        """Extract the policy key and version from the URI."""

        # Get the last part after splitting by slash
        base_name = self.uri.split("/")[-1]

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
