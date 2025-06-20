"""
PolicyRecord implementation with robust save/load functionality.

This module implements PolicyRecord which tracks trained policies along with their
construction context, enabling exact reconstruction even when class definitions change.
"""

import inspect
import logging
import sys
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import nn

from metta.agent.build_context import BuildContext
from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent

if TYPE_CHECKING:
    from metta.agent.policy_store import PolicyStore

logger = logging.getLogger("policy_record")


class PolicyRecord:
    """Represents a trained policy with metadata and reconstruction information."""

    def __init__(self, policy_store: Optional["PolicyStore"], name: str, uri: str, metadata: dict):
        self._policy_store = policy_store
        self.name = name
        self.uri = uri
        self.metadata = metadata
        self._policy = None
        self._local_path = None
        self._build_context = None

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
            self._build_context = pr._build_context
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

    def save(self, path: str, policy: nn.Module, build_context: Optional[BuildContext] = None) -> "PolicyRecord":
        """Save a policy with its build context for robust reconstruction."""
        logger.info(f"Saving policy to {path}")

        # Update local path
        self._local_path = path
        self.uri = "file://" + path

        # Get the actual policy (unwrap MettaAgent if needed)
        actual_policy = policy.policy if isinstance(policy, MettaAgent) else policy

        # Save build context if provided
        if build_context:
            self._build_context = build_context
            # Capture source code for the policy class if not already captured
            if not build_context.source_code:
                build_context.source_code = self._capture_source_code(actual_policy)

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": actual_policy.state_dict(),
            "policy_record": self,
            "metadata": self.metadata,
        }

        # Add build context if available
        if self._build_context:
            checkpoint_data["build_context"] = self._build_context.to_dict()

        # Save the checkpoint
        torch.save(checkpoint_data, path)
        logger.info(f"Saved policy with build context to {path}")

        return self

    def load(self, path: str, device: str = "cpu") -> nn.Module:
        """Load a policy from file using build context for reconstruction."""
        logger.info(f"Loading policy from {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Check if we have build context
        if "build_context" in checkpoint:
            build_context = BuildContext.from_dict(checkpoint["build_context"])
            self._build_context = build_context

            # Reconstruct using the original builder method
            try:
                # Import MettaAgentBuilder here to avoid circular imports
                from metta.agent.metta_agent import MettaAgentBuilder

                # Need the policy store's config for the builder
                if self._policy_store is None:
                    raise ValueError("PolicyStore is required for reconstruction")
                builder = MettaAgentBuilder(self._policy_store._cfg)

                # Call the appropriate builder method
                if build_context.method == "build_from_brain_policy":
                    # Reconstruct environment from saved attributes
                    env = self._reconstruct_env(build_context.env_attributes)
                    agent, _ = builder.build_from_brain_policy(env)

                elif build_context.method == "build_from_pytorch_policy":
                    # For pytorch policies, we need to save and reload the checkpoint
                    # This is a bit circular, but necessary for the current architecture
                    agent, _ = builder.build_from_pytorch_policy(path)

                elif build_context.method == "build_from_policy_class":
                    # Reconstruct the class from source code
                    class_name = build_context.kwargs.get("class_name", "PytorchPolicy")

                    # Try to find the class in the source code
                    policy_class = None
                    try:
                        policy_class = self._reconstruct_class_from_source(build_context.source_code, class_name)
                    except ValueError as e:
                        logger.warning(f"Could not reconstruct {class_name}: {e}")

                        # If the original class can't be found, try a simpler approach
                        # Just create a dummy module and load the state dict

                        # Create a simple wrapper that will hold the state
                        class ReconstructedPolicy(nn.Module):
                            def __init__(self):
                                super().__init__()
                                # The state dict will be loaded into this

                        policy_class = ReconstructedPolicy
                        logger.info("Using fallback ReconstructedPolicy class")

                    # Build the agent with the reconstructed class
                    # Remove 'class_name' from kwargs as it's not a constructor argument
                    constructor_kwargs = {k: v for k, v in build_context.kwargs.items() if k != "class_name"}
                    agent, _ = builder.build_from_policy_class(policy_class, *build_context.args, **constructor_kwargs)

                else:
                    raise ValueError(f"Unknown build method: {build_context.method}")

                # Load the state dict
                actual_policy = agent.policy if isinstance(agent, MettaAgent) else agent
                actual_policy.load_state_dict(checkpoint["model_state_dict"])

                logger.info(f"Successfully reconstructed policy using {build_context.method}")
                return agent

            except Exception as e:
                logger.warning(f"Failed to reconstruct using build context: {e}")
                logger.info("Falling back to legacy reconstruction")
        else:
            logger.info("No build context found in checkpoint, using legacy reconstruction")

        # For old checkpoints, try to reconstruct from metadata
        if "metadata" in checkpoint and "reconstruction_attributes" in checkpoint["metadata"]:
            try:
                # Import MettaAgentBuilder here to avoid circular imports
                from metta.agent.metta_agent import MettaAgentBuilder

                if self._policy_store is None:
                    raise ValueError("PolicyStore is required for reconstruction")

                builder = MettaAgentBuilder(self._policy_store._cfg)

                # Try to reconstruct as BrainPolicy using reconstruction_attributes
                env_attrs = checkpoint["metadata"]["reconstruction_attributes"]
                if env_attrs:
                    env = self._reconstruct_env(env_attrs)
                    agent, _ = builder.build_from_brain_policy(env)

                    # Load the state dict
                    actual_policy = agent.policy if isinstance(agent, MettaAgent) else agent
                    actual_policy.load_state_dict(checkpoint["model_state_dict"])

                    logger.info("Successfully reconstructed policy using legacy reconstruction_attributes")
                    return agent

            except Exception as e:
                logger.warning(f"Failed to reconstruct using legacy attributes: {e}")

        # Try to load the initial checkpoint's build context if this is a later checkpoint
        if self._try_initial_checkpoint_fallback(checkpoint, path, device):
            try:
                agent = self._reconstruct_with_initial_context(checkpoint, path, device)

                # Update our build context from the initial checkpoint
                # This is important so that subsequent saves will have the build context
                import os

                checkpoint_dir = os.path.dirname(path)
                initial_path = os.path.join(checkpoint_dir, "model_0000.pt")
                initial_checkpoint = torch.load(initial_path, map_location=device, weights_only=False)

                if "build_context" in initial_checkpoint:
                    self._build_context = BuildContext.from_dict(initial_checkpoint["build_context"])
                    logger.info("Updated PolicyRecord with build context from initial checkpoint")

                return agent
            except Exception as e:
                logger.warning(f"Failed to use initial checkpoint context: {e}")

        # Final fallback to source code reconstruction
        return self._load_from_source_code(checkpoint, device)

    def _capture_source_code(self, policy: nn.Module) -> dict:
        """Capture source code for a policy class and its dependencies."""
        source_code = {}
        try:
            # Handle wrapped policies (e.g., PytorchPolicy wrapping another policy)
            from metta.agent.pytorch_policy import PytorchPolicy

            if isinstance(policy, PytorchPolicy) and hasattr(policy, "policy"):
                # Capture source for the wrapped policy too
                wrapped_source = self._capture_source_code(policy.policy)
                source_code.update(wrapped_source)

            policy_class = policy.__class__
            module_name = policy_class.__module__
            class_name = policy_class.__name__
            full_class_path = f"{module_name}.{class_name}"

            # Save the class source
            try:
                source_code[full_class_path] = inspect.getsource(policy_class)
            except OSError:
                # If we can't get the source (e.g., built-in classes), skip
                logger.warning(f"Could not get source for class {full_class_path}")

            # Save the module source if possible
            module = sys.modules.get(module_name)
            if module and hasattr(module, "__file__") and module.__file__:
                try:
                    with open(module.__file__, "r") as f:
                        source_code[module_name] = f.read()
                except Exception:
                    # For __main__ module or modules without files, try to capture from source
                    if module_name == "__main__":
                        # Try to capture the entire __main__ module source
                        try:
                            import __main__

                            # Get all classes and functions defined in __main__
                            main_source = []
                            for name, obj in vars(__main__).items():
                                if inspect.isclass(obj) or inspect.isfunction(obj):
                                    try:
                                        main_source.append(inspect.getsource(obj))
                                    except:
                                        pass
                            if main_source:
                                source_code[module_name] = "\n\n".join(main_source)
                        except:
                            pass

        except Exception as e:
            logger.warning(f"Could not capture source code: {e}")

        return source_code

    def _reconstruct_env(self, env_attributes: dict):
        """Reconstruct an environment-like object from saved attributes."""
        from types import SimpleNamespace

        # Create a mock environment with the saved attributes
        env = SimpleNamespace()
        for key, value in env_attributes.items():
            setattr(env, key, value)
        return env

    def _reconstruct_class_from_source(self, source_code: dict, class_name: str):
        """Reconstruct a class from saved source code."""
        import importlib.util

        # Find the class in source code
        for path, code in source_code.items():
            if class_name in path or class_name in code:
                # Handle __main__ module specially
                if path.startswith("__main__"):
                    module_name = "__main__"
                else:
                    module_name = path.rsplit(".", 1)[0] if "." in path else path

                # Create a module from source
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)

                # Add common imports to the module's namespace
                module.__dict__["torch"] = torch
                module.__dict__["nn"] = nn
                module.__dict__["Optional"] = Optional

                # Execute the source code in the module
                exec(code, module.__dict__)

                # Get the class
                if hasattr(module, class_name):
                    return getattr(module, class_name)

        raise ValueError(f"Could not find class {class_name} in source code")

    def _load_from_source_code(self, checkpoint: dict, device: str) -> nn.Module:
        """Fallback method to load from source code (existing approach)."""
        # If we have source code in the checkpoint, try to reconstruct from that
        if "source_code" in checkpoint and checkpoint["source_code"]:
            try:
                # Try to find a class to reconstruct
                for class_path, source in checkpoint["source_code"].items():
                    if "." in class_path:
                        class_name = class_path.split(".")[-1]
                        try:
                            policy_class = self._reconstruct_class_from_source(checkpoint["source_code"], class_name)
                            # Create a simple instance
                            policy = policy_class()
                            policy.load_state_dict(checkpoint["model_state_dict"])

                            # Wrap in MettaAgent
                            from metta.agent.metta_agent import MettaAgent

                            return MettaAgent(policy)
                        except Exception as e:
                            logger.warning(f"Failed to reconstruct {class_name}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Failed to reconstruct from source code: {e}")

        # If all else fails, raise an error with helpful information
        error_msg = (
            f"Could not reconstruct policy from checkpoint at '{self._local_path or 'unknown path'}'.\n"
            "The checkpoint is missing build context and source code reconstruction failed.\n"
            "This appears to be a checkpoint saved with an older version of the code.\n\n"
            "To fix this, you can:\n"
            "1. Use the initial checkpoint (model_0000.pt) which should have build context\n"
            "2. Re-train from scratch with the updated code\n"
            "3. Manually reconstruct the policy if you know the exact configuration"
        )
        raise ValueError(error_msg)

    def _try_initial_checkpoint_fallback(self, checkpoint: dict, path: str, device: str) -> bool:
        """Check if we can use the initial checkpoint for reconstruction."""
        # Check if this is not the initial checkpoint itself
        if "metadata" in checkpoint and checkpoint["metadata"].get("epoch", 0) > 0:
            # Try to find the initial checkpoint in the same directory
            import os

            checkpoint_dir = os.path.dirname(path)
            initial_path = os.path.join(checkpoint_dir, "model_0000.pt")
            return os.path.exists(initial_path)
        return False

    def _reconstruct_with_initial_context(self, checkpoint: dict, path: str, device: str) -> nn.Module:
        """Reconstruct using the initial checkpoint's build context."""
        import os

        # Load the initial checkpoint
        checkpoint_dir = os.path.dirname(path)
        initial_path = os.path.join(checkpoint_dir, "model_0000.pt")

        logger.info(f"Attempting to use initial checkpoint build context from {initial_path}")

        initial_checkpoint = torch.load(initial_path, map_location=device, weights_only=False)

        if "build_context" not in initial_checkpoint:
            raise ValueError("Initial checkpoint also missing build context")

        # Use the initial checkpoint's build context
        build_context = BuildContext.from_dict(initial_checkpoint["build_context"])

        # Import MettaAgentBuilder here to avoid circular imports
        from metta.agent.metta_agent import MettaAgentBuilder

        if self._policy_store is None:
            raise ValueError("PolicyStore is required for reconstruction")

        builder = MettaAgentBuilder(self._policy_store._cfg)

        # Reconstruct using the build context
        if build_context.method == "build_from_brain_policy":
            env = self._reconstruct_env(build_context.env_attributes)
            agent, _ = builder.build_from_brain_policy(env)

            # Activate actions if available in metadata
            if "metadata" in checkpoint and "action_names" in checkpoint["metadata"]:
                action_names = checkpoint["metadata"]["action_names"]
                # Get max_action_args from env attributes
                max_action_args = env.max_action_args if hasattr(env, "max_action_args") else [0] * len(action_names)

                logger.info(f"Activating actions: {action_names}")
                agent.activate_actions(action_names, max_action_args, device)

            # Load the current checkpoint's state dict
            actual_policy = agent.policy if isinstance(agent, MettaAgent) else agent
            actual_policy.load_state_dict(checkpoint["model_state_dict"])

            logger.info("Successfully reconstructed using initial checkpoint's build context")
            return agent
        else:
            raise ValueError(f"Unsupported build method for fallback: {build_context.method}")

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

        # Add build context info if available
        if self._build_context:
            lines.append(f"Build Method: {self._build_context.method}")

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
