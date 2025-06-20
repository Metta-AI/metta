"""
PolicyRecord implementation with robust save/load functionality.

This module implements PolicyRecord which tracks trained policies along with their
construction context, enabling exact reconstruction even when class definitions change.
"""

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
                build_context.source_code = MettaAgent._capture_policy_source_code(actual_policy)

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

        # Handle legacy checkpoints
        if hasattr(self, "_is_legacy_checkpoint") and self._is_legacy_checkpoint:
            logger.warning("Loading legacy checkpoint without build context")
            return self._load_legacy_checkpoint(checkpoint, device)

        # Check if we have build context
        if "build_context" in checkpoint:
            build_context = BuildContext.from_dict(checkpoint["build_context"])
            self._build_context = build_context

            # For BrainPolicy, always try source reconstruction first if available
            if build_context.method == "build_from_brain_policy" and build_context.source_code:
                logger.info("Attempting BrainPolicy reconstruction from saved source code")
                try:
                    policy = self._reconstruct_brain_policy_from_source(build_context, checkpoint, device)
                    logger.info("Successfully reconstructed BrainPolicy from source code")
                    return policy
                except Exception as e:
                    logger.warning(f"Failed to reconstruct from source code: {e}")
                    # Fall through to other methods

            # Try standard reconstruction
            try:
                # Need the policy store's config
                if self._policy_store is None:
                    raise ValueError("PolicyStore is required for reconstruction")
                cfg = self._policy_store._cfg

                # Call the appropriate MettaAgent factory method
                if build_context.method == "build_from_brain_policy":
                    # For BrainPolicy, we need to reconstruct from config
                    # First, try to use the stored configuration
                    if build_context.config:
                        # Create a temporary config object
                        from omegaconf import DictConfig

                        # Use the stored config
                        cfg = DictConfig(build_context.config)

                        # Reconstruct environment from saved attributes
                        env = self._reconstruct_env(build_context.env_attributes)
                        agent, _ = MettaAgent.from_brain_policy(env, cfg)
                    else:
                        # Fallback to old method if no config stored
                        env = self._reconstruct_env(build_context.env_attributes)
                        agent, _ = MettaAgent.from_brain_policy(env, cfg)

                elif build_context.method == "build_from_pytorch_policy":
                    # For pytorch policies, we use the checkpoint path
                    agent, _ = MettaAgent.from_pytorch_policy(path, device=device, pytorch_cfg=cfg.get("pytorch", None))

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
                    agent, _ = MettaAgent.from_policy_class(policy_class, *build_context.args, **constructor_kwargs)

                else:
                    raise ValueError(f"Unknown build method: {build_context.method}")

                # Load the state dict
                actual_policy = agent.policy if isinstance(agent, MettaAgent) else agent
                actual_policy.load_state_dict(checkpoint["model_state_dict"])

                logger.info(f"Successfully reconstructed policy using {build_context.method}")
                return agent

            except Exception as e:
                logger.error(f"Failed to reconstruct from build context: {e}")

                # Try initial checkpoint fallback if available
                if self._try_initial_checkpoint_fallback(checkpoint, path, device):
                    logger.info("Attempting to reconstruct using initial checkpoint")
                    return self._reconstruct_with_initial_context(checkpoint, path, device)

        # For old checkpoints, try to reconstruct from metadata
        if "metadata" in checkpoint and "reconstruction_attributes" in checkpoint["metadata"]:
            try:
                if self._policy_store is None:
                    raise ValueError("PolicyStore is required for reconstruction")

                # Try to reconstruct as BrainPolicy using reconstruction_attributes
                env_attrs = checkpoint["metadata"]["reconstruction_attributes"]
                if env_attrs:
                    env = self._reconstruct_env(env_attrs)
                    agent, _ = MettaAgent.from_brain_policy(env, self._policy_store._cfg)

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

    def _reconstruct_brain_policy_from_source(
        self, build_context: BuildContext, checkpoint: dict, device: str
    ) -> nn.Module:
        """Reconstruct BrainPolicy from saved source code and configuration."""
        import importlib.util

        # Save current modules to restore later
        saved_modules = {}
        modules_to_save = [name for name in sys.modules.keys() if "metta.agent" in name]
        for module_name in modules_to_save:
            saved_modules[module_name] = sys.modules.get(module_name)

        # Create temporary modules from source code
        temp_modules = {}

        try:
            # First, remove current metta.agent modules to avoid conflicts
            for module_name in modules_to_save:
                if module_name in sys.modules:
                    del sys.modules[module_name]

            # Create all modules from source in memory
            for module_name, source_code in build_context.source_code.items():
                # Create a module spec
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)

                # Add common imports to the module's namespace
                module.__dict__["torch"] = torch
                module.__dict__["nn"] = nn
                module.__dict__["Optional"] = Optional
                module.__dict__["Union"] = Union
                module.__dict__["List"] = list
                module.__dict__["Dict"] = dict
                module.__dict__["Tuple"] = tuple
                module.__dict__["Any"] = type
                module.__dict__["gym"] = __import__("gymnasium")
                module.__dict__["gymnasium"] = __import__("gymnasium")
                module.__dict__["hydra"] = __import__("hydra")
                module.__dict__["logging"] = __import__("logging")
                module.__dict__["numpy"] = __import__("numpy")
                module.__dict__["einops"] = __import__("einops")

                # Import TensorDict if available
                try:
                    module.__dict__["TensorDict"] = __import__("tensordict").TensorDict
                except ImportError:
                    pass

                # Import PolicyState if this is a module that needs it
                if "brain_policy" in module_name or "lib" in module_name:
                    try:
                        # Use the saved PolicyState if available
                        if "metta.agent.policy_state" in build_context.source_code:
                            # Execute PolicyState module first
                            ps_spec = importlib.util.spec_from_loader("metta.agent.policy_state", loader=None)
                            ps_module = importlib.util.module_from_spec(ps_spec)
                            ps_module.__dict__.update(module.__dict__)  # Copy common imports
                            exec(build_context.source_code["metta.agent.policy_state"], ps_module.__dict__)
                            sys.modules["metta.agent.policy_state"] = ps_module
                            module.__dict__["PolicyState"] = ps_module.PolicyState
                        else:
                            # Fall back to current PolicyState
                            from metta.agent.policy_state import PolicyState

                            module.__dict__["PolicyState"] = PolicyState
                    except Exception:
                        pass

                # Add OmegaConf imports
                try:
                    module.__dict__["OmegaConf"] = __import__("omegaconf").OmegaConf
                    module.__dict__["DictConfig"] = __import__("omegaconf").DictConfig
                except ImportError:
                    pass

                # Remove torch.jit.script decorators from source code to avoid compilation issues
                cleaned_source = source_code
                if "@torch.jit.script" in cleaned_source:
                    lines = cleaned_source.split("\n")
                    cleaned_lines = []
                    for line in lines:
                        if "@torch.jit.script" in line:
                            # Skip the decorator line
                            continue
                        cleaned_lines.append(line)
                    cleaned_source = "\n".join(cleaned_lines)

                # Execute the source code in the module
                exec(cleaned_source, module.__dict__)

                # Register the module
                sys.modules[module_name] = module
                temp_modules[module_name] = module

            # Now that all modules are loaded, we can instantiate BrainPolicy
            if "metta.agent.brain_policy" in temp_modules:
                brain_module = temp_modules["metta.agent.brain_policy"]
                BrainPolicyClass = brain_module.BrainPolicy

                # Get the environment attributes
                env_attrs = build_context.env_attributes

                # Create the observation space
                import gymnasium as gym
                import numpy as np

                obs_space = gym.spaces.Dict(
                    {
                        "grid_obs": env_attrs.get("single_observation_space"),
                        "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
                    }
                )

                # Extract the agent config from the full config
                agent_config = build_context.config.get("agent", {})

                # Remove the _target_ field if present
                if "_target_" in agent_config:
                    agent_config = dict(agent_config)
                    del agent_config["_target_"]

                # Create BrainPolicy with the stored configuration
                brain = BrainPolicyClass(
                    obs_space=obs_space,
                    obs_width=env_attrs.get("obs_width", 11),
                    obs_height=env_attrs.get("obs_height", 11),
                    action_space=env_attrs.get("single_action_space"),
                    feature_normalizations=env_attrs.get("feature_normalizations", {}),
                    device=device,
                    **agent_config,
                )

                # Activate actions if available
                if "action_names" in env_attrs and env_attrs["action_names"]:
                    action_names = env_attrs["action_names"]
                    max_action_args = env_attrs.get("max_action_args", [0] * len(action_names))
                    brain.activate_actions(action_names, max_action_args, device)

                # Load the state dict
                brain.load_state_dict(checkpoint["model_state_dict"])

                # Wrap in MettaAgent - always use the current MettaAgent class
                # to ensure isinstance checks work correctly
                from metta.agent.metta_agent import MettaAgent

                return MettaAgent(brain)

            else:
                raise ValueError("BrainPolicy source code not found in build context")

        finally:
            # Clean up temporary modules
            for module_name in temp_modules:
                if module_name in sys.modules:
                    del sys.modules[module_name]

            # Restore original modules
            for module_name, module in saved_modules.items():
                if module is not None:
                    sys.modules[module_name] = module

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

        if self._policy_store is None:
            raise ValueError("PolicyStore is required for reconstruction")

        # Reconstruct using the build context
        if build_context.method == "build_from_brain_policy":
            env = self._reconstruct_env(build_context.env_attributes)
            agent, _ = MettaAgent.from_brain_policy(env, self._policy_store._cfg)

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

    def _load_legacy_checkpoint(self, checkpoint: dict, device: str) -> nn.Module:
        """Load a very old checkpoint that doesn't have proper metadata."""
        logger.warning("Loading legacy checkpoint - attempting best effort reconstruction")

        # Try to load the model state dict directly into current code
        try:
            if self._policy_store is None:
                raise ValueError("PolicyStore is required for reconstruction")

            # Try to extract any useful info from the checkpoint
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Look for clues about the model architecture
            if "policy.components._core_._net.weight_ih_l0" in state_dict:
                # This looks like a BrainPolicy
                logger.info("Detected BrainPolicy structure in legacy checkpoint")

                # Create a default environment
                from types import SimpleNamespace

                import gymnasium as gym

                env = SimpleNamespace(
                    obs_width=11,
                    obs_height=11,
                    single_observation_space=gym.spaces.Box(low=0, high=255, shape=(200, 3), dtype="uint8"),
                    single_action_space=SimpleNamespace(nvec=[9, 10]),
                    feature_normalizations={i: 1.0 for i in range(22)},
                    action_names=["action_" + str(i) for i in range(9)],
                    max_action_args=[9] * 9,
                )

                # Create a config that includes the agent key
                from omegaconf import DictConfig

                # Use the existing config if available, otherwise create a minimal one
                if hasattr(self._policy_store, "_cfg") and self._policy_store._cfg is not None:
                    cfg = self._policy_store._cfg
                else:
                    # Create a minimal config with agent key
                    cfg = DictConfig(
                        {
                            "device": device,
                            "agent": {
                                "_target_": "metta.agent.brain_policy.BrainPolicy",
                                "hidden_size": 128,
                                "clip_range": 0,
                                "observations": {"obs_key": "grid_obs"},
                                "components": {},  # Will use defaults
                            },
                        }
                    )

                agent, _ = MettaAgent.from_brain_policy(env, cfg)

                # Try to load the state dict
                actual_policy = agent.policy if hasattr(agent, "policy") else agent
                actual_policy.load_state_dict(state_dict, strict=False)

                logger.info("Loaded legacy checkpoint into current BrainPolicy architecture")
                return agent
            else:
                # Unknown architecture - create a simple wrapper
                logger.warning("Unknown architecture in legacy checkpoint")

                class LegacyPolicy(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # This will be populated by the state dict

                policy = LegacyPolicy()
                policy.load_state_dict(state_dict, strict=False)

                from metta.agent.metta_agent import MettaAgent

                return MettaAgent(policy)

        except Exception as e:
            logger.error(f"Failed to load legacy checkpoint: {e}")
            raise ValueError(f"Cannot load legacy checkpoint: {e}")

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
