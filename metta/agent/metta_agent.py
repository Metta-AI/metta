import logging
import os
from typing import Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.brain_policy import BrainPolicy
from metta.agent.policy_state import PolicyState
from metta.rl.pufferlib.policy import PytorchAgent
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig):
    """Create a MettaAgent policy for the given environment."""
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Create BrainPolicy (the actual neural network)
    brain_policy = hydra.utils.instantiate(
        cfg.agent,
        obs_space=obs_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        action_space=env.single_action_space,
        feature_normalizations=env.feature_normalizations,
        global_features=env.global_features,
        device=cfg.device,
        _recursive_=False,
        _target_="metta.agent.brain_policy.BrainPolicy",  # Override target to use BrainPolicy
    )

    # Wrap in MettaAgent
    return MettaAgent(
        model=brain_policy,
        model_type="brain",
        metadata={
            "action_names": env.action_names,
            "observation_space": obs_space,
            "action_space": env.single_action_space,
        },
    )


class MettaAgent(nn.Module):
    """
    Wrapper class for all policy models (BrainPolicy, PytorchPolicy).
    This class combines the functionality of the old PolicyRecord with the model itself.
    """

    def __init__(
        self,
        model: Optional[Union[BrainPolicy, PytorchAgent]] = None,
        model_type: str = "brain",
        name: str = "",
        uri: str = "",
        metadata: Optional[dict] = None,
        local_path: Optional[str] = None,
    ):
        super().__init__()
        self.model: Optional[Union[BrainPolicy, PytorchAgent]] = model
        self.model_type = model_type
        self.name = name
        self.uri = uri
        self.metadata = metadata or {}
        self.local_path = local_path

        # Placeholder for future versioning
        self.observation_space_version = None
        self.action_space_version = None
        self.layer_version = None

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the underlying model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model(x, state, action)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Activate actions on the underlying model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        self.model.activate_actions(action_names, action_max_params, device)

    def parameters(self, recurse: bool = True):
        """Return model parameters."""
        if self.model is None:
            return iter([])
        return self.model.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        """Return model state_dict."""
        if self.model is None:
            return {}
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Load model state_dict."""
        if self.model is None:
            raise RuntimeError("Model not loaded, cannot load state_dict.")
        return self.model.load_state_dict(*args, **kwargs)

    def num_params(self) -> int:
        """Get number of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save the agent to disk."""
        save_data = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "model_type": self.model_type,
            "name": self.name,
            "uri": self.uri,
            "metadata": self.metadata,
            "observation_space_version": self.observation_space_version,
            "action_space_version": self.action_space_version,
            "layer_version": self.layer_version,
        }

        # Save model-specific config for reconstruction
        if self.model_type == "brain" and self.model is not None:
            # For BrainPolicy, save all initialization parameters
            if hasattr(self.model, "agent_attributes"):
                save_data["agent_attributes"] = self.model.agent_attributes

            # Also save the component configuration if available
            if hasattr(self.model, "_component_config"):
                save_data["component_config"] = self.model._component_config

        torch.save(save_data, path)
        logger.info(f"Saved {self.model_type} agent to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MettaAgent":
        """Load an agent from disk."""
        logger.info(f"Loading agent from {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Create MettaAgent instance
        agent = cls(
            model=None,
            model_type=checkpoint.get("model_type", "brain"),
            name=checkpoint.get("name", os.path.basename(path)),
            uri=checkpoint.get("uri", f"file://{path}"),
            metadata=checkpoint.get("metadata", {}),
            local_path=path,
        )

        # Set version info
        agent.observation_space_version = checkpoint.get("observation_space_version")
        agent.action_space_version = checkpoint.get("action_space_version")
        agent.layer_version = checkpoint.get("layer_version")

        # Reconstruct the model
        model_state_dict = checkpoint.get("model_state_dict")
        if model_state_dict is not None and agent.model_type == "brain":
            # For BrainPolicy, we need the agent_attributes to reconstruct
            agent_attributes = checkpoint.get("agent_attributes")
            if agent_attributes and "components" in agent_attributes:
                try:
                    # Import and create BrainPolicy with saved attributes
                    from metta.agent.brain_policy import BrainPolicy

                    # Extract necessary attributes
                    obs_space = agent.metadata.get("observation_space")
                    action_space = agent.metadata.get("action_space")

                    if obs_space and action_space:
                        # Create a copy of agent_attributes to avoid modifying the original
                        brain_attrs = agent_attributes.copy()

                        # Ensure we don't duplicate arguments
                        brain_attrs["obs_space"] = obs_space
                        brain_attrs["action_space"] = action_space
                        brain_attrs["device"] = device

                        # Remove any keys that might conflict
                        for key in ["obs_space", "action_space", "device"]:
                            if key in brain_attrs and key != key:  # Don't remove the ones we just set
                                del brain_attrs[key]

                        # Reconstruct the BrainPolicy
                        brain_policy = BrainPolicy(**brain_attrs)

                        # Load the state dict
                        brain_policy.load_state_dict(model_state_dict)
                        brain_policy.to(device)
                        agent.model = brain_policy

                        logger.info(f"Successfully loaded BrainPolicy model from {path}")
                    else:
                        logger.warning("Missing observation_space or action_space in metadata")
                except Exception as e:
                    logger.error(f"Failed to reconstruct BrainPolicy: {e}")
                    logger.warning("Model will be None - this checkpoint may need migration")
            else:
                logger.warning("Checkpoint does not contain component configuration needed for full reconstruction")
                logger.info(
                    "For evaluation purposes, you can use the checkpoint with the same code version that created it"
                )
        else:
            if model_state_dict is None:
                logger.warning("No model state dict found in checkpoint")
            else:
                logger.warning(f"Model reconstruction for type '{agent.model_type}' not implemented")

        if model_state_dict is None:
            logger.warning("No model found in checkpoint")

        return agent

    def __repr__(self):
        """String representation with model details."""
        lines = [f"MettaAgent(type={self.model_type}, name={self.name})"]

        if self.uri:
            lines.append(f"URI: {self.uri}")

        # Add key metadata
        important_keys = ["epoch", "agent_step", "generation", "score"]
        metadata_items = []
        for k in important_keys:
            if k in self.metadata:
                metadata_items.append(f"{k}={self.metadata[k]}")

        if metadata_items:
            lines.append(f"Metadata: {', '.join(metadata_items)}")

        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            lines.append(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")

        return "\n".join(lines)

    # Delegate common methods to the underlying model
    @property
    def device(self):
        if self.model and hasattr(self.model, "device"):
            return self.model.device
        return None

    @property
    def hidden_size(self) -> int:
        """Get the hidden size from the underlying model."""
        if self.model and hasattr(self.model, "hidden_size"):
            hidden_size = self.model.hidden_size
            if isinstance(hidden_size, int):
                return hidden_size
            elif torch.is_tensor(hidden_size):
                # For properties that might return tensors, get the scalar value
                return int(hidden_size.item())
            else:
                # Try to convert to int directly, handling Module case
                try:
                    return int(hidden_size)
                except (TypeError, ValueError):
                    raise TypeError(f"Cannot convert hidden_size of type {type(hidden_size)} to int")
        raise AttributeError(f"{self.model_type} model does not have hidden_size attribute")

    @property
    def lstm(self):
        """Get the LSTM layer from the underlying model."""
        if self.model and hasattr(self.model, "lstm"):
            return self.model.lstm
        # For BrainPolicy, try to access through components
        if self.model_type == "brain" and self.model is not None:
            if hasattr(self.model, "components"):
                components = self.model.components
                if isinstance(components, nn.ModuleDict) and "_core_" in components:
                    component = components["_core_"]
                    if hasattr(component, "_net"):
                        return component._net
        raise AttributeError(f"{self.model_type} model does not have lstm attribute")

    @property
    def total_params(self):
        if self.model and hasattr(self.model, "total_params"):
            return self.model.total_params
        return self.num_params()

    def l2_reg_loss(self) -> torch.Tensor:
        if self.model:
            return self.model.l2_reg_loss()
        return torch.zeros(1)

    def l2_init_loss(self) -> torch.Tensor:
        if self.model:
            return self.model.l2_init_loss()
        return torch.zeros(1)

    def update_l2_init_weight_copy(self):
        if self.model:
            self.model.update_l2_init_weight_copy()

    def clip_weights(self):
        if self.model:
            self.model.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        if self.model:
            return self.model.compute_weight_metrics(delta)
        return []

    def key_and_version(self) -> tuple[str, int]:
        """
        Extract the policy key and version from the URI.

        Returns:
            tuple: (policy_key, version)
                - policy_key is the clean name without path or version
                - version is the numeric version or 0 if not present
        """
        # Get the last part after splitting by slash
        base_name = self.uri.split("/")[-1] if self.uri else self.name

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
        return self.key_and_version()[0]

    def version(self) -> int:
        return self.key_and_version()[1]

    def policy(self) -> Union[BrainPolicy, PytorchAgent]:
        """
        Get the underlying policy model.

        This method exists for backward compatibility with code that expects
        a policy() method. New code should use self.model directly.
        """
        if self.model is None:
            raise ValueError("No model loaded in MettaAgent")
        return self.model

    def policy_as_metta_agent(self) -> "MettaAgent":
        """Return self since we're already a MettaAgent."""
        return self

    @property
    def components(self):
        """
        Get the components from the underlying model.
        Returns an empty dict if the model is not component-based (e.g., PytorchAgent).
        """
        if self.model and hasattr(self.model, "components"):
            return self.model.components
        return {}

    def save_for_training(self, path: str) -> None:
        """
        Save the agent for training resumption.
        This saves the complete model object, allowing training to resume without reconstruction.
        """
        save_data = {
            "model": self.model,  # Save the full model object
            "model_type": self.model_type,
            "name": self.name,
            "uri": self.uri,
            "metadata": self.metadata,
            "observation_space_version": self.observation_space_version,
            "action_space_version": self.action_space_version,
            "layer_version": self.layer_version,
        }

        torch.save(save_data, path)
        logger.info(f"Saved complete {self.model_type} agent for training to {path}")

    @classmethod
    def load_for_training(cls, path: str, device: str = "cpu") -> "MettaAgent":
        """
        Load an agent for training resumption.
        This loads the complete model object without needing reconstruction.
        """
        logger.info(f"Loading agent for training from {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Create MettaAgent instance with the loaded model
        model = checkpoint.get("model")
        if model is not None:
            model = model.to(device)

        agent = cls(
            model=model,
            model_type=checkpoint.get("model_type", "brain"),
            name=checkpoint.get("name", os.path.basename(path)),
            uri=checkpoint.get("uri", f"file://{path}"),
            metadata=checkpoint.get("metadata", {}),
            local_path=path,
        )

        # Set version info
        agent.observation_space_version = checkpoint.get("observation_space_version")
        agent.action_space_version = checkpoint.get("action_space_version")
        agent.layer_version = checkpoint.get("layer_version")

        if model is None:
            logger.warning("No model found in checkpoint")

        return agent


class DistributedMettaAgent(DistributedDataParallel):
    """Distributed wrapper for MettaAgent that preserves the interface."""

    def __init__(self, agent: MettaAgent, device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")
        agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        return self.module.activate_actions(action_names, action_max_params, device)
