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
from metta.rl.policy import PytorchAgent
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

    def __setstate__(self, state):
        """Custom unpickling to ensure backward compatibility."""
        # First restore the state normally
        self.__dict__.update(state)

        # Ensure model_type exists for old checkpoints
        if not hasattr(self, "model_type"):
            logger.warning("Loading old MettaAgent without model_type, defaulting to 'brain'")
            self.model_type = "brain"

        # Ensure metadata is a dict
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}

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

    def save(self, path: str, full_model: bool = False) -> None:
        """
        Save the agent to disk.

        Args:
            path: Path to save the checkpoint
            full_model: If True, saves the complete model object for training resumption.
                       If False, saves only state_dict for inference/evaluation.
        """
        if full_model:
            # Save complete model object for training resumption
            save_data = {
                "model": self.model,
                "model_type": self.model_type,
                "name": self.name,
                "uri": self.uri,
                "metadata": self.metadata,
                "checkpoint_format_version": 2,
            }
            logger.info(f"Saved complete {self.model_type} agent for training to {path}")
        else:
            # Save state_dict for inference/evaluation
            save_data = {
                "model_state_dict": self.model.state_dict() if self.model else None,
                "model_type": self.model_type,
                "name": self.name,
                "uri": self.uri,
                "metadata": self.metadata,
                "checkpoint_format_version": 2,
            }

            # Save model-specific config for reconstruction
            if self.model_type == "brain" and self.model is not None:
                if hasattr(self.model, "agent_attributes"):
                    save_data["agent_attributes"] = self.model.agent_attributes
                if hasattr(self.model, "_component_config"):
                    save_data["component_config"] = self.model._component_config

            logger.info(f"Saved {self.model_type} agent to {path}")

        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", full_model: bool = False) -> "MettaAgent":
        """
        Load an agent from disk.

        Args:
            path: Path to load the checkpoint from
            device: Device to load the model to
            full_model: If True, expects a checkpoint with complete model object.
                       If False, expects state_dict and will reconstruct the model.
        """
        logger.info(f"Loading agent from {path} (full_model={full_model})")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Check checkpoint format version
        format_version = checkpoint.get("checkpoint_format_version", 1)
        if format_version > 2:
            logger.warning(
                f"Checkpoint has format version {format_version}, but this code supports up to version 2. "
                "Some features may not work correctly."
            )

        # Create MettaAgent instance
        agent = cls(
            model=None,
            model_type=checkpoint.get("model_type", "brain"),
            name=checkpoint.get("name", os.path.basename(path)),
            uri=checkpoint.get("uri", f"file://{path}"),
            metadata=checkpoint.get("metadata", {}),
            local_path=path,
        )

        if full_model:
            # Load complete model object
            model = checkpoint.get("model")
            if model is not None:
                model = model.to(device)
            agent.model = model

            if model is None:
                logger.warning("No model found in checkpoint (expected for full_model=True)")
        else:
            # Reconstruct model from state_dict
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

        if agent.model is None:
            logger.warning("No model loaded from checkpoint")

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
                except (TypeError, ValueError) as err:
                    raise TypeError(f"Cannot convert hidden_size of type {type(hidden_size)} to int") from err
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

    @property
    def action_index_tensor(self):
        """Get the action index tensor from the underlying model."""
        if self.model and hasattr(self.model, "action_index_tensor"):
            return self.model.action_index_tensor
        raise AttributeError(f"{self.model_type} model does not have action_index_tensor attribute")

    @property
    def cum_action_max_params(self):
        """Get the cumulative action max params from the underlying model."""
        if self.model and hasattr(self.model, "cum_action_max_params"):
            return self.model.cum_action_max_params
        raise AttributeError(f"{self.model_type} model does not have cum_action_max_params attribute")

    @property
    def active_actions(self):
        """Get the active actions from the underlying model."""
        if self.model and hasattr(self.model, "active_actions"):
            return self.model.active_actions
        raise AttributeError(f"{self.model_type} model does not have active_actions attribute")

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Delegate action conversion to the underlying model."""
        if self.model and hasattr(self.model, "_convert_action_to_logit_index"):
            return self.model._convert_action_to_logit_index(flattened_action)
        raise AttributeError(f"{self.model_type} model does not have _convert_action_to_logit_index method")

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Delegate logit index conversion to the underlying model."""
        if self.model and hasattr(self.model, "_convert_logit_index_to_action"):
            return self.model._convert_logit_index_to_action(action_logit_index)
        raise AttributeError(f"{self.model_type} model does not have _convert_logit_index_to_action method")


class DistributedMettaAgent(DistributedDataParallel):
    """Distributed wrapper for MettaAgent that properly delegates method calls."""

    def __init__(self, agent: MettaAgent, device):
        super().__init__(agent, device_ids=[device])
        self._wrapped_agent = agent

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped MettaAgent."""
        if hasattr(self._wrapped_agent, name):
            return getattr(self._wrapped_agent, name)
        return super().__getattr__(name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        """Delegate action activation to the wrapped agent."""
        self._wrapped_agent.activate_actions(action_names, action_max_params, device)

    def key_and_version(self) -> tuple[str, int]:
        """Delegate key_and_version to the wrapped agent."""
        return self._wrapped_agent.key_and_version()

    def policy_as_metta_agent(self) -> MettaAgent:
        """Return the wrapped MettaAgent."""
        return self._wrapped_agent
