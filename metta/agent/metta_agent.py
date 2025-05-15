import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.metta_layer import LayerBase
from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import sample_logits
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.util.omegaconf import convert_to_dict
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_policy(env: MettaGridEnv, cfg: Union[ListConfig, DictConfig]) -> "MettaAgent":
    """
    Create a policy instance based on environment and configuration.

    This factory function instantiates a MettaAgent for the given MettaGrid environment
    using the provided configuration parameters. It creates the necessary observation
    space wrapper that combines grid observations with global variables.

    Args:
        env: The MettaGrid environment that the agent will interact with
        cfg: Configuration parameters containing agent architecture settings
             and hyperparameters under the 'agent' key

    Returns:
        An initialized MettaAgent policy ready to process observations and generate actions
    """
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    return hydra.utils.instantiate(
        cfg.agent,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        device=cfg.device,
        _recursive_=False,
    )


class DistributedMettaAgent(DistributedDataParallel):
    """
    Wrapper for MettaAgent to support distributed training with PyTorch DDP.

    Extends DistributedDataParallel to maintain access to the underlying MettaAgent
    methods and properties that are needed during training but aren't part of the
    standard PyTorch module interface.

    This enables transparent use of a distributed agent while preserving the
    full MettaAgent interface for training and evaluation.
    """

    def __init__(self, agent: "MettaAgent", device: Union[torch.device, int]) -> None:
        """
        Initialize a distributed wrapper for MettaAgent.

        Args:
            agent: The MettaAgent to be wrapped for distributed training
            device: The device to use for distributed processing (GPU index or torch device)
        """
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the wrapped MettaAgent module.

        Allows transparent access to MettaAgent methods not defined in DistributedDataParallel.

        Args:
            name: Name of the attribute to access

        Returns:
            The requested attribute from the wrapped module

        Raises:
            AttributeError: If the attribute is not found in either class
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: List[str], action_max_params: List[int], device: torch.device) -> None:
        """
        Forward activate_actions to the wrapped module.

        Args:
            action_names: List of action names to activate
            action_max_params: List of maximum parameter values for each action
            device: Device for tensor operations
        """
        return self.module.activate_actions(action_names, action_max_params, device)

    @property
    def components(self) -> nn.ModuleDict:
        """
        Access the components dictionary from the wrapped module.

        Returns:
            ModuleDict containing all neural network components of the agent
        """
        return self.module.components

    def update_l2_init_weight_copy(self) -> None:
        """
        Forward update_l2_init_weight_copy to the wrapped module.

        Updates the saved copies of weights used for L2 initialization regularization.
        """
        return self.module.update_l2_init_weight_copy()

    def l2_reg_loss(self) -> torch.Tensor:
        """
        Forward l2_reg_loss to the wrapped module.

        Returns:
            Scalar tensor containing the L2 regularization loss
        """
        return self.module.l2_reg_loss()

    def l2_init_loss(self) -> torch.Tensor:
        """
        Forward l2_init_loss to the wrapped module.

        Returns:
            Scalar tensor containing the L2 initialization loss
        """
        return self.module.l2_init_loss()

    def clip_weights(self) -> None:
        """
        Forward clip_weights to the wrapped module.

        Applies weight clipping to all components if enabled.
        """
        return self.module.clip_weights()


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        action_space: gym.spaces.Space,
        grid_features: List[str],
        device: str,
        **cfg,
    ):
        """
        Initialize the MettaAgent.

        Sets up the agent's neural network components based on the provided configuration,
        initializes action spaces, and establishes connections between components.

        Note on tensor formats:
        - Observation tensors ("x") are in batch-first format [batch_size, ...]
        - LSTM states (lstm_h, lstm_c) are in layer-first format [num_layers, batch_size, hidden_size]
        for direct compatibility with PyTorch's LSTM module

        Args:
            obs_space: Observation space definition specifying the format of environment observations
            action_space: Action space definition specifying possible agent actions
            grid_features: List of grid feature names available in the observation
            device: Device to run the agent on (e.g., 'cuda:0', 'cpu')
            **cfg: Additional configuration parameters including:
                - components: Neural network component definitions
                - observations.obs_key: Key for accessing observations in the observation space
                - clip_range: Weight clipping threshold for regularization

        Raises:
            RuntimeError: If any component is not properly set up
            AssertionError: If required configuration fields are missing
        """
        super().__init__()
        cfg = OmegaConf.create(cfg)

        logger.info(f"obs_space: {obs_space} ")

        self.hidden_size: int = cfg.components._core_.output_size
        self.core_num_layers: int = cfg.components._core_.nn_params.num_layers
        self.clip_range: float = cfg.clip_range
        self.device: Optional[torch.device] = None
        self.action_max_params: Optional[List[int]] = None
        self.action_names: Optional[List[str]] = None
        self.active_actions: Optional[List[Tuple[str, int]]] = None
        self.cum_action_max_params: Optional[torch.Tensor] = None  # Shape: [num_action_types + 1]
        self.action_index_tensor: Optional[torch.Tensor] = None  # Shape: [total_num_actions, 2]

        assert hasattr(cfg.observations, "obs_key") and cfg.observations.obs_key is not None, (
            "Configuration is missing required field 'observations.obs_key'"
        )
        obs_key: str = cfg.observations.obs_key  # typically "grid_obs"

        obs_shape: Tuple[int, ...] = safe_get_from_obs_space(obs_space, obs_key, "shape")
        obs_input_shape: Tuple[int, ...] = obs_shape[1:]  # typ. obs_width, obs_height, number of observations
        num_objects: int = obs_shape[2]  # typ. number of observations

        agent_attributes: Dict[str, Any] = {
            "obs_shape": obs_shape,
            "clip_range": self.clip_range,
            "action_space": action_space,
            "grid_features": grid_features,
            "obs_key": cfg.observations.obs_key,
            "obs_input_shape": obs_input_shape,
            "num_objects": num_objects,
            "hidden_size": self.hidden_size,
            "core_num_layers": self.core_num_layers,
        }

        logging.info(f"agent_attributes: {agent_attributes}")

        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

        self.components = nn.ModuleDict()
        component_cfgs = convert_to_dict(cfg.components)

        for component_key in component_cfgs:
            # Convert key to string to ensure compatibility
            component_name: str = str(component_key)
            component_cfgs[component_key]["name"] = component_name
            logger.info(f"calling hydra instantiate from MettaAgent __init__ for {component_name}")
            component = hydra.utils.instantiate(component_cfgs[component_key], **agent_attributes)
            self.components[component_name] = component

        value_component: LayerBase = cast(LayerBase, self.components["_value_"])
        self._setup_components(value_component)

        action_component: LayerBase = cast(LayerBase, self.components["_action_"])
        self._setup_components(action_component)

        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(
                    f"Component {name} in MettaAgent was never setup. It might not be accessible by other components."
                )

        self.components = self.components.to(device)

        self._total_params: int = sum(p.numel() for p in self.parameters())
        logger.info(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")

    def _setup_components(self, component: LayerBase) -> None:
        """
        Recursively setup components and establish connections between them.

        Walks through the component dependency graph to ensure that all
        components are properly set up before they are used as inputs by
        other components.

        Args:
            component: The component to set up

        Note:
            Component dependencies are specified through the _sources attribute:
            1. Each component may have a `_sources` attribute (List[Dict]) defining its input layers
            2. Each dict in `_sources` contains a "name" key matching a component in self.components
            3. A dictionary of source components is built and passed to the component's setup method
            4. The setup process is recursive, ensuring all dependencies are set up first
        """
        # recursively setup all source components
        if component._sources is not None:
            for source in component._sources:
                source_component: LayerBase = cast(LayerBase, self.components[source["name"]])
                self._setup_components(source_component)

        # setup the current component and pass in the source components
        source_components = None
        if component._sources is not None:
            source_components = {}
            for source in component._sources:
                source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def activate_actions(self, action_names: List[str], action_max_params: List[int], device: torch.device) -> None:
        """
        Activate agent actions for training.

        Configures the action space by setting up action names, their parameter ranges,
        and precomputing data structures for efficient action conversion.

        Args:
            action_names: List of action names to activate (e.g., ["move", "turn"])
            action_max_params: List of maximum parameter values for each action
                              (e.g., [4, 1] for up to 5 move parameters and 2 turn parameters)
            device: Device for tensor operations

        Raises:
            TypeError: If component '_action_embeds_' is not an ActionEmbedding
            AssertionError: If action_max_params is not a list
        """
        assert isinstance(action_max_params, list), "action_max_params must be a list"

        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names

        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        # Shape: [num_action_types + 1]
        self.cum_action_max_params = torch.cumsum(torch.tensor([0] + action_max_params, device=self.device), dim=0)

        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        component = self.components["_action_embeds_"]

        if not isinstance(component, ActionEmbedding):
            raise TypeError(f"Component '_action_embeds_' is of type {type(component)}, expected ActionEmbedding")

        component.activate_actions(full_action_names, device)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        # Shape: [total_num_actions, 2]
        self.action_index_tensor = torch.tensor(action_index, device=self.device)
        logger.info(f"Agent actions activated with: {self.active_actions}")

    @property
    def lstm(self) -> nn.Module:
        """
        Get the LSTM core module.

        Returns:
            The LSTM network used as the agent's core recurrent component

        Raises:
            TypeError: If _core_._net is not an nn.Module
        """
        core_net = self.components["_core_"]._net
        if not isinstance(core_net, nn.Module):
            raise TypeError(f"Expected core_net to be nn.Module, got {type(core_net)}")
        return core_net

    @property
    def total_params(self) -> int:
        """
        Get the total number of parameters in the model.

        Returns:
            Integer count of all trainable parameters in the agent
        """
        return self._total_params

    def forward(
        self,
        x: torch.Tensor,  # Shape: [B, T, *obs_shape] or [B*T, *obs_shape]
        state: PolicyState,
        action: Optional[torch.Tensor] = None,  # Shape: [B, T, 2] or [B*T, 2] if provided
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MettaAgent.

        Processes observations through the agent's neural network architecture to produce
        action distributions, value estimates, and updated internal states.

        Args:
            x: Input observation tensor of shape [B, T, *obs_shape] or [B*T, *obs_shape]
            where B=batch_size, T=sequence_length
            state: Policy state containing LSTM hidden states (lstm_h) and cell states (lstm_c)
                in layer-first format [num_layers, batch_size, hidden_size]
            action: Optional action tensor of shape [B, T, 2] or [B*T, 2] where each action
                    is represented as [action_type, action_param].

        Returns:
            Tuple containing:
            - action: Tensor of shape [B*T, 2], where each row is [action_type, action_param]
            - logprob_act: Log probability of chosen action, shape [B*T]
            - entropy: Entropy of action distribution, shape [B*T]
            - value: Value function output, shape [B*T]
            - logprobs: All action log probabilities, shape [B*T, num_actions]

        Note:
            If action is None, actions will be sampled from the predicted distribution.
            The LSTM state in the provided PolicyState object is updated in-place.
        """
        # Initialize TensorDict for data flow
        td = TensorDict({}, batch_size=x.shape[0])
        td["x"] = x

        # Pass LSTM states directly - now in layer-first format
        if state.lstm_h is not None and state.lstm_c is not None:
            # Ensure states are on the same device as input
            td["lstm_h"] = state.lstm_h.to(x.device)  # [num_layers, batch_size, hidden_size]
            td["lstm_c"] = state.lstm_c.to(x.device)  # [num_layers, batch_size, hidden_size]

        # Forward pass through value network
        self.components["_value_"](td)
        value = td["_value_"]  # Shape: [B*T]

        # Forward pass through action network
        self.components["_action_"](td)
        logits = td["_action_"]  # Shape: [B*T, num_actions]

        # Update LSTM states directly - already in layer-first format
        state.lstm_h = td["lstm_h"]  # [num_layers, batch_size, hidden_size]
        state.lstm_c = td["lstm_c"]  # [num_layers, batch_size, hidden_size]

        # Sample actions
        action_logit_index: Optional[torch.Tensor] = (
            self._convert_action_to_logit_index(action) if action is not None else None
        )

        # Sample_logits returns (action_indices, log_probs, entropy, all_log_probs)
        action_indices, log_probs, entropy_val, all_log_probs = sample_logits(logits, action_logit_index)

        # Convert logit index to action if no action was provided
        if action is None:
            action = self._convert_logit_index_to_action(action_indices)  # Shape: [B*T, 2]

        return action, log_probs, entropy_val, value, all_log_probs

    def _convert_action_to_logit_index(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete action indices.

        Maps the two-element action representation to a single flattened index
        that can be used to index into logits or other flat action representations.

        Args:
            action: Tensor of shape [B, T, 2] or [B*T, 2] where each action is
                   represented as [action_type, action_param]

        Returns:
            action_logit_index: Tensor of shape [B*T] containing flattened action
                               indices for logit indexing

        Raises:
            ValueError: If agent actions have not been activated first

        Note:
            Uses precomputed cumulative sums (self.cum_action_max_params) for efficiency.
        """
        if self.cum_action_max_params is None:
            raise ValueError("Agent actions have not been activated. Call activate_actions first.")

        action = action.reshape(-1, 2)

        # Extract action components
        action_type_numbers = action[:, 0].long()  # Shape: [B*T]
        action_params = action[:, 1].long()  # Shape: [B*T]

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]  # Shape: [B*T]

        # Vectorized addition
        action_logit_index = action_type_numbers + cumulative_sum + action_params  # Shape: [B*T]

        return action_logit_index.view(-1)  # shape: [B*T]

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs using tensor indexing.

        Converts from flattened action indices to the structured representation of
        [action_type, action_param] pairs used by the environment.

        Args:
            action_logit_index: Tensor of shape [B*T] containing flattened action indices

        Returns:
            action: Tensor of shape [B*T, 2] where each row is [action_type, action_param]

        Raises:
            ValueError: If agent actions have not been activated first
        """
        if self.action_index_tensor is None:
            raise ValueError("Agent actions have not been activated. Call activate_actions first.")

        # direct tensor indexing on precomputed action_index_tensor
        return self.action_index_tensor[action_logit_index.reshape(-1)]  # Shape: [B*T, 2]

    def _apply_to_components(self, method_name: str, *args, **kwargs) -> List[torch.Tensor]:
        """
        Apply a method to all components, collecting and returning the results.

        Helper function that calls a named method on all components and collects the results,
        used primarily for regularization and metric computation.

        Args:
            method_name: Name of the method to call on each component
            *args, **kwargs: Arguments to pass to the method

        Returns:
            list: Results from calling the method on each component

        Raises:
            AttributeError: If any component doesn't have the requested method
            TypeError: If a component's method is not callable
            AssertionError: If no components are available
        """
        assert len(self.components) != 0, "No components available to apply method"

        results = []
        for name, component in self.components.items():
            if not hasattr(component, method_name):
                raise AttributeError(f"Component '{name}' does not have method '{method_name}'")

            method = getattr(component, method_name)
            if not callable(method):
                raise TypeError(f"Component '{name}' has {method_name} attribute but it's not callable")

            results.append(method(*args, **kwargs))

        return results

    def l2_reg_loss(self) -> torch.Tensor:
        """
        Calculate L2 regularization loss across all components.

        Aggregates the L2 regularization losses from all network components
        to produce a total regularization term for the loss function.

        Returns:
            Scalar tensor containing the summed L2 regularization loss

        Note:
            L2 regularization loss is controlled by l2_norm_coeff in the configuration.
            Each component can scale its contribution using l2_norm_scale, or set it to 0
            to disable regularization for that component.
        """
        component_loss_tensors = self._apply_to_components("l2_reg_loss")
        return torch.sum(torch.stack(component_loss_tensors))

    def l2_init_loss(self) -> torch.Tensor:
        """
        Calculate L2 initialization loss across all components.

        Computes regularization loss that penalizes deviation from the initial
        weights, which can help prevent catastrophic forgetting in continual learning.

        Returns:
            Scalar tensor containing the summed L2 initialization loss

        Note:
            L2 initialization loss is controlled by l2_init_coeff in the configuration.
            Each component can scale its contribution using l2_init_scale, or set it to 0
            to disable this regularization for that component.
        """
        component_loss_tensors = self._apply_to_components("l2_init_loss")
        return torch.sum(torch.stack(component_loss_tensors))

    def update_l2_init_weight_copy(self) -> None:
        """
        Update the weight copies used for L2 initialization regularization.

        Stores the current weights as the reference point for future L2 initialization
        regularization calculations. This can be called periodically to update
        the target weights.

        Note:
            The update interval is set by l2_init_weight_update_interval in the configuration.
            A value of 0 means no updating will occur.
        """
        self._apply_to_components("update_l2_init_weight_copy")

    def clip_weights(self) -> None:
        """
        Clip weights to stay within the specified range if clip_range > 0.

        Enforces weight constraints by clipping all weights to be within
        [-clip_range, clip_range]. This can improve training stability.

        Note:
            Weight clipping is controlled by clip_range in the configuration.
            Each component can scale its clipping threshold using clip_scale,
            or set it to 0 to disable clipping for that component.
        """
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> List[Dict[str, float]]:
        """
        Compute weight metrics for all components that have weights enabled for analysis.

        Collects weight statistics such as magnitude, sparsity, and changes from
        each component to help monitor training dynamics.

        Args:
            delta: Threshold for weight change metrics, used to determine if a
                  weight has changed significantly

        Returns:
            List of metric dictionaries, one per component that has weight analysis enabled

        Note:
            Weight analysis is enabled by setting analyze_weights=True in a component's
            configuration. Components without this setting will be skipped.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, component in self.components.items():
            method_name = "compute_weight_metrics"
            if not hasattr(component, method_name):
                continue  # Skip components that don't have this method instead of raising an error

            method = getattr(component, method_name)
            assert callable(method), f"Component '{name}' has {method_name} attribute but it's not callable"

            metric_result = method(delta)
            if isinstance(metric_result, dict):
                results[name] = cast(Dict[str, float], metric_result)

        metrics_list = [metrics for metrics in results.values() if metrics is not None]
        return metrics_list
