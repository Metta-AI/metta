from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from metta.agent.util.weights_analysis import analyze_weights


class LayerBase(nn.Module):
    """The base class for components that make up the Metta agent. All components
    are required to have a name and an input source, although the input source
    can be None (or null in your YAML). Output size is optional depending on
    your component.

    All components must have a method called `setup` and it must accept
    input_source_components. The setup assigns the output_size if it is not
    already set. Once it has been called on all components, components can
    initialize their parameters via `_initialize()`, if necessary. All
    components must also have a property called `ready` that returns True if
    the component has been setup.

    The `_forward` method should only take a dict as input and return
    only the dict. The tensor dict is constructed anew each time the
    metta_agent forward pass is run. The component's `_forward` should read
    from the value at the key name of its input_source. After performing its
    computation via self._net() or otherwise, it should store its output in the
    tensor dict at the key with its own name.

    Before doing this, it should first check if the dict already has a
    key with its own name, indicating that its computation has already been
    performed (due to some other run up the DAG) and return. After this check,
    it should check if its input source is not None and recursively call the
    forward method of the layer above it.

    Carefully passing input and output shapes is necessary to setup the agent.
    self._in_tensor_shape and self._out_tensor_shape are always of type list.
    Note that these lists not include the batch dimension so their shape is
    one dimension smaller than the actual shape of the tensor.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    _name: str
    _sources: List[Dict[str, Any]]
    _net: nn.Module
    _ready: bool
    _nn_params: Dict[str, Any]
    _in_tensor_shapes: List[List[int]]
    _out_tensor_shape: List[int]

    def __init__(
        self,
        name: str,
        sources: Optional[List[Union[Dict[str, Any], DictConfig]]] = None,
        nn_params: Optional[Dict[str, Any]] = None,
        **cfg,
    ):
        super().__init__()

        # Extract name from cfg if not provided directly
        if name is None and "name" in cfg:
            name = cfg.pop("name")  # Using pop to remove it from cfg
        self._name = name
        assert self._name, f"Invalid name {name}"

        # Convert from omegaconf's list class if needed
        if sources is None:
            self._sources = []
        else:
            # Convert each source to a standard Python dict
            self._sources = []
            for source in sources:
                # Direct check if it's a DictConfig
                if isinstance(source, DictConfig):
                    # Use to_container() for DictConfig objects
                    container = OmegaConf.to_container(source)
                    python_dict = cast(Dict[str, Any], container)
                else:
                    # For regular dicts or dict-like objects
                    python_dict = dict(source)
                self._sources.append(python_dict)

        # Validate the sources
        for i, source in enumerate(self._sources):
            if not isinstance(source, dict) or "name" not in source:
                raise ValueError(f"Invalid source at index {i}: each source must be a dict with at least a 'name' key")

        self._net: nn.Module = nn.Identity()
        self._ready = False
        if not hasattr(self, "_nn_params"):
            self._nn_params = nn_params if nn_params is not None else {}
        else:
            # If _nn_params already exists, update it with new values if provided
            if nn_params is not None:
                self._nn_params.update(nn_params)

        self._net = None
        self._ready = False

    @property
    def ready(self):
        return self._ready

    def setup(self, source_components: Optional[Dict[str, Any]] = None) -> None:
        if self._ready:
            return

        # Note - when a property is set on a torch nn.Module, it will automatically be
        # added to gradient tracking. This gets around the problem.
        self.__dict__["_source_components"] = source_components

        self._in_tensor_shapes = []

        typed_source_components = cast(Optional[Dict[str, Any]], self._source_components)
        if typed_source_components is not None:
            for _, source in typed_source_components.items():
                self._in_tensor_shapes.append(source._out_tensor_shape.copy())

        self._initialize()
        self._ready = True

    def _initialize(self):
        self._net = self._make_net()

    def _make_net(self) -> nn.Module:
        return nn.Identity()

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._name in data:
            return data

        # recursively call the forward method of the source components
        typed_source_components = cast(Optional[Dict[str, Any]], self._source_components)
        if typed_source_components is not None:
            for _, source in typed_source_components.items():
                source.forward(data)

        self._forward(data)

        return data

    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Components that have more than one input sources must have their own _forward() method."""
        # get the input tensor from the source component by calling its forward method (which recursively calls
        # _forward() on its source components)
        data[self._name] = self._net(data[self._sources[0]["name"]])
        return data

    def clip_weights(self):
        pass

    def l2_reg_loss(self):
        pass

    def l2_init_loss(self):
        pass

    def update_l2_init_weight_copy(self):
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        return []


class LinearWeightModule(nn.Module):
    weight: torch.nn.parameter.Parameter
    bias: Optional[torch.nn.parameter.Parameter]


class ParamLayer(LayerBase):
    """
    Extension of LayerBase that provides parameter management and regularization functionality.

    This class adds weight initialization, clipping, and regularization methods to the base layer
    functionality. It supports multiple initialization schemes (Orthogonal, Xavier, Normal, and custom),
    weight clipping to prevent exploding gradients, and L2 regularization options.

    Key features:
    - Weight initialization with various schemes (Orthogonal, Xavier, Normal, or custom)
    - Automatic nonlinearity addition (e.g., ReLU) after the weight layer
    - Weight clipping to prevent exploding gradients
    - L2 regularization (weight decay)
    - L2-init regularization (delta regularization) to prevent catastrophic forgetting
    - Weight metrics computation for analysis and debugging

    The implementation handles computation of appropriate clipping values and initialization
    scales based on network architecture, and provides methods to compute regularization
    losses during training.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    initial_weights: Optional[torch.Tensor]

    def __init__(
        self,
        clip_scale: float = 1,
        analyze_weights: Optional[bool] = None,
        l2_norm_scale: Optional[float] = None,
        l2_init_scale: Optional[float] = None,
        nonlinearity: Optional[str] = "nn.ReLU",
        initialization: str = "Orthogonal",
        clip_range: Optional[float] = None,
        **cfg,
    ):
        self.clip_scale = clip_scale
        self.analyze_weights_bool = analyze_weights
        self.l2_norm_scale = l2_norm_scale
        self.l2_init_scale = l2_init_scale
        self.nonlinearity = nonlinearity
        self.initialization = initialization
        self.global_clip_range = clip_range
        self.largest_weight: float = 0.0  # Initialize with default value
        super().__init__(**cfg)

    def _initialize(self):
        # Note - when a property is set on a torch nn.Module, it will automatically be
        # added to gradient tracking. This gets around the problem.
        self.__dict__["_weight_net"] = self._make_net()
        typed_weight_net = cast(LinearWeightModule, self._weight_net)

        self._initialize_weights()

        # Configure weight clipping
        if self.clip_scale > 0:
            # Handle the case where global_clip_range could be None
            clip_range = 1.0 if self.global_clip_range is None else self.global_clip_range
            self.clip_value = clip_range * self.largest_weight * self.clip_scale
        else:
            self.clip_value = 0  # disables clipping (not clip to 0)

        self.initial_weights = None
        if self.l2_init_scale != 0:
            self.initial_weights = typed_weight_net.weight.data.clone()

        # Setup the complete network with nonlinearity if needed
        if self.nonlinearity is not None:
            # expecting a string of the form 'nn.ReLU'
            try:
                _, class_name = self.nonlinearity.split(".")
                if class_name not in dir(nn):
                    raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")
                nonlinearity_class = getattr(nn, class_name)
                self._net = nn.Sequential(typed_weight_net, nonlinearity_class())
            except (AttributeError, KeyError, ValueError) as e:
                raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}") from e
        else:
            # If no nonlinearity, just use the weight network directly
            self._net = typed_weight_net

    def _initialize_weights(self):
        """
        Initialize weights based on the specified initialization method.

        Supports Orthogonal, Xavier, Normal, and custom max_0_01 initializations.
        Each method scales weights appropriately based on fan-in and fan-out dimensions.
        Also initializes biases to zero if present.
        """

        typed_weight_net = cast(LinearWeightModule, self._weight_net)

        fan_in = self._in_tensor_shapes[0][0]
        fan_out = self._out_tensor_shape[0]

        if self.initialization.lower() == "orthogonal":
            if self.nonlinearity == "nn.Tanh":
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(typed_weight_net.weight, gain=gain)
            largest_weight = typed_weight_net.weight.max().item()
        elif self.initialization.lower() == "xavier":
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(typed_weight_net.weight)
        elif self.initialization.lower() == "normal":
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(typed_weight_net.weight, mean=0, std=largest_weight)
        elif self.initialization.lower() == "max_0_01":
            # set to uniform with largest weight = 0.01
            largest_weight = 0.01
            nn.init.uniform_(typed_weight_net.weight, a=-largest_weight, b=largest_weight)
        else:
            raise ValueError(f"Invalid initialization method: {self.initialization}")

        if hasattr(self._weight_net, "bias") and isinstance(typed_weight_net.bias, torch.nn.parameter.Parameter):
            typed_weight_net.bias.data.fill_(0)

        self.largest_weight = largest_weight

    def clip_weights(self):
        """
        Clips weights to prevent exploding gradients.

        If clip_value is positive, clamps all weights to the range [-clip_value, clip_value].
        """
        typed_weight_net = cast(LinearWeightModule, self._weight_net)
        if self.clip_value > 0:
            with torch.no_grad():
                typed_weight_net.weight.data = typed_weight_net.weight.data.clamp(-self.clip_value, self.clip_value)

    def l2_reg_loss(self) -> torch.Tensor:
        typed_weight_net = cast(LinearWeightModule, self._weight_net)
        l2_reg_loss = torch.tensor(0.0, device=typed_weight_net.weight.data.device)
        if self.l2_norm_scale != 0 and self.l2_norm_scale is not None:
            l2_reg_loss = (torch.sum(typed_weight_net.weight.data**2)) * self.l2_norm_scale
        return l2_reg_loss

    def l2_init_loss(self) -> torch.Tensor:
        typed_weight_net = cast(LinearWeightModule, self._weight_net)
        l2_init_loss = torch.tensor(0.0, device=typed_weight_net.weight.data.device)
        if self.l2_init_scale != 0 and self.l2_init_scale is not None and self.initial_weights is not None:
            l2_init_loss = torch.sum((typed_weight_net.weight.data - self.initial_weights) ** 2) * self.l2_init_scale
        return l2_init_loss

    def update_l2_init_weight_copy(self, alpha: float = 0.9):
        typed_weight_net = cast(LinearWeightModule, self._weight_net)
        if self.initial_weights is not None:
            self.initial_weights = (self.initial_weights * alpha + typed_weight_net.weight.data * (1 - alpha)).clone()

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        typed_weight_net = cast(LinearWeightModule, self._weight_net)
        if (
            typed_weight_net.weight.data.dim() != 2
            or not hasattr(self, "analyze_weights_bool")
            or self.analyze_weights_bool is None
            or self.analyze_weights_bool is False
        ):
            return []

        metrics = analyze_weights(typed_weight_net.weight.data, delta)
        metrics["name"] = self._name
        return [metrics]
