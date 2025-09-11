from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.util.weights_analysis import analyze_weights
from metta.rl.trainer_state import TrainerState


@dataclass
class NNParams:
    out_features: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }

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

    The `_forward` method should only take a tensordict as input and return
    only the tensordict. The tensor dict is constructed anew each time the
    metta_agent forward pass is run. The component's `_forward` should read
    from the value at the key name of its input_source. After performing its
    computation via self._net() or otherwise, it should store its output in the
    tensor dict at the key with its own name.

    Before doing this, it should first check if the tensordict already has a
    key with its own name, indicating that its computation has already been
    performed (due to some other run up the DAG) and return. After this check,
    it should check if its input source is not None and recursively call the
    forward method of the layer above it.

    Carefully passing input and output shapes is necessary to setup the agent.
    self.in_tensor_shapes and self.out_tensor_shape are always of type list.
    Note that these lists not include the batch dimension so their shape is
    one dimension smaller than the actual shape of the tensor.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy."""

    def __init__(self, name: str, sources: list[str] | None = None,
                 nn_params: NNParams | None = None):
        super().__init__()

        # Extract name from cfg if not provided directly
        self._name = name
        self._nn_params = nn_params or NNParams()
        self._source_component_names = sources or []
        self._source_components: dict[str, "LayerBase"] = {} # xcxc
        self._in_tensor_shapes: list[torch.Size] = []
        self._out_tensor_shape: torch.Size = torch.Size([])

        self._net: nn.Module | None = None
        self._ready = False

    @property
    def ready(self):
        return self._ready

    def setup(self, source_components: dict[str, "LayerBase"]):
        """Sets up layer with source components and tensor shapes."""
        if self._ready:
            return

        self._in_tensor_shapes = [
            c._out_tensor_shape for c in source_components.values()]
        self._source_components = source_components
        self._initialize()
        self._ready = True

    def _initialize(self):
        self._net = self._make_net()

    def _make_net(self):
        pass

    def forward(self, td: TensorDict):
        if self._name in td:
            return td

        # recursively call the forward method of the source components
        for source in self._source_components.values():
            source.forward(td)

        self._forward(td)

        return td

    def _forward(self, td: TensorDict):
        """Components that have more than one input sources must have their own _forward() method."""
        # get the input tensor from the source component by calling its forward method (which recursively calls
        # _forward() on its source components)
        assert self._net is not None
        assert len(self._source_components) == 1

        td[self._name] = self._net(td[self.src_component_name()])
        return td

    def on_new_training_run(self):
        pass

    def on_rollout_start(self):
        pass

    def on_train_mb_start(self):
        pass

    def on_eval_start(self):
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> dict | None:
        pass

    @property
    def name(self) -> str:
        return self._name

    def src_component_name(self, index: int = 0) -> str:
        return list(self._source_components.values())[index].name

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

    def __init__(
        self,
        name: str,
        sources: list[str] | None = None,
        clip_scale=1,
        analyze_weights=None,
        l2_init_scale=1,
        nonlinearity: str | None = None,
        initialization="Orthogonal",
        clip_range=None,
        nn_params: NNParams | None = None,
    ):
        self.clip_scale = clip_scale
        self.analyze_weights_bool = analyze_weights
        self.l2_init_scale = l2_init_scale
        self.initialization = initialization
        self.global_clip_range = clip_range
        self.nonlinearity = nonlinearity
        self.nn_params = nn_params or NNParams()
        super().__init__(name, sources, nn_params)

    def _initialize(self):
        self.__dict__["weight_net"] = self._make_net()

        self._initialize_weights()

        if self.clip_scale > 0 and self.global_clip_range is not None:
            self.clip_value = self.global_clip_range * self.largest_weight * self.clip_scale
        else:
            self.clip_value = 0  # disables clipping (not clip to 0)

        self.initial_weights = None
        if self.l2_init_scale != 0:
            self.initial_weights = self.weight_net.weight.data.clone()

        self._net = self.weight_net
        if self.nonlinearity is not None:
            # expecting a string of the form 'nn.ReLU'
            try:
                _, class_name = self.nonlinearity.split(".")
                if class_name not in dir(nn):
                    raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")
                nonlinearity_class = getattr(nn, class_name)
                self._net = nn.Sequential(self.weight_net, nonlinearity_class())
                self.__dict__["weight_net"] = self._net[0]
            except (AttributeError, KeyError, ValueError) as e:
                raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}") from e

    def _initialize_weights(self):
        """
        Initialize weights based on the specified initialization method.

        Supports Orthogonal, Xavier, Normal, and custom max_0_01 initializations.
        Each method scales weights appropriately based on fan-in and fan-out dimensions.
        Also initializes biases to zero if present.
        """
        fan_in = self._in_tensor_shapes[0][0]
        fan_out = self._out_tensor_shape[0]

        if self.initialization.lower() == "orthogonal":
            if self.nonlinearity == "nn.Tanh":
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(self.weight_net.weight, gain=gain)
            largest_weight = self.weight_net.weight.max().item()
        elif self.initialization.lower() == "xavier":
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(self.weight_net.weight)
        elif self.initialization.lower() == "normal":
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(self.weight_net.weight, mean=0, std=largest_weight)
        elif self.initialization.lower() == "max_0_01":
            # set to uniform with largest weight = 0.01
            largest_weight = 0.01
            nn.init.uniform_(self.weight_net.weight, a=-largest_weight, b=largest_weight)
        else:
            raise ValueError(f"Invalid initialization method: {self.initialization}")

        if hasattr(self.weight_net, "bias") and isinstance(self.weight_net.bias, torch.nn.parameter.Parameter):
            self.weight_net.bias.data.fill_(0)

        self.largest_weight = largest_weight

    def clip_weights(self):
        """Clips weights to prevent exploding gradients."""
        if self.clip_value > 0:
            with torch.no_grad():
                self.weight_net.weight.data = self.weight_net.weight.data.clamp(-self.clip_value, self.clip_value)

    def l2_init_loss(self) -> torch.Tensor:
        """Computes L2-init regularization loss to prevent catastrophic forgetting."""
        l2_init_loss = torch.tensor(0.0, device=self.weight_net.weight.data.device, dtype=torch.float32)
        l2_init_loss = torch.sum((self.weight_net.weight.data - self.initial_weights) ** 2) * self.l2_init_scale
        return l2_init_loss

    def compute_weight_metrics(self, delta: float = 0.01) -> dict:
        """Computes weight matrix dynamics metrics for analysis and debugging."""
        if (
            self.weight_net.weight.data.dim() != 2
            or not hasattr(self, "analyze_weights_bool")
            or self.analyze_weights_bool is None
            or self.analyze_weights_bool is False
        ):
            return None

        metrics = analyze_weights(self.weight_net.weight.data, delta)
        metrics["name"] = self._name
        return metrics

    @property
    def out_tensor_shape(self) -> torch.Size:
        return self._out_tensor_shape

    @property
    def in_tensor_shapes(self) -> list[torch.Size]:
        return self._in_tensor_shapes
