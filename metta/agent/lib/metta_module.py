"""Base module for Metta architecture.

Author: Axel
Created: 2024-03-19

Structure:
- Imports
- MettaModule class
  - __init__
  - forward
  - _compute (abstract)
  - _check_input_keys
  - _check_output_keys
- Subclasses

TODO: Figure out key/shape validation pipeline.

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


# 1 - Base class
class MettaModule(nn.Module, ABC):
    """Base class for all modules in the Metta architecture.

    This abstract base class provides a standardized interface for modules that process
    TensorDicts. It enforces a clear input/output contract through key-based tensor
    management and shape validation.

    Attributes:
        in_keys: List of keys for input tensors in the TensorDict
        out_keys: List of keys for output tensors in the TensorDict
        input_features_shape: Optional expected shape of input features (excluding batch dimension)
        output_features_shape: Optional expected shape of output features (excluding batch dimension)
    """

    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        input_features_shape: Optional[List[int]] = None,
        output_features_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.input_features_shape = torch.Size(input_features_shape) if input_features_shape else None
        self.output_features_shape = torch.Size(output_features_shape) if output_features_shape else None

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass that computes outputs and updates the TensorDict.
        This method should only be overridden if the module needs to perform additional operations
        beyond the computation of the output tensors.

        Args:
            td: Input TensorDict containing the input tensors

        Returns:
            Updated TensorDict with new output tensors
        """
        self._check_input_keys(td)
        if self.input_features_shape is not None:
            self._check_shapes(td)
        outputs = self._compute(td)
        for key, tensor in outputs.items():
            td[key] = tensor
        return td

    @abstractmethod
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        """Compute the module's output tensors.
        This method is called by the forward method to compute the output tensors.
        Override this method in your subclass to implement the actual computation.
        This is a PURELY computational method, no validation should be done here.

        Args:
            td: Input TensorDict containing the input tensors

        Returns:
            Dictionary mapping output keys to their corresponding tensors
        """
        pass

    def _check_input_keys(self, td: TensorDict) -> None:
        """Check if the input keys are present in the TensorDict.
        If not, raise a KeyError.
        """
        for key in self.in_keys:
            if key not in td:
                raise KeyError(f"Input key {key} not found in the TensorDict")

    def _check_output_keys(self, td: TensorDict) -> None:
        """Check if the output keys are present in the TensorDict.
        If they are, raise a KeyError.

        NOTE: This is not a problem if the output keys are already in the TensorDict,
        as the forward method will overwrite them, so maybe we don't need to check for this.
        """
        for key in self.out_keys:
            if key in td:
                raise KeyError(f"Output key {key} already exists in the TensorDict")

    def _check_shapes(self, td: TensorDict) -> None:
        """Check if the input tensors have the expected shapes.
        If not, raise a ValueError.

        Note:
            Shape validation ignores the batch dimension (first dimension).
            For example, if input_features_shape is [10, 20] and the tensor
            has shape [batch_size, 10, 20], the validation will pass.
        """
        for key in self.in_keys:
            tensor = td[key]
            if len(tensor.shape) < 2:
                raise ValueError(f"Input tensor {key} must have at least 2 dimensions (batch + features)")
            if tensor.shape[1:] != self.input_features_shape:
                raise ValueError(
                    f"Input tensor {key} has feature shape {tensor.shape[1:]}, "
                    f"expected {self.input_features_shape} (ignoring batch dimension)"
                )


# 2 - Subclasses (specific modules)


class LinearModule(MettaModule):
    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        input_features_shape: List[int],
        output_features_shape: List[int],
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError("LinearModule requires exactly one input and one output key")

        if self.input_features_shape is None or self.output_features_shape is None:
            raise ValueError("LinearModule requires both input_features_shape and output_features_shape")

        self.linear = nn.Linear(self.input_features_shape[0], self.output_features_shape[0])

    @property
    def in_key(self) -> str:
        return self.in_keys[0]

    @property
    def out_key(self) -> str:
        return self.out_keys[0]

    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        return {self.out_key: self.linear(td[self.in_key])}


class ReLUModule(MettaModule):
    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
    ):
        super().__init__(in_keys, out_keys)
        self.relu = nn.ReLU()

    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        return {"output": self.relu(td["input"])}
