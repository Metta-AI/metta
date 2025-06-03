"""Base module for Metta architecture.

Author: Axel
Created: 2024-03-19

This module defines the MettaModule base class and the MettaData wrapper for the Metta architecture.

MettaModule provides a standardized interface for modules that process batched tensor data and propagate
arbitrary metadata through the network. It supports both TensorDict and MettaData as input, always working
internally with MettaData, and returns the same type as it receives. This design enables compatibility with
PyTorch and flexible metadata propagation for modular and extensible architectures.

MettaData is a wrapper that holds a TensorDict (for batched tensors) and a metadata dictionary (for arbitrary
information to be shared between modules). It is used to propagate both tensors and side-channel information
through the network in a unified way.

TODO: Figure out key/shape validation pipeline.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn.modules.normalization import _shape_t


class MettaDict:
    """
    Wrapper for tensor data and arbitrary metadata in the Metta architecture.

    MettaDict holds a TensorDict (for batched tensors) and a metadata dictionary (for arbitrary
    information to be shared between modules). It is used to propagate both tensors and side-channel
    information through the network in a unified way. Modules should only modify their own keys in
    the metadata dictionary to avoid conflicts.

    Access the TensorDict via the 'td' attribute and the metadata via the 'data' attribute.
    """

    def __init__(self, td: TensorDict, data: Dict[str, Any]):
        self.td = td
        self.data = data

    def __repr__(self) -> str:
        return f"MettaDict(td={self.td}, data={self.data})"


class MettaModule(nn.Module, ABC):
    """Base class for all modules in the Metta architecture.

    This abstract base class provides a standardized interface for modules that process
    TensorDicts or MettaData. It enforces a clear input/output contract through key-based tensor
    management and shape validation, and supports propagation of arbitrary metadata via MettaData.

    Attributes:
        in_keys: List of keys for input tensors in the TensorDict
        out_keys: List of keys for output tensors in the TensorDict
        input_features_shape: Optional expected shape of input features (excluding batch dimension)
        output_features_shape: Optional expected shape of output features (excluding batch dimension)

    Usage:
        - The public forward method accepts either a TensorDict or a MettaData as input.
        - The return type always matches the input type (TensorDict in, TensorDict out; MettaData in, MettaData out).
        - All internal computation and validation is performed using MettaData.
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

    def forward(self, td: Union[TensorDict, MettaDict]) -> Union[TensorDict, MettaDict]:
        """Forward pass that computes outputs and updates the TensorDict or MettaData.
        This method should only be overridden if the module needs to perform additional operations
        beyond the computation of the output tensors.

        Args:
            td: Input TensorDict or MettaData containing the input tensors

        Returns:
            Updated TensorDict or MettaData with new output tensors (matches input type)
        """
        is_metta = isinstance(td, MettaDict)
        md = td if is_metta else MettaDict(td, {})

        if __debug__:
            self._validate_input_keys(md.td)
            if self.input_features_shape is not None:
                self._check_shapes(md.td)

        outputs = self._compute(md)

        for key, tensor in outputs.items():
            md.td[key] = tensor

        return md if is_metta else md.td

    @abstractmethod
    def _compute(self, md: MettaDict) -> Dict[str, torch.Tensor]:
        """Compute the module's output tensors.
        This method is called by the forward method to compute the output tensors.
        Override this method in your subclass to implement the actual computation.
        This is a PURELY computational method, no validation should be done here.

        Args:
            md: Input MettaData containing the input tensors

        Returns:
            Dictionary mapping output keys to their corresponding tensors
        """
        pass

    def _validate_input_keys(self, td: TensorDict) -> None:
        """Validate that all required input keys are present in the TensorDict.

        This method enforces the input contract by ensuring all keys specified in
        self.in_keys exist in the input TensorDict.

        Args:
            td: Input TensorDict to validate

        Raises:
            KeyError: If any required input key is missing from the TensorDict
        """
        for key in self.in_keys:
            if key not in td:
                raise KeyError(f"Input key {key} not found in the TensorDict")

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


class UniqueInKeyMixin:
    """Mixin for MettaModule subclasses that require exactly one input key.
    Checks the contract at initialization and provides the 'in_key' property."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "in_keys") or len(self.in_keys) != 1:  # type: ignore
            raise ValueError(f"{self.__class__.__name__} requires exactly one input key.")

    @property
    def in_key(self) -> str:
        return self.in_keys[0]  # type: ignore


class UniqueOutKeyMixin:
    """Mixin for MettaModule subclasses that require exactly one output key.
    Checks the contract at initialization and provides the 'out_key' property."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "out_keys") or len(self.out_keys) != 1:  # type: ignore
            raise ValueError(f"{self.__class__.__name__} requires exactly one output key.")

    @property
    def out_key(self) -> str:
        return self.out_keys[0]  # type: ignore


class MettaLinear(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        input_features_shape: List[int],
        output_features_shape: List[int],
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError("MettaLinear requires exactly one input and one output key")

        if self.input_features_shape is None or self.output_features_shape is None:
            raise ValueError("LinearModule requires both input_features_shape and output_features_shape")

        self.linear = nn.Linear(self.input_features_shape[0], self.output_features_shape[0])

    def _compute(self, td) -> Dict[str, torch.Tensor]:
        return {self.out_key: self.linear(td.td[self.in_key])}


class MettaReLU(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
    ):
        super().__init__(in_keys, out_keys)
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError("MettaReLU requires exactly one input and one output key")
        self.relu = nn.ReLU()

    def _compute(self, td) -> Dict[str, torch.Tensor]:
        return {self.out_key: self.relu(td.td[self.in_key])}


class MettaConv2d(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        input_features_shape: list[int],
        output_features_shape: list[int],
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError("MettaConv2d requires exactly one input and one output key")
        self.conv = nn.Conv2d(
            in_channels=input_features_shape[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def _compute(self, md: MettaDict) -> dict:
        return {self.out_key: self.conv(md.td[self.in_key])}


class MettaFlatten(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        input_features_shape: list[int],
        output_features_shape: list[int],
        start_dim: int = 1,
        end_dim: int = -1,
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError("MettaFlatten requires exactly one input and one output key")
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def _compute(self, md: MettaDict) -> dict:
        return {self.out_key: self.flatten(md.td[self.in_key])}


class MettaLayerNorm(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        input_features_shape: list[int],
        output_features_shape: list[int],
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError("MettaLayerNorm requires exactly one input and one output key")
        self.ln = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def _compute(self, md: MettaDict) -> dict:
        return {self.out_key: self.ln(md.td[self.in_key])}
