from typing import Any, Dict, List, Optional, cast

import omegaconf
import torch
from typing_extensions import override

from metta.agent.lib.metta_layer import LayerBase


class MergeLayerBase(LayerBase):
    """
    Base class for layers that combine multiple tensors from different sources.

    This class provides the framework for merging tensors from multiple sources in various ways.
    Subclasses implement specific merging operations (concatenation, addition, subtraction, averaging).
    The class handles tensor shape validation, optional slicing of source tensors, and tracking of
    input/output tensor dimensions.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, name: str, **cfg: Any):
        self._ready: bool = False
        self._source_components: Optional[Dict[str, Any]] = None
        self._in_tensor_shapes: List[Any] = []
        self.dims: List[int] = []
        self.processed_lengths: List[int] = []
        super().__init__(name, **cfg)

    @property
    def ready(self) -> bool:
        return self._ready

    @override
    def setup(self, source_components: Optional[Dict[str, Any]] = None) -> None:
        if self._ready:
            return

        self._source_components = source_components

        # NOTE: in and out tensor shapes do not include batch sizes
        # however, all other sizes do, including processed_lengths
        self._in_tensor_shapes = []

        self.dims = []
        self.processed_lengths = []

        for src_cfg in self._sources:
            source_name = src_cfg["name"]

            if source_name not in self._source_components:
                raise ValueError(f"sources contains no layer named {source_name}")

            source_component = cast(Any, self._source_components[source_name])
            processed_size = source_component._out_tensor_shape.copy()
            self._in_tensor_shapes.append(processed_size)

            processed_size = processed_size[0]
            if src_cfg.get("slice") is not None:
                slice_range = src_cfg["slice"]
                if isinstance(slice_range, omegaconf.listconfig.ListConfig):
                    slice_range = list(slice_range)
                if not (isinstance(slice_range, (list, tuple)) and len(slice_range) == 2):
                    raise ValueError(f"'slice' must be a two-element list/tuple for source {source_name}.")

                start, end = slice_range
                slice_dim = src_cfg.get("dim", None)
                if slice_dim is None:
                    raise ValueError(
                        f"Slice 'dim' must be specified for {source_name}. If a vector, use dim=1 (0 is batch size)."
                    )
                length = end - start
                src_cfg["_slice_params"] = {"start": start, "length": length, "dim": slice_dim}
                processed_size = length

            self.processed_lengths.append(processed_size)

            self.dims.append(src_cfg.get("dim", 1))  # check if default dim is good to have or will cause problems

        self._setup_merge_layer()
        self._ready = True

    def _setup_merge_layer(self) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    @override
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = []
        # TODO: do this without a for loop or dictionary lookup for perf
        for src_cfg in self._sources:
            source_name = src_cfg["name"]

            if not self._source_components or source_name not in self._source_components:
                raise ValueError(f"_source_components contains no layer named {source_name}")

            source_component = cast(Any, self._source_components[source_name])
            source_component.forward(data)
            src_tensor = data[source_name]

            if "_slice_params" in src_cfg:
                params = src_cfg["_slice_params"]
                src_tensor = torch.narrow(src_tensor, dim=params["dim"], start=params["start"], length=params["length"])
            outputs.append(src_tensor)

        return self._merge(outputs, data)

    def _merge(self, outputs: List[torch.Tensor], data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this method.")


class ConcatMergeLayer(MergeLayerBase):
    """
    Concatenates tensors along a specified dimension.

    This layer combines multiple tensors by concatenating them along a specified dimension,
    resulting in a larger tensor with the combined content. For vectors, use dim=1.
    When used with observations, it can concatenate channels (dim=1) with their associated
    fields. Note that concatenating along width and height dimensions (dim=2 or dim=3) would
    lead to non-uniform shapes in the field of view.

    All input tensors must use the same dimension for concatenation.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _setup_merge_layer(self) -> None:
        if not all(d == self.dims[0] for d in self.dims):
            raise ValueError(f"For 'concat', all sources must have the same 'dim'. Got dims: {self.dims}")
        self._merge_dim = self.dims[0]
        cat_dim_length = 0
        for size in self.processed_lengths:
            cat_dim_length += size
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        self._out_tensor_shape[self._merge_dim - 1] = cat_dim_length  # the -1 is to account for batch size

    def _merge(self, outputs: List[torch.Tensor], data: Dict[str, Any]) -> Dict[str, Any]:
        merged = torch.cat(outputs, dim=self._merge_dim)
        data[self._name] = merged
        return data


class AddMergeLayer(MergeLayerBase):
    """
    Combines tensors by element-wise addition.

    This layer adds multiple tensors element-wise, requiring that all input tensors have
    identical shapes. The output tensor maintains the same shape as the inputs.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _setup_merge_layer(self) -> None:
        if not all(s == self._in_tensor_shapes[0] for s in self._in_tensor_shapes):
            raise ValueError(f"For 'add', all source sizes must match. Got sizes: {self._in_tensor_shapes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape = self._in_tensor_shapes[0]

    def _merge(self, outputs: List[torch.Tensor], data: Dict[str, Any]) -> Dict[str, Any]:
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        data[self._name] = merged
        return data


class SubtractMergeLayer(MergeLayerBase):
    """
    Subtracts the second tensor from the first tensor element-wise.

    This layer performs element-wise subtraction between exactly two input tensors,
    requiring that both tensors have identical shapes. The operation computes
    outputs[0] - outputs[1], maintaining the same shape as the inputs.

    Raises ValueError if more or fewer than exactly two sources are provided.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _setup_merge_layer(self) -> None:
        if not all(s == self._in_tensor_shapes[0] for s in self._in_tensor_shapes):
            raise ValueError(f"For 'subtract', all source sizes must match. Got sizes: {self._in_tensor_shapes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape = self._in_tensor_shapes[0]

    def _merge(self, outputs: List[torch.Tensor], data: Dict[str, Any]) -> Dict[str, Any]:
        if len(outputs) != 2:
            raise ValueError("Subtract merge_op requires exactly two sources.")
        merged = outputs[0] - outputs[1]
        data[self._name] = merged
        return data


class MeanMergeLayer(MergeLayerBase):
    """
    Computes the element-wise mean (average) of input tensors.

    This layer calculates the average of all input tensors element-wise, requiring
    that all tensors have identical shapes. The output tensor maintains the same
    shape as the inputs.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _setup_merge_layer(self) -> None:
        if not all(s == self._in_tensor_shapes[0] for s in self._in_tensor_shapes):
            raise ValueError(f"For 'mean', all source sizes must match. Got sizes: {self._in_tensor_shapes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape = self._in_tensor_shapes[0]

    def _merge(self, outputs: List[torch.Tensor], data: Dict[str, Any]) -> Dict[str, Any]:
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        merged = merged / len(outputs)
        data[self._name] = merged
        return data


class ExpandLayer(LayerBase):
    """
    Expands a tensor along a specified dimension.

    This layer can expand a tensor in one of two ways:
    1. By a fixed value specified by expand_value parameter
    2. By deriving the expansion size from another tensor (specified by dims_source and source_dim)

    The expanded dimension is inserted at the position specified by expand_dim.

    Args:
        name (str): Name of the layer
        expand_dim (int): Dimension along which to expand (0 is batch dimension)
        sources: Input source(s)
        expand_value (int, optional): Fixed value to expand by
        source_dim (int, optional): Dimension in the dims_source tensor to use for expansion
        dims_source (str, optional): Name of tensor to get expansion size from

    Note: This layer has not been unit tested.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(
        self,
        name: str,
        expand_dim: int,
        sources: Any,
        expand_value: Optional[int] = None,
        source_dim: Optional[int] = None,
        dims_source: Optional[str] = None,
        **cfg: Any,
    ):
        self._ready: bool = False
        self.expand_dim = expand_dim
        self.expand_value = expand_value
        self.source_dim = source_dim
        self.dims_source = dims_source
        if dims_source is not None and isinstance(sources, dict):
            self._sources = [sources[0], dims_source]
        super().__init__(name, sources, **cfg)

    @property
    def ready(self) -> bool:
        return self._ready

    @override
    def setup(self, source_components: Optional[Dict[str, Any]] = None) -> None:
        if self._ready:
            return

        self._source_components = source_components

        if not self._source_components:
            raise ValueError("source_components cannot be None for ExpandLayer setup")

        first_component = next(iter(self._source_components.values()))
        self._out_tensor_shape = first_component._out_tensor_shape.copy()

        if self.dims_source is not None and self.source_dim is not None and self._source_components:
            dims_source_component = self._source_components.get(self.dims_source)
            if dims_source_component:
                self.expand_value = dims_source_component._out_tensor_shape[
                    self.source_dim - 1
                ]  # -1 because _out_tensor_shape doesn't account for batch size

        if self.expand_dim > 0 and self.expand_value is not None:
            self._out_tensor_shape.insert(
                self.expand_dim - 1, self.expand_value
            )  # -1 because _out_tensor_shape doesn't account for batch size
        else:
            raise ValueError("Expand dim must be greater than 0. 0 is the batch dimension.")

        self._ready = True

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source_name = self._sources[0]["name"] if isinstance(self._sources[0], dict) else self._sources[0]
        tensor = data[source_name]

        if self.dims_source is not None and self.source_dim is not None:
            self.expand_value = data[self.dims_source].size(self.source_dim)

        if self.expand_value is None:
            raise ValueError("expand_value must be set before forward pass")

        expanded = tensor.unsqueeze(self.expand_dim)
        expand_shape = [-1] * expanded.dim()
        expand_shape[self.expand_dim] = self.expand_value
        data[self._name] = expanded.expand(*expand_shape).contiguous()
        return data


class ReshapeLayer(LayerBase):
    """
    Multiplies two dimensions together, squeezing them into a single dimension.

    This layer combines two dimensions of a tensor by multiplying their sizes and
    placing the result in the squeezed_dim position, while removing the popped_dim.
    This is useful for flattening or reorganizing tensor dimensions.

    Args:
        name (str): Name of the layer
        popped_dim (int): Dimension to be removed after multiplication
        squeezed_dim (int): Dimension to place the combined result in

    Note: This layer has not been unit tested.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, name: str, popped_dim: int, squeezed_dim: int, **cfg: Any):
        self._ready: bool = False
        self.popped_dim = popped_dim
        self.squeezed_dim = squeezed_dim
        super().__init__(name, **cfg)

    @override
    def setup(self, source_components: Optional[Dict[str, Any]] = None) -> None:
        if self._ready:
            return

        self._source_components = source_components

        if not self._source_components:
            raise ValueError("source_components cannot be None for ReshapeLayer setup")

        first_component = next(iter(self._source_components.values()))
        self._out_tensor_shape = first_component._out_tensor_shape.copy()

        if self.squeezed_dim == 0 or self.popped_dim == 0:
            # we are involving the batch size, which we don't have ahead of time
            self._out_tensor_shape.pop(self.popped_dim - 1)
        else:
            compressed_size = (
                self._out_tensor_shape[self.popped_dim - 1] * self._out_tensor_shape[self.squeezed_dim - 1]
            )
            self._out_tensor_shape[self.squeezed_dim - 1] = compressed_size
            self._out_tensor_shape.pop(self.popped_dim - 1)

        self._ready = True

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source_name = self._sources[0]["name"] if isinstance(self._sources[0], dict) else self._sources[0]
        tensor = data[source_name]
        shape = list(tensor.shape)
        compressed_size = shape[self.squeezed_dim] * shape[self.popped_dim]
        shape.pop(self.popped_dim)
        shape[self.squeezed_dim] = compressed_size
        data[self._name] = tensor.view(*shape)
        return data


class BatchReshapeLayer(LayerBase):
    """
    Reshapes a tensor to introduce a time dimension from the batch dimension.

    This layer takes a flattened batch of shape [B*T, ...] and reshapes it to
    [B, T, ...] by inferring B from the "_BxTT" value in the component dict.
    It's typically used to convert between flattened and structured batch-time
    representations.

    The layer first expands the tensor at dimension 1, sets it equal to the value
    at dimension 0 divided by B, sets dimension 0 to B, and finally squeezes
    the tensor to remove any singleton dimensions.

    Note: This layer has not been unit tested.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, name: str, **cfg: Any):
        self._ready: bool = False
        super().__init__(name, **cfg)

    @override
    def setup(self, source_components: Optional[Dict[str, Any]] = None) -> None:
        if self._ready:
            return

        self._source_components = source_components

        if not self._source_components:
            raise ValueError("source_components cannot be None for BatchReshapeLayer setup")

        first_component = next(iter(self._source_components.values()))
        self._out_tensor_shape = first_component._out_tensor_shape.copy()
        # the out_tensor_shape is NOT ACCURATE because we don't know the batch size ahead of time.
        self._ready = True

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source_name = self._sources[0]["name"] if isinstance(self._sources[0], dict) else self._sources[0]
        tensor = data[source_name]
        B_T = data["_BxT_"]
        shape = list(tensor.shape)
        shape.insert(1, 0)
        shape[1] = shape[0] // (B_T)
        shape[0] = B_T
        data[self._name] = tensor.view(*shape).squeeze()
        return data


class CenterPixelLayer(LayerBase):
    """
    Extracts the center pixel from a tensor with shape (B, C, H, W).

    This layer selects only the center pixel from spatial dimensions H and W,
    resulting in a tensor of shape (B, C). This is useful for focusing on the
    central part of an image or feature map.

    Note: H and W must be odd numbers for there to be a clear center pixel.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, name: str, **cfg: Any):
        self._ready: bool = False
        super().__init__(name, **cfg)

    @override
    def setup(self, source_components: Optional[Dict[str, Any]] = None) -> None:
        if self._ready:
            return

        self._source_components = source_components

        if not self._source_components:
            raise ValueError("source_components cannot be None for CenterPixelLayer setup")

        first_component = next(iter(self._source_components.values()))
        self._out_tensor_shape = first_component._out_tensor_shape.copy()
        del self._out_tensor_shape[-2:]

        self._ready = True

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source_name = self._sources[0]["name"] if isinstance(self._sources[0], dict) else self._sources[0]
        tensor = data[source_name]
        B, C, H, W = tensor.shape
        center_h = H // 2
        center_w = W // 2
        data[self._name] = tensor[:, :, center_h, center_w]
        return data
