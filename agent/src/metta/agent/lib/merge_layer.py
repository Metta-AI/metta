import omegaconf
import torch
from tensordict import TensorDict

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

    def __init__(self, name, **cfg):
        self._ready = False
        super().__init__(name, **cfg)

    @property
    def ready(self):
        return self._ready

    def setup(self, _source_components=None):
        if self._ready:
            return

        self._source_components = _source_components

        # NOTE: in and out tensor shapes do not include batch sizes
        # however, all other sizes do, including processed_lengths
        self._in_tensor_shapes = []

        self.dims = []
        self.processed_lengths = []
        for src_cfg in self._sources:
            source_name = src_cfg["name"]

            processed_size = self._source_components[source_name]._out_tensor_shape.copy()
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

    def _setup_merge_layer(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, td: TensorDict):
        outputs = []
        # TODO: do this without a for loop or dictionary lookup for perf
        for src_cfg in self._sources:
            source_name = src_cfg["name"]
            self._source_components[source_name].forward(td)
            src_tensor = td[source_name]

            if "_slice_params" in src_cfg:
                params = src_cfg["_slice_params"]
                src_tensor = torch.narrow(src_tensor, dim=params["dim"], start=params["start"], length=params["length"])
            outputs.append(src_tensor)

        return self._merge(outputs, td)

    def _merge(self, outputs, td):
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

    def _setup_merge_layer(self):
        if not all(d == self.dims[0] for d in self.dims):
            raise ValueError(f"For 'concat', all sources must have the same 'dim'. Got dims: {self.dims}")
        self._merge_dim = self.dims[0]
        cat_dim_length = 0
        for size in self.processed_lengths:
            cat_dim_length += size
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        self._out_tensor_shape[self._merge_dim - 1] = cat_dim_length  # the -1 is to account for batch size

    def _merge(self, outputs, td):
        merged = torch.cat(outputs, dim=self._merge_dim)
        td[self._name] = merged
        return td


class AddMergeLayer(MergeLayerBase):
    """
    Combines tensors by element-wise addition.

    This layer adds multiple tensors element-wise, requiring that all input tensors have
    identical shapes. The output tensor maintains the same shape as the inputs.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _setup_merge_layer(self):
        if not all(s == self._in_tensor_shapes[0] for s in self._in_tensor_shapes):
            raise ValueError(f"For 'add', all source sizes must match. Got sizes: {self._in_tensor_shapes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape = self._in_tensor_shapes[0]

    def _merge(self, outputs, td):
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        td[self._name] = merged
        return td


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

    def _setup_merge_layer(self):
        if not all(s == self._in_tensor_shapes[0] for s in self._in_tensor_shapes):
            raise ValueError(f"For 'subtract', all source sizes must match. Got sizes: {self._in_tensor_shapes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape = self._in_tensor_shapes[0]

    def _merge(self, outputs, td):
        if len(outputs) != 2:
            raise ValueError("Subtract merge_op requires exactly two sources.")
        merged = outputs[0] - outputs[1]
        td[self._name] = merged
        return td


class MeanMergeLayer(MergeLayerBase):
    """
    Computes the element-wise mean (average) of input tensors.

    This layer calculates the average of all input tensors element-wise, requiring
    that all tensors have identical shapes. The output tensor maintains the same
    shape as the inputs.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _setup_merge_layer(self):
        if not all(s == self._in_tensor_shapes[0] for s in self._in_tensor_shapes):
            raise ValueError(f"For 'mean', all source sizes must match. Got sizes: {self._in_tensor_shapes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape = self._in_tensor_shapes[0]

    def _merge(self, outputs, td):
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        merged = merged / len(outputs)
        td[self._name] = merged
        return td


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

    def __init__(self, name, expand_dim, sources, expand_value=None, source_dim=None, dims_source=None, **cfg):
        super().__init__(name, sources, **cfg)
        self._ready = False
        self.expand_dim = expand_dim
        self.expand_value = expand_value
        self.source_dim = source_dim
        self.dims_source = dims_source
        if dims_source is not None:
            self._sources = [sources[0], dims_source]

    @property
    def ready(self):
        return self._ready

    def setup(self, _source_components=None):
        if self._ready:
            return

        self._source_components = _source_components
        self._out_tensor_shape = next(iter(self._source_components.values()))._out_tensor_shape.copy()

        if self.dims_source is not None:
            self.expand_value = self._source_components[self.dims_source]._out_tensor_shape[
                self.source_dim - 1
            ]  # -1 because _out_tensor_shape doesn't account for batch size

        if self.expand_dim > 0:
            self._out_tensor_shape.insert(
                self.expand_dim - 1, self.expand_value
            )  # -1 because _out_tensor_shape doesn't account for batch size
        else:
            raise ValueError("Expand dim must be greater than 0. 0 is the batch dimension.")

        self._ready = True

    def _forward(self, td: TensorDict):
        tensor = td[self._sources[0]["name"]]

        if self.dims_source is not None:
            self.expand_value = td[self.dims_source].size(self.source_dim)

        expanded = tensor.unsqueeze(self.expand_dim)
        expand_shape = [-1] * expanded.dim()
        expand_shape[self.expand_dim] = self.expand_value
        td[self._name] = expanded.expand(*expand_shape).contiguous()
        return td


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

    def __init__(self, name, popped_dim, squeezed_dim, **cfg):
        self._ready = False
        self.popped_dim = popped_dim
        self.squeezed_dim = squeezed_dim
        super().__init__(name, **cfg)

    def setup(self, _source_components=None):
        if self._ready:
            return

        self._source_components = _source_components
        self._out_tensor_shape = next(iter(self._source_components.values()))._out_tensor_shape.copy()
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

    def _forward(self, td: TensorDict):
        tensor = td[self._sources[0]["name"]]
        shape = list(tensor.shape)
        compressed_size = shape[self.squeezed_dim] * shape[self.popped_dim]
        shape.pop(self.popped_dim)
        shape[self.squeezed_dim] = compressed_size
        td[self._name] = tensor.view(*shape)
        return td


class BatchReshapeLayer(LayerBase):
    """
    Reshapes a tensor to introduce a time dimension from the batch dimension.

    This layer takes a flattened batch of shape [B*TT, ...] and reshapes it to
    [B, TT, ...] by inferring B from the "_BxTT_" value in the TensorDict.
    It's typically used to convert between flattened and structured batch-time
    representations.

    The layer first expands the tensor at dimension 1, sets it equal to the value
    at dimension 0 divided by B, sets dimension 0 to B, and finally squeezes
    the tensor to remove any singleton dimensions.

    Note: This layer has not been unit tested.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, name, **cfg):
        self._ready = False
        super().__init__(name, **cfg)

    def setup(self, _source_components=None):
        if self._ready:
            return

        self._source_components = _source_components
        self._out_tensor_shape = next(iter(self._source_components.values()))._out_tensor_shape.copy()
        # the out_tensor_shape is NOT ACCURATE because we don't know the batch size ahead of time.
        self._ready = True

    def _forward(self, td: TensorDict):
        tensor = td[self._sources[0]["name"]]
        B_TT = td["_BxTT_"]
        shape = list(tensor.shape)
        shape.insert(1, 0)
        shape[1] = shape[0] // (B_TT)
        shape[0] = B_TT
        td[self._name] = tensor.view(*shape).squeeze()
        return td


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

    def __init__(self, name, **cfg):
        super().__init__(name, **cfg)

    def setup(self, _source_components=None):
        if self._ready:
            return

        self._source_components = _source_components
        self._out_tensor_shape = next(iter(self._source_components.values()))._out_tensor_shape.copy()
        del self._out_tensor_shape[-2:]

        self._ready = True

    def _forward(self, td: TensorDict):
        tensor = td[self._sources[0]["name"]]
        B, C, H, W = tensor.shape
        center_h = H // 2
        center_w = W // 2
        td[self._name] = tensor[:, :, center_h, center_w]
        return td
