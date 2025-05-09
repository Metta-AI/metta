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
