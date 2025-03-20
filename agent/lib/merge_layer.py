import copy

import omegaconf
import torch
from tensordict import TensorDict

from agent.lib.metta_layer import LayerBase

class MergeLayerBase(LayerBase):
    def __init__(self, name, **cfg):
        self._ready = False
        super().__init__(name, **cfg)
        self.sources_full_list = self._input_source
        # redefine _input_source to only be the names so MettaAgent can find the components
        # it's ugly but it maintains consistency in the YAML config
        self._input_source = []
        for src_cfg in self.sources_full_list:
            self._input_source.append(src_cfg['source_name'])

    @property
    def ready(self):
        return self._ready

    def setup(self, input_source_components=None):
        if self._ready:
            return

        # shouldn't this check if it's a dict or not?
        self.input_source_components = input_source_components
        self._in_tensor_shape = []
        self._out_tensor_shape = []

        self.dims = []
        self.out_shapes = []
        for src_cfg in self.sources_full_list:
            source_name = src_cfg['source_name']
            
            processed_size = self.input_source_components[source_name]._out_tensor_shape.copy()
            self._in_tensor_shape.append(processed_size)
            if src_cfg.get('slice') is not None:
                slice_range = src_cfg['slice']
                if isinstance(slice_range, omegaconf.listconfig.ListConfig):
                    slice_range = list(slice_range)
                if not (isinstance(slice_range, (list, tuple)) and len(slice_range) == 2):
                    raise ValueError(f"'slice' must be a two-element list/tuple for source {source_name}.")

                start, end = slice_range
                slice_dim = src_cfg.get("dim", None)
                if slice_dim is None:
                    raise ValueError(f"For slice 'dim' must be specified for source {source_name}.")
                length = end - start
                src_cfg['_slice_params'] = {
                    'start': start,
                    'length': length,
                    'dim': slice_dim
                }
# ----- note to self: need to figure out what we want to do with processed_size since it's just the length of the one dim ---
                processed_size[slice_dim] = length

            self.out_shapes.append(processed_size)

            self.dims.append(src_cfg.get("dim", 1)) # check if default dim is good to have or will cause problems

        self._setup_merge_layer()
        self._ready = True

    def _setup_merge_layer(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, td: TensorDict):
        outputs = []
        for src_cfg in self.sources_full_list:
            source_name = src_cfg['source_name']
            self.input_source_components[source_name].forward(td)
            src_tensor = td[source_name]

            if '_slice_params' in src_cfg:
                params = src_cfg['_slice_params']
                src_tensor = torch.narrow(src_tensor, dim=params['dim'], start=params['start'], length=params['length'])
            outputs.append(src_tensor)

        return self._merge(outputs, td)

    def _merge(self, outputs, td):
        raise NotImplementedError("Subclasses should implement this method.")


class ConcatMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self):
        if not all(d == self.dims[0] for d in self.dims):
            raise ValueError(f"For 'concat', all sources must have the same 'dim'. Got dims: {self.dims}")
        self._merge_dim = self.dims[0]
        cat_dim_length = 0
        for size in self.out_shapes:
            cat_dim_length += size[self._merge_dim - 1]
        self._out_tensor_shape = self._in_tensor_shape[0]
        self._out_tensor_shape[self._merge_dim - 1] = cat_dim_length

    def _merge(self, outputs, td):
        merged = torch.cat(outputs, dim=self._merge_dim)
        td[self._name] = merged
        return td


class AddMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self):
        if not all(s == self.sizes[0] for s in self.sizes):
            raise ValueError(f"For 'add', all source sizes must match. Got sizes: {self.sizes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape[self._merge_dim - 1] = self.sizes[0]

    def _merge(self, outputs, td):
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        td[self._name] = merged
        return td


class SubtractMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self):
        if not all(s == self.sizes[0] for s in self.sizes):
            raise ValueError(f"For 'subtract', all source sizes must match. Got sizes: {self.sizes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape[self._merge_dim - 1] = self.sizes[0]

    def _merge(self, outputs, td):
        if len(outputs) != 2:
            raise ValueError("Subtract merge_op requires exactly two sources.")
        merged = outputs[0] - outputs[1]
        td[self._name] = merged
        return td


class MeanMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self):
        if not all(s == self.sizes[0] for s in self.sizes):
            raise ValueError(f"For 'mean', all source sizes must match. Got sizes: {self.sizes}")
        self._merge_dim = self.dims[0]
        self._out_tensor_shape[self._merge_dim - 1] = self.sizes[0]

    def _merge(self, outputs, td):
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        merged = merged / len(outputs)
        td[self._name] = merged
        return td
