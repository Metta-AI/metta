import omegaconf
import torch
from tensordict import TensorDict
from agent.lib.metta_layer import LayerBase

class MergeLayerBase(LayerBase):
    def __init__(self, metta_agent, **cfg):
        super().__init__(metta_agent, **cfg)
        cfg = omegaconf.OmegaConf.create(cfg)
        self.sources_list = list(cfg.sources)
        self.default_dim = -1
        self.name = cfg.name
        self.metta_agent = metta_agent

    def setup_layer(self):
        sizes = []
        dims = []
        for idx, src_cfg in enumerate(self.sources_list):       
            source_name = src_cfg.source_name
            # delete this
            print(f"input source: {source_name}")     
            
            self.metta_agent.components[source_name].setup_layer()
            full_source_size = self.metta_agent.components[source_name].output_size

            if src_cfg.get('slice') is not None:
                slice_range = src_cfg.slice
                if isinstance(slice_range, omegaconf.listconfig.ListConfig):
                    slice_range = list(slice_range)
                if not (isinstance(slice_range, (list, tuple)) and len(slice_range) == 2):
                    raise ValueError(f"'slice' must be a two-element list/tuple for source {source_name}.")
                processed_size = slice_range[1] - slice_range[0]
            else:
                processed_size = full_source_size

            sizes.append(processed_size)
            dims.append(src_cfg.get("dim", self.default_dim))

        self._setup_merge_layer(sizes, dims)
        
    def _setup_merge_layer(self, sizes, dims):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, td: TensorDict):
        outputs = []
        for src_cfg in self.sources_list:
            source_name = src_cfg.source_name
            self.metta_agent.components[source_name].forward(td)
            src_tensor = td[source_name]

            if "slice" in src_cfg:
                start, end = src_cfg["slice"]
                slice_dim = src_cfg.get("dim", self.default_dim)
                length = end - start
                src_tensor = torch.narrow(src_tensor, dim=slice_dim, start=start, length=length)
            outputs.append(src_tensor)
        
        return self._merge(outputs, td)

    def _merge(self, outputs, td):
        raise NotImplementedError("Subclasses should implement this method.")


class ConcatMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self, sizes, dims):
        if not all(d == dims[0] for d in dims):
            raise ValueError(f"For 'concat', all sources must have the same 'dim'. Got dims: {dims}")
        self.merge_dim = dims[0]
        self.output_size = sum(sizes)

    def _merge(self, outputs, td):
        merged = torch.cat(outputs, dim=self.merge_dim)
        td[self.name] = merged
        return td


class AddMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self, sizes, dims):
        if not all(s == sizes[0] for s in sizes):
            raise ValueError(f"For 'add', all source sizes must match. Got sizes: {sizes}")
        self.output_size = sizes[0]

    def _merge(self, outputs, td):
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        td[self.name] = merged
        return td


class SubtractMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self, sizes, dims):
        if not all(s == sizes[0] for s in sizes):
            raise ValueError(f"For 'subtract', all source sizes must match. Got sizes: {sizes}")
        self.output_size = sizes[0]

    def _merge(self, outputs, td):
        if len(outputs) != 2:
            raise ValueError("Subtract merge_op requires exactly two sources.")
        merged = outputs[0] - outputs[1]
        td[self.name] = merged
        return td


class MeanMergeLayer(MergeLayerBase):
    def _setup_merge_layer(self, sizes, dims):
        if not all(s == sizes[0] for s in sizes):
            raise ValueError(f"For 'mean', all source sizes must match. Got sizes: {sizes}")
        self.output_size = sizes[0]

    def _merge(self, outputs, td):
        merged = outputs[0]
        for tensor in outputs[1:]:
            merged = merged + tensor
        merged = merged / len(outputs)
        td[self.name] = merged
        return td