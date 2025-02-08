from __future__ import annotations

import omegaconf
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
import torch
import numpy as np

class LayerBase(nn.Module):
    def __init__(self, metta_agent, **cfg):
        cfg = OmegaConf.create(cfg)
        super().__init__()
        self.metta_agent = metta_agent
        self.cfg = cfg
        self.name = cfg.name
        self.input_source = cfg.get('input_source', None)
        self.output_size = cfg.get('output_size', None)
        self.clip_multiplier = cfg.get('clip_multiplier', None)
        # can change the above to default to None and then handle it in the parameter_layer_helper

    def set_input_size_and_initialize_layer(self):
        if self.input_source is None:
            if self.output_size is None:
                raise ValueError(f"Output size is not set for layer {self.name}")
        else:
            self.metta_agent.components[self.input_source].set_input_size_and_initialize_layer()
            self.input_size = self.metta_agent.components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

        self._initialize_layer()

    def _initialize_layer(self):
        pass

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if self.input_source is not None:
            self.metta_agent.components[self.input_source].forward(td)

        td[self.name] = self.layer(td[self.input_source])

        return td
        
    def clip_weights(self):
        pass
    def get_l2_reg_loss(self):
        pass
    def get_l2_init_loss(self):
        pass
    def update_l2_init_weight_copy(self):
        pass
    def get_effective_rank(self):
        pass
    
class ParameterizedLayer(LayerBase):
    def __init__(self, metta_agent, **cfg): 
        cfg = OmegaConf.create(cfg)
        # this should be only for the layer that calls this
        for key, value in cfg.items():
            setattr(self, key, value)
        self.global_clip_multiplier = metta_agent.clip_range
        super().__init__(metta_agent, **cfg)

    def _parameter_layer_helper(self):
        # attributes = ['clip_scale', 'l2_norm_scale', 'l2_init_scale', 'effective_rank', 'initialization']
        # for attr in attributes:
        #     if attr in self.cfg:
        #         setattr(self, attr, self.cfg[attr])
        #     else:
        #         setattr(self, attr, None)

        self.largest_weight = self.initialize_layer()

        if self.clip_scale:
            self.clip_value = self.metta_agent.clip_multiplier * self.largest_weight * self.clip_scale

        if self.l2_init_scale:
            self.initial_weights = self.layer.weight.data.clone()

        self.nonlinearity = self.cfg.get('nonlinearity', 'ReLU')
        self.weights_data = self.layer.weight.data
        if self.nonlinearity is not None:
            self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())
            self.weights_data = self.layer[0].weight.data


    def _initialize_weights(self):
        '''
        Assumed that this is run before appending a nonlinear layer.
        '''
        fan_in, fan_out = self.layer.weight.shape

        if self.initialization is None or self.initialization == 'Orthogonal':
            if self.nonlinearity == 'Tanh':
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(self.layer.weight, gain=gain)
            largest_weight = self.layer.weight.max().item()
        elif self.initialization == 'Xavier':
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(self.layer.weight)
        elif self.initialization == 'Normal':
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(self.layer.weight, mean=0, std=largest_weight)
        elif self.initialization == 'Max_0_01':
            #set to uniform with largest weight = 0.01
            largest_weight = 0.01
            nn.init.uniform_(self.layer.weight, a=-largest_weight, b=largest_weight)

        if hasattr(self.layer, "bias") and isinstance(self.layer.bias, torch.nn.parameter.Parameter):
            self.layer.bias.data.fill_(0)

        return largest_weight

    def clip_weights(self):
        if self.clip_scale:
            with torch.no_grad():
                self.weights_data = self.weights_data.clamp(-self.clip_value, self.clip_value)

    def get_l2_reg_loss(self) -> torch.Tensor:
        '''
        Also known as Weight Decay Loss or L2 Ridge Regularization
        '''
        l2_reg_loss = torch.tensor(0.0, device=self.weights_data.device)
        if self.l2_norm_scale:
            l2_reg_loss = (torch.sum(self.weights_data ** 2))*self.l2_norm_scale
        return l2_reg_loss

    def get_l2_init_loss(self) -> torch.Tensor:
        '''
        Also known as Delta Regularization Loss
        '''
        l2_init_loss = torch.tensor(0.0, device=self.weights_data.device)
        if self.l2_init_scale:
            l2_init_loss = torch.sum((self.weights_data - self.initial_weights) ** 2) * self.l2_init_scale
        return l2_init_loss
    
    def update_l2_init_weight_copy(self, alpha: float = 0.9):
        '''
        Potentially useful to prevent catastrophic forgetting.
        Update the initial weights copy with a weighted average of the previous and current weights.
        '''
        self.initial_weights = (self.initial_weights * alpha + self.weights_data * (1 - alpha)).clone()
    
    def get_effective_rank(self, delta: float = 0.01):
        """
        Computes the effective rank of a matrix based on the given delta value.
        Effective rank formula:
        srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
        See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning' by A. Kumar et al.
        """
        # Singular value decomposition. We only need the singular value matrix.
        _, S, _ = torch.linalg.svd(self.weights_data.detach())
        
        # Calculate the cumulative sum of singular values
        total_sum = S.sum()
        cumulative_sum = torch.cumsum(S, dim=0)
        
        # Find the smallest k that satisfies the effective rank condition
        threshold = (1 - delta) * total_sum
        effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1  # Add 1 for 1-based indexing
        
        return {'name': self.name, 'effective_rank': effective_rank}



class MettaLayerBase(nn.Module):
    '''
    This is the base class and instructions for custom layers in the stead of MettaLayers.
    '''
    def __init__(self, metta_agent, **cfg):
        super().__init__()
        self.metta_agent = metta_agent
        self.cfg = cfg
        #required attributes
        self.name = None
        self.input_source = None
        self.output_size = None

    def set_input_size_and_initialize_layer(self):
        '''
        Recursively set the input size for the component above your layer.
        This is necessary unless you are a top layer, in which case, you can skip this.
        self.metta_agent.components[self.input_source].set_input_source_size()
        
        Set your input size to be the output size of the layer above you or otherwise ensure that this is the case.
        self.input_size = self.metta_agent.components[self.input_source].output_size

        With your own input and output sizes set, initialize your layer, if necessary.
        self.layer = ...

        '''
        raise NotImplementedError(f"The method set_input_source_size() is not implemented yet for object {self.__class__.__name__}.")

    def forward(self, td: TensorDict):
        '''
        First, ensure we're not recomputing in case your layer is already computed.
        if self.name in td:
            return td[self.name]

        First, recursively compute the input to the layers above this layer.
        Skip this if you are a top layer.
        if isinstance(self.input_source, list):
            for src in self.input_source:
                self.metta_agent.components[src].forward(td) 
        else:
            self.metta_agent.components[self.input_source].forward(td)

        Compute this layer's output (assuming you have a .layer attribute).
        Write your layer's name on your output so the next layer can find it.
        if isinstance(self.input_source, list):
            inputs = [td[src] for src in self.input_source]
            x = torch.cat(inputs, dim=-1)
            td[self.name] = self.layer(x)
        else:
            td[self.name] = self.layer(td[self.input_source])

        Pass the full td back.
        return td
        '''
        raise NotImplementedError(f"The method forward() is not implemented yet for object {self.__class__.__name__}.")
    

class Linear(ParameterizedLayer):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.Linear(
            self.input_size,
            self.output_size,
            **self.cfg.get('nn_params', {})
        )
        self._parameter_layer_helper()
        self._initialize_weights()

class Conv1d(ParameterizedLayer):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.Conv1d(
            self.input_size,
            self.output_size,
            **self.cfg.get('nn_params', {})
        )
        self._parameter_layer_helper()
        self._initialize_weights()

class Conv2d(ParameterizedLayer):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.Conv2d(
            self.input_size,
            self.output_size,
            **self.cfg.get('nn_params', {})
        )
        self._parameter_layer_helper()
        self._initialize_weights()

class MaxPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.MaxPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class MaxPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.MaxPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveAvgPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveAvgPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveMaxPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveMaxPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )
        
class AvgPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AvgPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AvgPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AvgPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class Dropout(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Dropout(
            **self.cfg.get('nn_params', {'p': 0.5})
        )
    
class Dropout2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Dropout2d(
            **self.cfg.get('nn_params', {'p': 0.5})
        )
    
class AlphaDropout(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.AlphaDropout(
            **self.cfg.get('nn_params', {'p': 0.5})
        )

class BatchNorm1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.BatchNorm1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class BatchNorm2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
            self.layer = nn.BatchNorm2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class Flatten(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Flatten()

class Identity(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Identity()
        

class MergeLayerBase(MettaLayerBase):
    def __init__(self, metta_agent, **cfg):
        super().__init__(metta_agent, **cfg)
        cfg = omegaconf.OmegaConf.create(cfg)
        self.sources_list = list(cfg.sources)
        # self.sources_cfg = self.cfg.get('sources')
        # if self.sources_cfg is None:
        #     raise ValueError("MergeLayer requires a 'sources' configuration key.")
        # if not isinstance(self.sources_cfg, omegaconf.listconfig.ListConfig):
        #     raise ValueError("The 'sources' configuration must be a list of dictionaries.")
        self.default_dim = -1

    def set_input_size_and_initialize_layer(self):
        sizes = []
        dims = []
        for idx, src_cfg in enumerate(self.sources_list):
            # if not isinstance(src_cfg, dict):
            #     raise ValueError(
            #         f"Each source configuration must be a dictionary. "
            #         f"Invalid format at index {idx}: {src_cfg}"
            #     )
            
            source_name = src_cfg.source_name
            
            self.metta_agent.components[source_name].set_input_size_and_initialize_layer()
            full_source_size = self.metta_agent.components[source_name].output_size

            if src_cfg.get('slice') is not None:
                slice_range = src_cfg.slice
                if not (isinstance(slice_range, (list, tuple)) and len(slice_range) == 2):
                    raise ValueError(f"'slice' must be a two-element list/tuple for source {source_name}.")
                processed_size = slice_range[1] - slice_range[0]
            else:
                processed_size = full_source_size

            sizes.append(processed_size)
            dims.append(src_cfg.get("dim", self.default_dim))

        self._set_input_size_and_initialize_layer(sizes, dims)

    def _set_input_size_and_initialize_layer(self, sizes, dims):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, td: TensorDict):
        outputs = []
        for src_cfg in self.sources_cfg:
            source_name = src_cfg.get("source")
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
    def _set_input_size_and_initialize_layer(self, sizes, dims):
        if not all(d == dims[0] for d in dims):
            raise ValueError(f"For 'concat', all sources must have the same 'dim'. Got dims: {dims}")
        self.merge_dim = dims[0]
        #this should calculate it's own output size
        self.output_size = sum(sizes)

    def _merge(self, outputs, td):
        merged = torch.cat(outputs, dim=self.merge_dim)
        td[self.name] = merged
        return td


class AddMergeLayer(MergeLayerBase):
    def _set_input_size_and_initialize_layer(self, sizes, dims):
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
    def _set_input_size_and_initialize_layer(self, sizes, dims):
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
    def _set_input_size_and_initialize_layer(self, sizes, dims):
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