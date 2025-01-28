import numpy as np
import torch
from torch import nn
import omegaconf

class WeightTransformer():
    def __init__(self, cfg):
        self._clip_scales = []
        self._l2_norm_scales = []
        self._effective_rank = []
        self.rank_matrix = None
        self._layers = []
        self.cfg = cfg
        self._global_clipping_coeff = cfg.agent.clipping_coeff

    def add_layer(self, layer, key, layer_idx):
        layer_attributes = self._get_layer_attributes(self.cfg.agent.get(key), layer_idx)

        largest_weight = self._initialize(layer, layer_attributes)

        self._clip_scales.append(layer_attributes['clip_scales'] * largest_weight * self._global_clipping_coeff if layer_attributes['clip_scales'] is not None else None)
        self._l2_norm_scales.append(layer_attributes['l2_norm_scales'])
        self._effective_rank.append(layer_attributes['effective_rank'])

        layer.name = f"{key}_{layer_idx}"

        self._layers.append(layer)

    def key(self, key):
        return lambda layer, idx: self.add_layer(layer, key, idx)

    def _get_layer_attributes(self, network_cfg, layer_idx):
        attributes = ['clip_scales', 'l2_norm_scales', 'effective_rank', 'initialization', 'nonlinearity']
        layer_attributes = {}

        for attr in attributes:
            attribute = getattr(network_cfg, attr, None)
            if isinstance(attribute, omegaconf.listconfig.ListConfig):
                attribute = list(attribute)
                layer_attributes[attr] = attribute[layer_idx - 1] if layer_idx - 1 < len(attribute) else None
            else:
                layer_attributes[attr] = attribute

        return layer_attributes
    
    def _initialize(self, layer, layer_attributes):
        fan_in, fan_out = layer.weight.size(1), layer.weight.size(0)
        # default to orthogonal initialization
        if layer_attributes['initialization'] is None or layer_attributes['initialization'] == 'Orthogonal':
            if layer_attributes['nonlinearity'] == 'Tanh':
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(layer.weight, gain=gain)
            largest_weight = layer.weight.max().item()
        elif layer_attributes['initialization'] == 'Xavier':
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(layer.weight)
        elif layer_attributes['initialization'] == 'Normal':
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(layer.weight, mean=0, std=largest_weight)

        if hasattr(layer, "bias") and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        return largest_weight
    
    def clip_weights(self):
        for layer, clip in zip(self._layers, self._clip_scales):
            if clip is not None and clip != 0:
                layer.weight.data.clamp_(clip, -clip)

    def get_l2_norm_loss(self):
        l2_norm_loss = 0
        for layer, l2_norm_scale in zip(self._layers, self._l2_norm_scales):
            l2_norm = (l2_norm_scale * (torch.sum(layer.weight ** 2)) if l2_norm_scale or l2_norm_scale != 0 else 0)
            l2_norm_loss += l2_norm
        return l2_norm_loss
    
    def get_effective_ranks(self, delta: float = 0.01):
        effective_ranks = []
        for layer, effective_rank in zip(self._layers, self._effective_rank):
            if effective_rank:
                self.rank_matrix = layer.weight.data.detach()
                effective_rank = self._compute_effective_rank(delta)
                effective_ranks.append({'name': layer.name, 'effective_rank': effective_rank})
        return effective_ranks
    
    def _compute_effective_rank(self, delta: float = 0.01):
        """
        Computes the effective rank of a matrix based on the given delta value.
        Effective rank formula:
        srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
        See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning' by A. Kumar et al.
        """
        # Singular value decomposition. We only need the singular value matrix.
        _, S, _ = torch.linalg.svd(self.rank_matrix)
        
        # Calculate the cumulative sum of singular values
        total_sum = S.sum()
        cumulative_sum = torch.cumsum(S, dim=0)
        
        # Find the smallest k that satisfies the effective rank condition
        threshold = (1 - delta) * total_sum
        effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1  # Add 1 for 1-based indexing
        
        return effective_rank 