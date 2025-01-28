import torch.nn as nn
from sample_factory.model.decoder import MlpDecoder
from sample_factory.utils.attr_dict import AttrDict
from omegaconf import ListConfig
import hydra
from copy import deepcopy
from omegaconf import OmegaConf

class Decoder(MlpDecoder):
    def __init__(self, input_size: int):
        super().__init__(
            AttrDict({
                'decoder_mlp_layers': [],
                'nonlinearity': 'elu',
            }),
            input_size
        )

class LinearLayer(nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        l2_norm_scale: float,
        l2_init_scale: float,
        clip_scale: float):

        super().__init__(input_size, output_size)
        self.l2_norm_scale = l2_norm_scale
        self.l2_init_scale = l2_init_scale
        self.clip_scale = clip_scale

    def normalize_weights(self):
        # do all the clippings and l2 normings here
        pass

    def clip_weights(self):
        # do all the clippings here
        pass

    def get_losses(self):
        return 0

    def initialize_weights(self):
        # do all the initializations here
        pass

class Stack(nn.Module):
    def __init__(
        self,
        layers: ListConfig,
        input_size: int,
        output_size: int,
        skip: bool = False,
        nonlinearity: nn.Module = nn.ReLU()):

        super().__init__()

        # Make the first layer take the input size, and the last layer output the output size
        layer_size = input_size
        for layer_cfg in layers:
            layer_cfg.input_size = layer_size
            layer_size = layer_cfg.size
            layer_cfg.output_size = layer_size
        layers[-1].output_size = output_size

        self.layers = nn.ModuleList([
            hydra.utils.instantiate(layer_cfg)
            for layer_cfg in layers
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiStack(Stack):
    def __init__(self,
                template: OmegaConf,
                repeat: int,
                **kwargs):
        layers = []
        for i in range(repeat):
            layers.append(deepcopy(template))
        super().__init__(layers, **kwargs)
