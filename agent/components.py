import torch.nn as nn
from sample_factory.model.decoder import MlpDecoder
from sample_factory.utils.attr_dict import AttrDict
from omegaconf import ListConfig
import hydra
from copy import deepcopy
import omegaconf
from omegaconf import OmegaConf
# from agent.metta_agent import MettaAgent

# class Layer(nn.Module):
#     def __init__(self, name: str, input_size: int, output_size: int = None, layer_type: nn.Module = nn.Linear):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size if output_size is not None else input_size # useful for non-linear layers
#         self.layer_type = layer_type
#         self.layer = layer_type(input_size, output_size)
#         self.layer.name = name
#         self.layer.initialize_weights()
#         self.layer.normalize_weights()
#         self.layer.clip_weights()
#         self.layer.get_losses()

#     def forward(self, x):
#         return self.layer(x)

#     def get_out_size(self):
#         return self.output_size

#     def get_in_size(self):
#         return self.input_size

# class Composer(nn.Module):
#     def __init__(self, net_cfg, input, output, MettaAgent: MettaAgent):
#     # def __init__(self, layers: ListConfig, input, output, MettaAgent: MettaAgent):
#         super().__init__()
#         self.input = self.get_size(input, "input")
#         self.output = self.get_size(output, "output")
#         self.MettaAgent = MettaAgent
#         self.input_layers = net_cfg.layers


#         # make the layers
#         self.layers = nn.ModuleList()
#         for layer in self.input_layers:
#             layer.input_size = self.input
#             self.input = layer.output_size
#             layer = Layer(layer.name, layer.input_size, layer.output_size)
#             self.layers.append(layer)

#         # self.layers = nn.ModuleList([
#         #     hydra.utils.instantiate(layer)
#         #     for layer in layers
#         # ])

#     def get_size(self, value, type):
#         if isinstance(value, int):
#             return value
#         elif isinstance(value, str):
#             attr = getattr(self.MettaAgent, value, None)
#             return attr.get_out_size() if type == "output" else attr.get_in_size()
#         elif isinstance(value, omegaconf.listconfig.ListConfig):
#             size = 0
#             for layer in value:
#                 size += self.get_size(layer, type)
#             return size
#         else:
#             raise ValueError(f"Invalid value type: {type(value)}")
        
#     # need to figure out how to route the correct input in
#     # maybe do this in MettaAgent?


#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def get_out_size(self):
#         return self.layers[-1].output_size
    
#     def get_in_size(self):
#         return self.layers[0].input_size

class Decoder(MlpDecoder):
    def __init__(self, input_size: int):
        super().__init__(
            AttrDict({
                'decoder_mlp_layers': [],
                'nonlinearity': 'elu',
            }),
            input_size
        )

    def get_out_size(self):
        return self.output_size
    
    def get_in_size(self):
        return self.input_size

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
    
    def get_out_size(self):
        return self.layers[-1].output_size
    
    def get_in_size(self):
        return self.layers[0].input_size
    
    #create skip connection
    def create_skip_connection(self, x):
        return x + self.forward(x)
    
    def get_skip_connection_size(self):
        return self.get_in_size()
    
    #create residual connection
    def create_residual_connection(self, x):
        return x + self.forward(x)
    
    def get_residual_connection_size(self):
        return self.get_in_size()
    
    #create residual connection with skip connection
    def create_residual_connection_with_skip(self, x):
        return self.create_skip_connection(x) + self.create_residual_connection(x)
    
    def get_residual_connection_with_skip_size(self):
        return self.get_in_size()
    
    
    
class MultiStack(Stack):
    def __init__(self,
                template: OmegaConf,
                repeat: int,
                **kwargs):
        layers = []
        for i in range(repeat):
            layers.append(deepcopy(template))
        super().__init__(layers, **kwargs)
