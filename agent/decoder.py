
from sample_factory.model.decoder import MlpDecoder
from sample_factory.utils.attr_dict import AttrDict


class Decoder(MlpDecoder):
    def __init__(self, input_size: int):
        super().__init__(
            AttrDict({
                'decoder_mlp_layers': [],
                'nonlinearity': 'elu',
            }),
            input_size
        )
