from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase

class ObsShaper(LayerBase):
    def __init__(self, obs_shape, num_objects, **cfg):
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._output_size = num_objects

    def _forward(self, td: TensorDict):
        x = td["x"]

        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        x = x.reshape(B * TT, *space_shape)
        x = x.float()

        # conv expects [batch, channel, w, h]. Below is hardcoded for [batch, w, h, channel]
        if x.device.type == "mps":
            x = self._mps_permute(x)
        else:
            x = x.permute(0, 3, 1, 2)

        td[self._name] = x
        return td

    def _mps_permute(self, x):
        """For compatibility with MPS, it throws an error on .permute()"""
        bs, h, w, c = x.shape
        x = x.contiguous().view(bs, h * w, c)
        x = x.transpose(1, 2)
        x = x.contiguous().view(bs, c, h, w)
        return x
