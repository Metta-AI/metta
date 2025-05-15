from typing import Any, Dict

from typing_extensions import override

from metta.agent.lib.metta_layer import LayerBase


class ObsShaper(LayerBase):
    """
    This class does the following:
    1) permutes input observations from [B, H, W, C] or [B, TT, H, W, C] to [..., C, H, W]
    2) inspects tensor shapes, ensuring that input observations match expectations from the environment
    3) inserts batch size, TT, and B * TT into the tensor dict for certain other layers in the network to use
       if they need reshaping.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent is instantiated
    and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, obs_shape, num_objects, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        self._out_tensor_shape = [self._obs_shape[2], self._obs_shape[0], self._obs_shape[1]]
        self._output_size = num_objects

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        x = data["x"]

        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            expected_shape = f"[B(, T), {', '.join(str(dim) for dim in space_shape)}]"
            actual_shape = f"{list(x_shape)}"
            raise ValueError(
                f"Shape mismatch error:\n"
                f"x.shape: {x.shape}\n"
                f"self._obs_shape: {self._obs_shape}\n"
                f"Expected tensor with shape {expected_shape}\n"
                f"Got tensor with shape {actual_shape}\n"
                f"The last {space_n} dimensions should match {tuple(space_shape)}"
            )

        # Validate overall tensor dimensionality with improved error message
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError(
                f"Invalid input tensor dimensionality:\n"
                f"Expected tensor with {space_n + 1} or {space_n + 2} dimensions\n"
                f"Got tensor with {x_n} dimensions: {list(x_shape)}\n"
                f"Expected format: [batch_size(, time_steps), {', '.join(str(dim) for dim in space_shape)}]"
            )

        x = x.reshape(B * TT, *space_shape)
        x = x.float()

        # conv expects [batch, channel, w, h]. Below is hardcoded for [batch, w, h, channel]
        x = self._permute(x)

        data["_TT_"] = TT
        data["_batch_size_"] = B
        data["_BxTT_"] = B * TT
        data[self._name] = x
        return data

    def _permute(self, x):
        if x.device.type == "mps":
            x = self._mps_permute(x)
        else:
            x = x.permute(0, 3, 1, 2)
        return x

    def _mps_permute(self, x):
        """For compatibility with MPS, it throws an error on .permute()"""
        bs, h, w, c = x.shape
        x = x.contiguous().view(bs, h * w, c)
        x = x.transpose(1, 2)
        x = x.contiguous().view(bs, c, h, w)
        return x
