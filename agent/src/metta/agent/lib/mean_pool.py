import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class MeanPool(LayerBase):
    """
    Performs mean pooling over a specified dimension.

    This is useful for aggregating variable-length token sequences into a fixed-size representation.

    Args:
        dim (int): The dimension to pool over. Default is 1 (token dimension).
        keepdim (bool): Whether to keep the pooled dimension. Default is False.
        **cfg: Additional configuration for LayerBase.

    Input TensorDict:
        - From source: Tensor of shape [..., variable_dim, ...]

    Output TensorDict:
        - self._name: Tensor with the specified dimension reduced by mean pooling
    """

    def __init__(self, dim: int = 1, keepdim: bool = False, **cfg) -> None:
        super().__init__(**cfg)
        self._dim = dim
        self._keepdim = keepdim

    def _make_net(self) -> None:
        # Calculate output shape based on input shape
        in_shape = list(self._in_tensor_shapes[0])

        if self._keepdim:
            out_shape = in_shape.copy()
            out_shape[self._dim] = 1
        else:
            # Remove the pooled dimension
            out_shape = in_shape[: self._dim] + in_shape[self._dim + 1 :]

        self._out_tensor_shape = out_shape

    def _forward(self, td: TensorDict) -> TensorDict:
        x = td[self._sources[0]["name"]]

        # Handle case where we might have a mask
        if "obs_mask" in td:
            # obs_mask is True for elements to be masked (ignored)
            # We need to compute mean only over non-masked elements
            mask = td["obs_mask"]

            # Convert mask to float (0 for masked, 1 for valid)
            valid_mask = (~mask).float()

            # Apply mask and compute sum
            masked_x = x * valid_mask.unsqueeze(-1)  # Expand mask to match x dimensions
            sum_x = torch.sum(masked_x, dim=self._dim, keepdim=self._keepdim)

            # Count valid elements
            count = torch.sum(valid_mask, dim=self._dim, keepdim=self._keepdim)
            count = torch.clamp(count, min=1.0)  # Avoid division by zero

            # Compute mean
            output = sum_x / count.unsqueeze(-1) if not self._keepdim else sum_x / count
        else:
            # Simple mean without masking
            output = torch.mean(x, dim=self._dim, keepdim=self._keepdim)

        td[self._name] = output
        return td
