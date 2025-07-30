import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class MeanPool(LayerBase):
    """
    Performs mean pooling over a specified dimension.

    This is useful for aggregating variable-length token sequences into a fixed-size representation.

    Args:
        dim (int): The dimension to pool over (not including batch dimension).
                   For token observations with shape [M, feat_dim], use dim=0 to pool over tokens.
        keepdim (bool): Whether to keep the pooled dimension. Default is False.
        **cfg: Additional configuration for LayerBase.

    Input TensorDict:
        - From source: Tensor of shape [B, M, feat_dim] where B is batch, M is variable token count

    Output TensorDict:
        - self._name: Tensor with shape [B, feat_dim] after pooling over M dimension
    """

    def __init__(self, dim: int = 0, keepdim: bool = False, **cfg) -> None:
        super().__init__(**cfg)
        self._dim = dim  # This is the dimension in the shape without batch
        self._keepdim = keepdim

    def _make_net(self) -> None:
        # Input shape is without batch dimension, e.g., [M, feat_dim]
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

        # The actual dimension to pool over includes the batch dimension
        pool_dim = self._dim + 1  # +1 because of batch dimension

        # Handle case where we might have a mask
        if "obs_mask" in td:
            # obs_mask is True for elements to be masked (ignored)
            # We need to compute mean only over non-masked elements
            mask = td["obs_mask"]  # Shape: [B, M]

            # Convert mask to float (0 for masked, 1 for valid)
            valid_mask = (~mask).float()

            # Expand mask to match x dimensions if needed
            if x.dim() > mask.dim():
                # x is [B, M, feat_dim], mask is [B, M]
                for _ in range(x.dim() - mask.dim()):
                    valid_mask = valid_mask.unsqueeze(-1)

            # Apply mask and compute sum
            masked_x = x * valid_mask
            sum_x = torch.sum(masked_x, dim=pool_dim, keepdim=self._keepdim)

            # Count valid elements
            count = torch.sum(valid_mask, dim=pool_dim, keepdim=self._keepdim)
            count = torch.clamp(count, min=1.0)  # Avoid division by zero

            # Compute mean
            output = sum_x / count
        else:
            # Simple mean without masking
            output = torch.mean(x, dim=pool_dim, keepdim=self._keepdim)

        td[self._name] = output
        return td
