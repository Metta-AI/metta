from typing import List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class TransformerMemoryCore(LayerBase):
    """
    A Transformer-based recurrent core for the MettaAgent.

    This component processes a sequence consisting of:
    1. Current observation features.
    2. Memory tokens from the previous timestep.
    3. A learnable action-specific "class" token (action-clts).
    4. A learnable value-specific "class" token (value-clts).

    It uses a Transformer encoder to update these tokens. The updated memory tokens
    are passed to the next timestep. The updated action-clts and value-clts
    are used as input precursors for the action and value heads, respectively.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        num_mem_tokens: int = 1,  # Number of memory tokens to use
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,  # TransformerEncoderLayer default is False
        norm_first: bool = False,  # TransformerEncoderLayer default is False
        **cfg,  # Captures LayerBase configs like _sources, _name, _nn_params, etc.
    ):
        super().__init__(**cfg)

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_mem_tokens = num_mem_tokens
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first

        # In LayerBase, _nn_params is usually for the main nn.Module (self._net)
        # We are directly using d_model etc. for TransformerEncoderLayer.
        # If _nn_params is still expected by LayerBase for other things,
        # ensure it's handled or passed appropriately.
        # self.output_size from cfg (passed via **cfg) is used by LayerBase.
        # We should ensure it's consistent with d_model.
        if hasattr(self, "output_size") and self.output_size != self.d_model:
            raise ValueError(
                f"TransformerMemoryCore: output_size ({self.output_size}) from config "
                f"must match d_model ({self.d_model})."
            )
        elif not hasattr(self, "output_size"):
            # LayerBase uses self.output_size to set self._out_tensor_shape
            # if _make_net doesn't set it directly.
            self.output_size = self.d_model

        # Learnable tokens, initialized as parameters
        self.mem_token_params = nn.Parameter(torch.randn(self.num_mem_tokens, self.d_model))
        self.action_clts_param = nn.Parameter(torch.randn(1, self.d_model))
        self.val_clts_param = nn.Parameter(torch.randn(1, self.d_model))

        # Input projection layer if the observation feature dimension doesn't match d_model
        # This will be determined in setup() once input shapes are known.
        self.input_projection: Optional[nn.Module] = None

    def _make_net(self) -> None:
        """
        Constructs the Transformer encoder network.
        This method is called by LayerBase.setup() after input shapes are determined.
        """
        if not self._in_tensor_shapes or not self._in_tensor_shapes[0]:
            raise RuntimeError(
                "TransformerMemoryCore: _in_tensor_shapes not properly set by LayerBase.setup(). "
                "Ensure component has sources configured."
            )
        obs_feature_dim = self._in_tensor_shapes[0][0]

        if obs_feature_dim != self.d_model:
            self.input_projection = nn.Linear(obs_feature_dim, self.d_model)
            # Initialize projection weights if desired
            # nn.init.xavier_uniform_(self.input_projection.weight)
            # if self.input_projection.bias is not None:
            #     nn.init.zeros_(self.input_projection.bias)
        else:
            self.input_projection = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
        )
        self._net = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # LayerBase expects _out_tensor_shape to be set if output is not just from _net
        # Here, the conceptual output of the "core" is d_model (features of tokens)
        self._out_tensor_shape = [self.d_model]

    @torch.compile(disable=True)  # Potentially, if issues arise with torch.compile
    def _forward(self, td: TensorDict) -> TensorDict:
        if not self._sources or not self._sources[0] or "name" not in self._sources[0]:
            raise RuntimeError("TransformerMemoryCore: Component sources not properly configured or accessible.")

        current_obs_features = td[self._sources[0]["name"]]

        if self.input_projection is None:
            raise RuntimeError("TransformerMemoryCore: _make_net() must be called (via setup()) before _forward().")

        projected_obs_features = self.input_projection(current_obs_features)

        if projected_obs_features.dim() == 2:
            current_obs_seq = projected_obs_features.unsqueeze(1)
        elif projected_obs_features.dim() == 3:
            current_obs_seq = projected_obs_features
        else:
            raise ValueError(f"Unsupported observation feature shape: {projected_obs_features.shape}")

        batch_size = current_obs_seq.size(0)
        # prev_mem_tokens is expected from Experience buffer (via PufferTrainer._rollout and td["state"])
        # with shape (num_mem_tokens, B, d_model).
        prev_mem_tokens_from_state = td.get("state", None)

        if prev_mem_tokens_from_state is None:
            # Initialize memory tokens for the first step, shape (B, num_mem_tokens, d_model) for batch-first Transformer
            prev_mem_tokens = self.mem_token_params.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Arrives as (num_mem_tokens, B, d_model), permute to (B, num_mem_tokens, d_model) for batch-first concat
            prev_mem_tokens = prev_mem_tokens_from_state.permute(1, 0, 2)

        prev_mem_tokens = prev_mem_tokens.to(current_obs_seq.device)

        action_clts = self.action_clts_param.unsqueeze(0).expand(batch_size, -1, -1)
        val_clts = self.val_clts_param.unsqueeze(0).expand(batch_size, -1, -1)
        action_clts = action_clts.to(current_obs_seq.device)
        val_clts = val_clts.to(current_obs_seq.device)

        full_sequence_parts: List[torch.Tensor] = [current_obs_seq, prev_mem_tokens, action_clts, val_clts]
        full_sequence = torch.cat(full_sequence_parts, dim=1)

        if not self.batch_first:  # Should be True by default for this core
            full_sequence = full_sequence.transpose(0, 1)

        transformer_output_seq = self._net(full_sequence)

        if not self.batch_first:
            transformer_output_seq = transformer_output_seq.transpose(0, 1)

        obs_len = current_obs_seq.size(1)
        start_idx_mem = obs_len
        end_idx_mem = obs_len + self.num_mem_tokens
        # new_mem_tokens are extracted as (B, num_mem_tokens, d_model)
        new_mem_tokens_batch_first = transformer_output_seq[:, start_idx_mem:end_idx_mem, :]

        idx_action_clts = end_idx_mem
        action_precursor = transformer_output_seq[:, idx_action_clts, :]

        idx_val_clts = idx_action_clts + 1
        val_precursor = transformer_output_seq[:, idx_val_clts, :]

        # Detach and permute new_mem_tokens back to (num_mem_tokens, B, d_model) for Experience buffer
        detached_new_mem_tokens_for_state = new_mem_tokens_batch_first.detach().permute(1, 0, 2)
        td["state"] = detached_new_mem_tokens_for_state

        td["core_action_output"] = action_precursor
        td["core_value_output"] = val_precursor
        td[self._name] = action_precursor

        return td
