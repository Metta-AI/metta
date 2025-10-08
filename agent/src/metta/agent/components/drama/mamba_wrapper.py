# Copyright (c) 2023, Albert Gu, Tri Dao.

import copy
import math
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from metta.agent.components.mamba_ssm.modules.block import Block
from metta.agent.components.mamba_ssm.modules.mamba2 import Mamba2
from metta.agent.components.mamba_ssm.modules.mha import MHA
from metta.agent.components.mamba_ssm.modules.mlp import MLP
from metta.agent.components.mamba_ssm.utils.generation import GenerationMixin
from metta.agent.components.mamba_ssm.ops.triton.layer_norm import (
    RMSNorm,
    layer_norm_fn,
    rms_norm_fn,
)


@dataclass
class MambaConfig:
    d_model: int
    n_layer: int
    d_intermediate: int
    stoch_dim: int
    action_dim: int
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list[int] = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    pff_cfg: dict = field(default_factory=dict)
    dropout_p: float = 0.0
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    use_triton_norms: bool = True


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    pff_cfg=None,
    dropout_p=0.0,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    *,
    guard_triton: bool = True,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    if pff_cfg is None:
        pff_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba2")
        if ssm_layer != "Mamba2":
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba2")
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    if guard_triton and rms_norm and RMSNorm is None:
        raise RuntimeError(
            "DRAMA MambaWrapperModel requires Triton RMSNorm kernels; install torchao/triton or disable rms_norm"
        )
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(MLP, hidden_features=d_intermediate, out_features=d_model, **pff_cfg, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, embed_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim, **factory_kwargs)

    def forward(self, feat):
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, position : position + 1, :]
        return feat


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        stoch_dim: int,
        action_dim: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        pff_cfg=None,
        dropout_p: float = 0.0,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.action_dim = action_dim
        self.feat_dim = d_model

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        norm_cls = RMSNorm if rms_norm and RMSNorm is not None else nn.LayerNorm
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, d_model, bias=True, **factory_kwargs),
            norm_cls(d_model, eps=norm_epsilon, **factory_kwargs),
            nn.SiLU(),
        )

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        else:
            self.dropout = nn.Dropout(dropout_p)  # "Attention is all you need sec 5.4 dropout"
        self.dropout_p = dropout_p

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    pff_cfg=pff_cfg,
                    dropout_p=dropout_p,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        norm_final = nn.LayerNorm if not rms_norm or RMSNorm is None else RMSNorm
        self.norm_f = norm_final(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, samples, action, inference_params=None, **mixer_kwargs):
        action = F.one_hot(action.long(), self.action_dim).float()
        hidden_states = self.stem(torch.cat([samples, action], dim=-1))

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params, **mixer_kwargs)
        if not self.fused_add_norm:
            hidden_states = self.dropout(hidden_states)
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            norm_f_bias = getattr(self.norm_f, "bias", None)
            hidden_states, _ = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                norm_f_bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return hidden_states


class MambaWrapperModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        stoch_dim = config.stoch_dim
        action_dim = config.action_dim
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        pff_cfg = config.pff_cfg
        dropout_p = config.dropout_p
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            stoch_dim=stoch_dim,
            action_dim=action_dim,
            ssm_cfg=ssm_cfg,
            pff_cfg=pff_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            dropout_p=dropout_p,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, samples, action, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(samples, action, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        # lm_logits = self.lm_head(hidden_states)
        return hidden_states
