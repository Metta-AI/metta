"""
Agent configuration system following the dehydration branch pattern.

This module provides factory functions for creating agent instances,
replacing the string-based agent_mapper with a type-safe configuration approach.
"""

from typing import Any

import torch.nn as nn

# ComponentPolicy implementations (modular architecture)
from metta.agent.component_policies.fast import Fast as ComponentFast
from metta.agent.component_policies.latent_attn_med import LatentAttnMed as ComponentLatentAttnMed
from metta.agent.component_policies.latent_attn_small import LatentAttnSmall as ComponentLatentAttnSmall
from metta.agent.component_policies.latent_attn_tiny import LatentAttnTiny as ComponentLatentAttnTiny

# PyTorch implementations (vanilla models)
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast as PyTorchFast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed as PyTorchLatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall as PyTorchLatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny as PyTorchLatentAttnTiny


def fast(
    obs_space: Any = None,
    obs_width: int = None,
    obs_height: int = None,
    feature_normalizations: dict = None,
    config: dict = None,
    **kwargs,
) -> nn.Module:
    """Create a Fast CNN-based component policy."""
    config = config or {"clip_range": 0, "analyze_weights_interval": 300}
    return ComponentFast(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
        config=config,
    )


def latent_attn_tiny(
    obs_space: Any = None,
    obs_width: int = None,
    obs_height: int = None,
    feature_normalizations: dict = None,
    config: dict = None,
    **kwargs,
) -> nn.Module:
    """Create a Latent Attention Tiny component policy."""
    config = config or {"clip_range": 0, "analyze_weights_interval": 300}
    return ComponentLatentAttnTiny(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
        config=config,
    )


def latent_attn_small(
    obs_space: Any = None,
    obs_width: int = None,
    obs_height: int = None,
    feature_normalizations: dict = None,
    config: dict = None,
    **kwargs,
) -> nn.Module:
    """Create a Latent Attention Small component policy."""
    config = config or {"clip_range": 0, "analyze_weights_interval": 300}
    return ComponentLatentAttnSmall(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
        config=config,
    )


def latent_attn_med(
    obs_space: Any = None,
    obs_width: int = None,
    obs_height: int = None,
    feature_normalizations: dict = None,
    config: dict = None,
    **kwargs,
) -> nn.Module:
    """Create a Latent Attention Medium component policy."""
    config = config or {"clip_range": 0, "analyze_weights_interval": 300}
    return ComponentLatentAttnMed(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
        config=config,
    )


# PyTorch implementations with different signature
def pytorch_example(env: Any = None, **kwargs) -> nn.Module:
    """Create a PyTorch Example policy."""
    config = kwargs.get("config", {"clip_range": 0, "analyze_weights_interval": 300})
    return Example(env=env, **config)


def pytorch_fast(env: Any = None, **kwargs) -> nn.Module:
    """Create a PyTorch Fast policy."""
    config = kwargs.get("config", {"clip_range": 0, "analyze_weights_interval": 300})
    return PyTorchFast(env=env, **config)


def pytorch_latent_attn_tiny(env: Any = None, **kwargs) -> nn.Module:
    """Create a PyTorch Latent Attention Tiny policy."""
    config = kwargs.get("config", {"clip_range": 0, "analyze_weights_interval": 300})
    return PyTorchLatentAttnTiny(env=env, **config)


def pytorch_latent_attn_small(env: Any = None, **kwargs) -> nn.Module:
    """Create a PyTorch Latent Attention Small policy."""
    config = kwargs.get("config", {"clip_range": 0, "analyze_weights_interval": 300})
    return PyTorchLatentAttnSmall(env=env, **config)


def pytorch_latent_attn_med(env: Any = None, **kwargs) -> nn.Module:
    """Create a PyTorch Latent Attention Medium policy."""
    config = kwargs.get("config", {"clip_range": 0, "analyze_weights_interval": 300})
    return PyTorchLatentAttnMed(env=env, **config)