from types import SimpleNamespace

import torch
from tensordict import TensorDict

from metta.agent.components.obs_shim import ObsTokenToBoxShim


def test_obs_token_to_box_shaper_forward():
    # Test parameters
    obs_width = 11
    obs_height = 11
    feature_normalizations = {0: 1, 1: 2, 2: 3}  # 3 layers
    batch_size = 2

    env_meta = SimpleNamespace(
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
    )

    shaper = ObsTokenToBoxShim(env_meta)

    # Create test input - we need exactly 128 tokens as specified in obs_shape
    # Format: [batch, tokens, channels] where channels are [coord_byte, atr_index, atr_value]
    # coord_byte: first 4 bits are x, last 4 bits are y
    # Example: 0x12 means x=1, y=2

    # Create empty tokens for padding to 128
    empty_token = [0xFF, 0xFF, 0xFF]

    # Batch 0 tokens
    batch0_tokens = [
        [0x12, 0, 1],  # x=1, y=2, layer 0, value 1.0
        [0x56, 2, 3],  # x=5, y=6, layer 2, value 3.0
        [0x34, 1, 2],  # x=3, y=4, layer 1, value 2.0
    ] + [empty_token] * (128 - 3)  # Pad to 128 tokens

    # Batch 1 tokens
    batch1_tokens = [
        [0x78, 0, 4],  # x=7, y=8, layer 0, value 4.0
        [0x9A, 1, 5],  # x=9, y=10, layer 1, value 5.0
    ] + [empty_token] * (128 - 2)  # Pad to 128 tokens

    input_tokens = torch.tensor(
        [batch0_tokens, batch1_tokens],
        dtype=torch.uint8,
    )

    # Create input TensorDict with proper batch_size
    td = TensorDict({"env_obs": input_tokens}, batch_size=[batch_size])

    shaper(td)

    box_obs = td["box_obs"]

    # Verify output shape
    assert box_obs.shape == (batch_size, len(feature_normalizations), obs_width, obs_height)

    # Verify values are placed correctly
    # Batch 0
    assert box_obs[0, 0, 1, 2] == 1.0  # Layer 0, x=1, y=2
    assert box_obs[0, 1, 3, 4] == 2.0  # Layer 1, x=3, y=4
    assert box_obs[0, 2, 5, 6] == 3.0  # Layer 2, x=5, y=6

    # Batch 1
    assert box_obs[1, 0, 7, 8] == 4.0  # Layer 0, x=7, y=8
    assert box_obs[1, 1, 9, 10] == 5.0  # Layer 1, x=9, y=10

    # Verify invalid tokens are ignored (0xFF)
    assert torch.all(box_obs[:, :, :, :] >= 0)  # No negative values
    assert torch.all(box_obs[:, :, :, :] <= 5.0)  # No values larger than our max input

    # Metadata fields are implementation details and may not be set in all contexts
