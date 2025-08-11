import torch
from tensordict import TensorDict

from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper


def test_obs_token_to_box_shaper_forward():
    # Test parameters
    obs_shape = (128, 3)  # 128 tokens, 3 bytes each
    obs_width = 11
    obs_height = 11
    feature_normalizations = {0: 1, 1: 2, 2: 3}  # 3 layers
    batch_size = 2

    # Create the shaper
    shaper = ObsTokenToBoxShaper(
        obs_shape=obs_shape,
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
    )

    shaper._name = "test_obs_token_to_box_shaper"

    # Create test input
    # Format: [batch, tokens, channels] where channels are [coord_byte, atr_index, atr_value]
    # coord_byte: first 4 bits are x, last 4 bits are y
    # Example: 0x12 means x=1, y=2
    input_tokens = torch.tensor(
        [
            # Batch 0
            [
                [0x12, 0, 1],  # x=1, y=2, layer 0, value 1.0
                [0x56, 2, 3],  # x=5, y=6, layer 2, value 3.0
                [0x34, 1, 2],  # x=3, y=4, layer 1, value 2.0
                [0xFF, 0xFF, 0xFF],  # Empty token (should be ignored)
            ],
            # Batch 1
            [
                [0x78, 0, 4],  # x=7, y=8, layer 0, value 4.0
                [0x9A, 1, 5],  # x=9, y=10, layer 1, value 5.0
                [0xFF, 0xFF, 0xFF],  # Empty token (should be ignored)
                [0xFF, 0xFF, 0xFF],  # Empty token (should be ignored)
            ],
        ],
        dtype=torch.uint8,
    )

    # Create input TensorDict
    td = TensorDict({"env_obs": input_tokens})

    # Run forward pass
    output_td = shaper._forward(td)

    # Get the output box observation
    box_obs = output_td[shaper._name]

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

    # Verify metadata is set correctly
    assert output_td["_batch_size_"] == batch_size
    assert output_td["_TT_"] == 1
    assert output_td["_BxTT_"] == batch_size
