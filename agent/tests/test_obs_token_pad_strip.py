from types import SimpleNamespace

import torch
from tensordict import TensorDict

from metta.agent.components.obs_shim import ObsShimTokens, ObsShimTokensConfig, ObsTokenPadStrip
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrEmbedFourierConfig


def _make_game_rules(feature_map):
    obs_features = {
        name: SimpleNamespace(id=feat_id, normalization=norm) for name, (feat_id, norm) in feature_map.items()
    }
    return SimpleNamespace(
        obs_features=obs_features, feature_normalizations={feat_id: norm for feat_id, norm in (feature_map.values())}
    )


def test_obs_token_pad_strip_stores_original_mapping():
    feature_map = {"hp": (2, 30.0), "mana": (4, 10.0)}
    env = _make_game_rules(feature_map)
    pad_strip = ObsTokenPadStrip(env)
    pad_strip.train()

    log = pad_strip.initialize_to_environment(env, torch.device("cpu"))

    assert pad_strip.original_feature_mapping == {"hp": 2, "mana": 4}
    assert torch.equal(pad_strip.feature_id_remap.cpu(), torch.arange(256, dtype=torch.uint8))
    assert "Stored original feature mapping" in log


def test_obs_token_pad_strip_eval_maps_unknown_features_to_255():
    base_features = {"hp": (2, 30.0)}
    env = _make_game_rules(base_features)
    pad_strip = ObsTokenPadStrip(env)
    pad_strip.initialize_to_environment(env, torch.device("cpu"))

    pad_strip.eval()
    updated_env = _make_game_rules({"hp": (5, 30.0), "mana": (7, 10.0)})
    pad_strip.initialize_to_environment(updated_env, torch.device("cpu"))

    remap_table = pad_strip.feature_id_remap
    assert remap_table[5].item() == 2  # known feature remapped to original id
    assert remap_table[7].item() == 255  # unknown maps to 255 in eval mode

    # Forward pass remaps incoming tokens
    tokens = torch.tensor([[[0x00, 5, 10], [0xFF, 0xFF, 0xFF]]], dtype=torch.uint8)
    td = TensorDict({"env_obs": tokens}, batch_size=[1])
    output = pad_strip(td)
    remapped = output[pad_strip.out_key][0, 0, 1]
    assert remapped == 2


def test_obs_token_pad_strip_training_learns_new_features():
    base_features = {"hp": (2, 30.0)}
    env = _make_game_rules(base_features)
    pad_strip = ObsTokenPadStrip(env)
    pad_strip.initialize_to_environment(env, torch.device("cpu"))

    pad_strip.train()
    updated_env = _make_game_rules({"hp": (5, 30.0), "mana": (7, 10.0)})
    pad_strip.initialize_to_environment(updated_env, torch.device("cpu"))

    remap_table = pad_strip.feature_id_remap
    assert remap_table[5].item() == 2
    assert "mana" in pad_strip.original_feature_mapping and pad_strip.original_feature_mapping["mana"] == 7


def test_obs_token_pad_strip_keeps_dense_sequences():
    feature_map = {"hp": (2, 30.0)}
    env = _make_game_rules(feature_map)
    pad_strip = ObsTokenPadStrip(env)

    tokens = torch.tensor(
        [
            [[0x00, 2, 10], [0x12, 2, 20], [0x21, 2, 30], [0x33, 2, 40]],
            [[0x00, 2, 10], [0x12, 2, 20], [0xFF, 0xFF, 0xFF], [0xFF, 0xFF, 0xFF]],
        ],
        dtype=torch.uint8,
    )
    td = TensorDict({"env_obs": tokens}, batch_size=[2])

    output = pad_strip(td)
    dense_row = output[pad_strip.out_key][0]

    assert dense_row[3, 0].item() == 0x33


def test_obs_token_pad_strip_preserves_padding_bytes():
    feature_map = {"hp": (2, 30.0)}
    env = _make_game_rules(feature_map)
    pad_strip = ObsTokenPadStrip(env)

    tokens = torch.tensor(
        [
            [[0x00, 2, 10], [0x12, 2, 20], [0xFF, 0xFF, 0xFF]],
            [[0x00, 2, 10], [0xFF, 0xFF, 0xFF], [0xFF, 0xFF, 0xFF]],
        ],
        dtype=torch.uint8,
    )
    td = TensorDict({"env_obs": tokens}, batch_size=[2])

    output = pad_strip(td)

    padded_row = output[pad_strip.out_key][1]
    mask = output["obs_mask"][1]

    assert torch.all(padded_row[mask][:, 0] == 0xFF)
    assert torch.all(padded_row[mask][:, 1] == 0xFF)


def test_obs_token_pad_strip_enforces_max_tokens():
    feature_map = {"hp": (2, 30.0)}
    env = _make_game_rules(feature_map)
    pad_strip = ObsTokenPadStrip(env, max_tokens=2)

    tokens = torch.tensor(
        [
            [[0x00, 2, 10], [0x12, 2, 20], [0x21, 2, 30]],
        ],
        dtype=torch.uint8,
    )
    td = TensorDict({"env_obs": tokens}, batch_size=[1])

    output = pad_strip(td)

    trimmed = output[pad_strip.out_key]
    mask = output["obs_mask"]

    assert trimmed.shape[1] == 2
    assert torch.equal(mask, torch.zeros_like(mask, dtype=torch.bool))


def test_obs_attr_embed_fourier_zeroes_masked_tokens():
    feature_map = {"hp": (2, 30.0)}
    env = _make_game_rules(feature_map)

    tokens = torch.tensor(
        [
            [[0x00, 2, 10], [0xFF, 0xFF, 0xFF]],
        ],
        dtype=torch.uint8,
    )
    td = TensorDict({"env_obs": tokens}, batch_size=[1])

    shim_config = ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens")
    shim = ObsShimTokens(env, config=shim_config)
    embed_config = ObsAttrEmbedFourierConfig(
        in_key="obs_shim_tokens", out_key="obs_attr_embed", attr_embed_dim=4, num_freqs=1
    )
    embed = ObsAttrEmbedFourier(embed_config)

    td = shim(td)
    td = embed(td)

    features = td["obs_attr_embed"]
    mask = td["obs_mask"]

    assert torch.all(features[mask] == 0)
