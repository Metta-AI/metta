from types import SimpleNamespace

import torch
from tensordict import TensorDict

from metta.agent.components.obs_shim import ObsTokenPadStrip


def _make_env_metadata(feature_map):
    obs_features = {
        name: SimpleNamespace(id=feat_id, normalization=norm) for name, (feat_id, norm) in feature_map.items()
    }
    return SimpleNamespace(obs_features=obs_features, feature_normalizations={feat_id: norm for feat_id, norm in (feature_map.values())})


def test_obs_token_pad_strip_stores_original_mapping():
    feature_map = {"hp": (2, 30.0), "mana": (4, 10.0)}
    env = _make_env_metadata(feature_map)
    pad_strip = ObsTokenPadStrip(env)
    pad_strip.train()

    log = pad_strip.initialize_to_environment(env, torch.device("cpu"))

    assert pad_strip.original_feature_mapping == {"hp": 2, "mana": 4}
    assert pad_strip.feature_id_remap == {}
    assert "Stored original feature mapping" in log


def test_obs_token_pad_strip_eval_maps_unknown_features_to_255():
    base_features = {"hp": (2, 30.0)}
    env = _make_env_metadata(base_features)
    pad_strip = ObsTokenPadStrip(env)
    pad_strip.initialize_to_environment(env, torch.device("cpu"))

    pad_strip.eval()
    updated_env = _make_env_metadata({"hp": (5, 30.0), "mana": (7, 10.0)})
    pad_strip.initialize_to_environment(updated_env, torch.device("cpu"))

    assert pad_strip.feature_id_remap[5] == 2  # known feature remapped to original id
    assert pad_strip.feature_id_remap[7] == 255  # unknown maps to 255 in eval mode

    # Forward pass remaps incoming tokens
    tokens = torch.tensor([[[0x00, 5, 10], [0xFF, 0xFF, 0xFF]]], dtype=torch.uint8)
    td = TensorDict({"env_obs": tokens}, batch_size=[1])
    output = pad_strip(td)
    remapped = output[pad_strip.out_key][0, 0, 1]
    assert remapped == 2


def test_obs_token_pad_strip_training_learns_new_features():
    base_features = {"hp": (2, 30.0)}
    env = _make_env_metadata(base_features)
    pad_strip = ObsTokenPadStrip(env)
    pad_strip.initialize_to_environment(env, torch.device("cpu"))

    pad_strip.train()
    updated_env = _make_env_metadata({"hp": (5, 30.0), "mana": (7, 10.0)})
    pad_strip.initialize_to_environment(updated_env, torch.device("cpu"))

    # Known feature remapped to original id; new feature learned with new id
    assert pad_strip.feature_id_remap[5] == 2
    assert "mana" in pad_strip.original_feature_mapping and pad_strip.original_feature_mapping["mana"] == 7
    assert 7 not in pad_strip.feature_id_remap
