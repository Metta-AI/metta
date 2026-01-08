"""Tests for obs_shim components with PolicyEnvInterface."""

import torch
from tensordict import TensorDict

from metta.agent.components.obs_shim import (
    ObsAttrValNorm,
    ObservationNormalizer,
    ObsShimBox,
    ObsShimBoxConfig,
    ObsShimTokens,
    ObsShimTokensConfig,
    ObsTokenPadStrip,
    ObsTokenToBoxShim,
)
from mettagrid.config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def _build_policy_env_info() -> PolicyEnvInterface:
    """Build a PolicyEnvInterface from default MettaGridConfig."""
    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def _build_token_observations(batch_size: int, num_tokens: int) -> TensorDict:
    """Create test token observations."""
    obs = torch.full((batch_size, num_tokens, 3), 0xFF, dtype=torch.uint8)
    # Token 0: coordinates (0,0), feature 0, value 10
    obs[:, 0] = torch.tensor([0x00, 0, 10], dtype=torch.uint8)
    # Token 1: coordinates (1,2) encoded as 0x12, feature 0, value 20
    obs[:, 1] = torch.tensor([0x12, 0, 20], dtype=torch.uint8)
    return TensorDict({"env_obs": obs}, batch_size=[batch_size])


def test_obs_token_pad_strip_initializes_with_policy_env_info():
    """Test that ObsTokenPadStrip can initialize with PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    pad_strip = ObsTokenPadStrip(policy_env_info)
    device = torch.device("cpu")

    log = pad_strip.initialize_to_environment(policy_env_info, device)

    assert "Stored original feature mapping" in log
    assert hasattr(pad_strip, "original_feature_mapping")
    assert hasattr(pad_strip, "feature_id_to_name")
    assert hasattr(pad_strip, "feature_normalizations")


def test_obs_attr_val_norm_initializes_with_policy_env_info():
    """Test that ObsAttrValNorm can initialize with PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    attr_val_norm = ObsAttrValNorm(policy_env_info)
    device = torch.device("cpu")

    attr_val_norm.initialize_to_environment(policy_env_info, device)

    assert hasattr(attr_val_norm, "_norm_factors")
    assert attr_val_norm._norm_factors.shape[0] > 0


def test_obs_shim_tokens_initializes_with_policy_env_info():
    """Test that ObsShimTokens can initialize with PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    config = ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens")
    obs_shim = ObsShimTokens(policy_env_info, config)
    device = torch.device("cpu")

    log = obs_shim.initialize_to_environment(policy_env_info, device)

    assert log is not None
    assert hasattr(obs_shim.token_pad_striper, "original_feature_mapping")


def test_obs_shim_tokens_forward_pass():
    """Test that ObsShimTokens can process observations."""
    policy_env_info = _build_policy_env_info()
    config = ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens")
    obs_shim = ObsShimTokens(policy_env_info, config)
    device = torch.device("cpu")

    obs_shim.initialize_to_environment(policy_env_info, device)
    obs_shim.eval()

    td = _build_token_observations(batch_size=2, num_tokens=4)
    output_td = obs_shim(td)

    assert "obs_shim_tokens" in output_td
    assert "obs_mask" in output_td
    assert output_td["obs_shim_tokens"].dtype == torch.float32


def test_obs_token_to_box_shim_with_policy_env_info():
    """Test that ObsTokenToBoxShim works with PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    token_to_box = ObsTokenToBoxShim(policy_env_info, in_key="env_obs", out_key="box_obs")

    td = _build_token_observations(batch_size=2, num_tokens=4)
    output_td = token_to_box(td)

    assert "box_obs" in output_td
    box_obs = output_td["box_obs"]
    assert box_obs.ndim == 4  # [B, L, W, H]
    assert box_obs.shape[2] == policy_env_info.obs_width
    assert box_obs.shape[3] == policy_env_info.obs_height


def test_observation_normalizer_initializes_with_policy_env_info():
    """Test that ObservationNormalizer can initialize with PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    normalizer = ObservationNormalizer(policy_env_info, in_key="box_obs", out_key="normalized_obs")
    device = torch.device("cpu")

    normalizer.initialize_to_environment(policy_env_info, device)

    assert hasattr(normalizer, "obs_norm")
    assert normalizer.obs_norm.shape[0] == 1  # Batch dimension


def test_obs_shim_box_initializes_with_policy_env_info():
    """Test that ObsShimBox can initialize with PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    config = ObsShimBoxConfig(in_key="env_obs", out_key="box_obs_normalized")
    obs_shim_box = ObsShimBox(policy_env_info, config)
    device = torch.device("cpu")

    obs_shim_box.initialize_to_environment(policy_env_info, device)

    assert hasattr(obs_shim_box.observation_normalizer, "obs_norm")


def test_obs_shim_box_forward_pass():
    """Test that ObsShimBox can process observations end-to-end."""
    policy_env_info = _build_policy_env_info()
    config = ObsShimBoxConfig(in_key="env_obs", out_key="box_obs_normalized")
    obs_shim_box = ObsShimBox(policy_env_info, config)
    device = torch.device("cpu")

    obs_shim_box.initialize_to_environment(policy_env_info, device)
    obs_shim_box.eval()

    td = _build_token_observations(batch_size=2, num_tokens=4)
    output_td = obs_shim_box(td)

    assert "box_obs_normalized" in output_td
    box_obs = output_td["box_obs_normalized"]
    assert box_obs.ndim == 4  # [B, L, W, H]
    assert box_obs.dtype == torch.float32


def test_obs_shim_preserves_feature_count():
    """Test that obs_shim correctly handles the number of features from PolicyEnvInterface."""
    policy_env_info = _build_policy_env_info()
    num_features = len(policy_env_info.obs_features)

    config = ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens")
    obs_shim = ObsShimTokens(policy_env_info, config)

    device = torch.device("cpu")
    obs_shim.initialize_to_environment(policy_env_info, device)

    # Check that all features were registered
    assert len(obs_shim.token_pad_striper.feature_id_to_name) == num_features
    assert len(obs_shim.token_pad_striper.feature_normalizations) == num_features


def test_obs_shim_device_placement():
    """Test that obs_shim components respect device placement."""
    policy_env_info = _build_policy_env_info()
    config = ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens")
    obs_shim = ObsShimTokens(policy_env_info, config)

    device = torch.device("cpu")
    obs_shim.initialize_to_environment(policy_env_info, device)

    # Check that buffers are on the correct device
    assert obs_shim.token_pad_striper.feature_id_remap.device.type == "cpu"
    assert obs_shim.attr_val_normer._norm_factors.device.type == "cpu"
