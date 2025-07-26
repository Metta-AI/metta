"""Test the unified registry system for features and actions."""

import torch

from metta.agent.registry import ActionRegistry, FeatureRegistry, Registry


def test_registry_basic():
    """Test basic registry functionality."""
    reg = Registry("test")

    # First initialization
    names = ["feature1", "feature2", "feature3"]
    indices = reg.initialize(names, torch.device("cpu"))

    assert len(indices) == 3
    assert indices.tolist() == [0, 1, 2]
    assert reg.next_id == 3

    # Second initialization with overlap
    names2 = ["feature2", "feature3", "feature4"]
    indices2 = reg.initialize(names2, torch.device("cpu"))

    assert indices2.tolist() == [1, 2, 3]  # Reuses existing IDs
    assert reg.next_id == 4


def test_registry_remapping():
    """Test remapping functionality."""
    reg = Registry("test", unknown_id=255)

    # Initialize with some names
    reg.initialize(["a", "b", "c"], torch.device("cpu"))

    # Test remapping in training mode
    remap = reg.get_remapping(["b", "c", "d"], is_training=True)
    assert remap == {0: 1, 1: 2}  # b and c get remapped to their stored positions

    # Test remapping in eval mode
    remap = reg.get_remapping(["b", "c", "d"], is_training=False)
    assert remap == {0: 1, 1: 2, 2: 255}  # Unknown 'd' maps to 255


def test_feature_registry():
    """Test feature registry with normalizations."""
    reg = FeatureRegistry()

    features = {
        "health": {"id": 0, "normalization": 100.0},
        "ammo": {"id": 1, "normalization": 50.0},
        "position_x": {"id": 2},
    }

    remap, norm = reg.initialize_with_features(features, torch.device("cpu"))

    # Check remapping is identity (since env IDs match stored IDs)
    assert remap[0].item() == 0
    assert remap[1].item() == 1
    assert remap[2].item() == 2

    # Check normalizations
    assert norm[0].item() == 100.0
    assert norm[1].item() == 50.0
    assert norm[2].item() == 1.0  # Default

    # Test serialization
    data = reg.to_dict()
    reg2 = FeatureRegistry.from_dict(data)
    assert reg2.name_to_id == reg.name_to_id
    assert reg2.normalizations == reg.normalizations


def test_action_registry():
    """Test action registry with embedding expansion."""
    reg = ActionRegistry()

    # Initialize with actions
    indices = reg.initialize_with_actions(
        ["move", "attack", "use"],
        [3, 1, 0],  # max params
        torch.device("cpu"),
    )

    # Should create move_0, move_1, move_2, move_3, attack_0, attack_1, use_0
    assert len(indices) == 7

    # Test embedding expansion
    small_embed = torch.nn.Embedding(5, 16)
    expanded = reg.ensure_embedding_size(small_embed)

    assert expanded.num_embeddings >= 7
    assert expanded.embedding_dim == 16
    assert torch.allclose(expanded.weight[:5], small_embed.weight)


def test_registry_persistence():
    """Test saving and loading registries."""
    # Create and populate registries
    feat_reg = FeatureRegistry()
    feat_reg.initialize_with_features(
        {"health": {"id": 0, "normalization": 100.0}, "ammo": {"id": 1}}, torch.device("cpu")
    )

    act_reg = ActionRegistry()
    act_reg.initialize_with_actions(["move", "attack"], [3, 1], torch.device("cpu"))

    # Save to dict
    feat_data = feat_reg.to_dict()
    act_data = act_reg.to_dict()

    # Load from dict
    feat_reg2 = FeatureRegistry.from_dict(feat_data)
    act_reg2 = ActionRegistry.from_dict(act_data)

    # Verify
    assert feat_reg2.name_to_id == feat_reg.name_to_id
    assert feat_reg2.normalizations == feat_reg.normalizations
    assert act_reg2.name_to_id == act_reg.name_to_id
