#!/usr/bin/env python3
"""Test that PolicyStore can save/load policies"""

import json
import os
import tempfile

import torch
from omegaconf import OmegaConf

from metta.agent.mocks import MockPolicy
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore


def test_policy_save_load_without_pydantic():
    """Test that we can save and load a policy without pydantic errors"""

    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "vectorization": "serial",
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
            # Add minimal agent config needed for make_policy
            "agent": {
                "type": "metta",
                "hidden_size": 256,
                "num_layers": 2,
            },
        }
    )

    # Create PolicyStore (without wandb_run for simplicity)
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test policy
    policy = MockPolicy()

    # Create metadata using PolicyMetadata class
    metadata = PolicyMetadata(
        action_names=["move", "turn"],
        agent_step=100,
        epoch=5,
        generation=1,
        train_time=60.0,
    )

    # Save the policy
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        print(temp_path)

        # Create a policy record
        pr = policy_store.create_empty_policy_record(name=temp_path)
        pr.metadata = metadata
        pr.policy = policy

        # Save
        policy_store.save(pr)

        # Test that the new format is correct
        # Should have created both .pt and .json files
        base_path = temp_path[:-3] if temp_path.endswith('.pt') else temp_path
        metadata_path = base_path + '.json'
        
        assert os.path.exists(temp_path), f"Model file {temp_path} not created"
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} not created"
        
        # Verify .pt file contains only state_dict
        state_dict = torch.load(temp_path, map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict), "Model file should contain state_dict"
        assert "fc.weight" in state_dict, "State dict should contain model weights"
        
        # Verify metadata JSON contains all necessary info
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        assert metadata_dict["action_names"] == ["move", "turn"]
        assert metadata_dict["agent_step"] == 100
        assert metadata_dict["epoch"] == 5
        assert metadata_dict["generation"] == 1
        assert metadata_dict["train_time"] == 60.0
        assert "model_info" in metadata_dict
        assert metadata_dict["model_info"]["type"] == "MockPolicy"

        print("✅ New checkpoint format verified - separate .pt and .json files!")

        # Load the policy back and verify it works
        loaded_pr = policy_store.load_from_uri(f"file://{temp_path}")
        loaded_policy = loaded_pr.policy

        # Verify the loaded policy works with a forward pass
        test_input = torch.randn(1, 10)
        output = loaded_policy(test_input)

        # Assertions
        assert type(loaded_policy).__name__ == "MockPolicy"
        assert loaded_pr.metadata == metadata
        assert output.shape == torch.Size([1, 10])

        # Verify the loaded policy has the expected structure
        assert hasattr(loaded_policy, "fc")
        assert hasattr(loaded_policy, "components")
        assert hasattr(loaded_policy.components, "_core_")
        assert hasattr(loaded_policy.components, "_value_")
        assert hasattr(loaded_policy.components, "_action_")

        print("✅ Policy loading and forward pass verified!")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        base_path = temp_path[:-3] if temp_path.endswith('.pt') else temp_path
        metadata_path = base_path + '.json'
        if os.path.exists(metadata_path):
            os.remove(metadata_path)


def test_policy_save_load_with_dict_metadata():
    """Test that we can save and load a policy with plain dict metadata"""

    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "vectorization": "serial",
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
        }
    )

    # Create PolicyStore (without wandb_run for simplicity)
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test policy
    policy = MockPolicy()

    # Create metadata as plain dict (testing backwards compatibility)
    metadata = {
        "action_names": ["move", "turn"],
        "agent_step": 100,
        "epoch": 5,
        "generation": 1,
        "train_time": 60.0,
    }

    # Save the policy
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        # Create a policy record
        pr = policy_store.create_empty_policy_record(name=temp_path)
        pr.metadata = metadata
        pr.policy = policy

        # Save
        policy_store.save(pr)

        # Load it back
        loaded_pr = policy_store.load_from_uri(f"file://{temp_path}")
        loaded_policy = loaded_pr.policy

        # Verify the loaded policy works
        test_input = torch.randn(1, 10)
        output = loaded_policy(test_input)

        assert output.shape == torch.Size([1, 10])
        assert loaded_pr.metadata["action_names"] == ["move", "turn"]

        print("✅ Policy save/load with dict metadata verified!")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        base_path = temp_path[:-3] if temp_path.endswith('.pt') else temp_path
        metadata_path = base_path + '.json'
        if os.path.exists(metadata_path):
            os.remove(metadata_path)


def test_load_old_format_checkpoint():
    """Test backward compatibility with old format checkpoints"""
    
    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "vectorization": "serial",
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
            "agent": {
                "type": "metta",
                "hidden_size": 256,
                "num_layers": 2,
            },
        }
    )

    # Create PolicyStore
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test policy
    policy = MockPolicy()
    
    # Create metadata
    metadata = PolicyMetadata(
        action_names=["move", "turn"],
        agent_step=100,
        epoch=5,
        generation=1,
        train_time=60.0,
    )

    # Create a PolicyRecord in old format
    pr = policy_store.create_empty_policy_record(name="old_checkpoint")
    pr.metadata = metadata
    pr._cached_policy = policy
    
    # Save in old format (simulate legacy checkpoint)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name
    
    try:
        # Temporarily clear _policy_store to avoid pickling issues (mimicking old save behavior)
        pr._policy_store = None
        torch.save(pr, temp_path)
        pr._policy_store = policy_store
        
        print(f"Created old format checkpoint at {temp_path}")
        
        # Load the old format checkpoint
        loaded_pr = policy_store._load_from_file(temp_path, metadata_only=False)
        
        # Verify it loaded correctly
        assert loaded_pr.metadata["action_names"] == ["move", "turn"]
        assert loaded_pr.metadata["epoch"] == 5
        
        # Get the policy and verify it works
        loaded_policy = loaded_pr.policy
        test_input = torch.randn(1, 10)
        output = loaded_policy(test_input)
        
        assert output.shape == torch.Size([1, 10])
        assert type(loaded_policy).__name__ == "MockPolicy"
        
        print("✅ Old format checkpoint loading verified!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_policy_record_backwards_compatibility():
    """Test that PolicyRecord can handle old metadata attribute names"""
    # Create minimal config for PolicyStore
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
        }
    )

    policy_store = PolicyStore(cfg, wandb_run=None)

    # Test different old attribute names
    old_attribute_names = ["checkpoint"]

    for old_name in old_attribute_names:
        # Create a PolicyRecord without using the normal constructor
        # to simulate loading an old checkpoint
        pr = PolicyRecord.__new__(PolicyRecord)
        pr._policy_store = policy_store
        pr.run_name = "test_policy"
        pr.uri = "file:///tmp/test.pt"
        pr._cached_policy = None

        # Set metadata using the old attribute name
        old_metadata = {
            "action_names": ["move", "turn"],
            "agent_step": 100,
            "epoch": 5,
            "generation": 1,
            "train_time": 60.0,
        }
        setattr(pr, old_name, old_metadata)

        # Access metadata property - this should trigger backwards compatibility
        metadata = pr.metadata

        # Verify it was converted properly
        assert isinstance(metadata, PolicyMetadata)
        assert metadata["action_names"] == ["move", "turn"]
        assert metadata["agent_step"] == 100
        assert metadata["epoch"] == 5
        assert metadata["generation"] == 1
        assert metadata["train_time"] == 60.0

        # Verify the old attribute was not removed (for safety)
        assert hasattr(pr, old_name)

        # Verify the new _metadata attribute exists
        assert hasattr(pr, "_metadata")

        print(f"✅ Backwards compatibility test passed for old attribute name: {old_name}")

    # Test with no metadata attribute at all
    pr_no_metadata = PolicyRecord.__new__(PolicyRecord)
    pr_no_metadata._policy_store = policy_store
    pr_no_metadata.run_name = "test_policy"
    pr_no_metadata.uri = "file:///tmp/test.pt"
    pr_no_metadata._cached_policy = None

    # This should raise AttributeError
    try:
        _ = pr_no_metadata.metadata
        raise AssertionError("Expected AttributeError when no metadata found")
    except AttributeError as e:
        assert "No metadata found" in str(e)
        print("✅ Correctly raised AttributeError when no metadata found")


def test_new_checkpoint_format():
    """Test the new checkpoint format with separate .pt and .json files"""
    
    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "vectorization": "serial",
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
            "agent": {
                "type": "metta",
                "hidden_size": 256,
                "num_layers": 2,
            },
        }
    )

    # Create PolicyStore
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test policy
    policy = MockPolicy()

    # Create metadata using PolicyMetadata class
    metadata = PolicyMetadata(
        action_names=["move", "turn"],
        agent_step=100,
        epoch=5,
        generation=1,
        train_time=60.0,
        # Add some extra metadata
        evals={"score": 0.95},
        avg_reward=42.0,
    )

    # Save the policy
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        # Create a policy record
        pr = policy_store.create_empty_policy_record(name=temp_path)
        pr.metadata = metadata
        pr.policy = policy

        # Save in new format
        policy_store.save(pr)

        # Verify files were created
        base_path = temp_path[:-3]
        assert os.path.exists(temp_path), "Model .pt file should exist"
        assert os.path.exists(base_path + ".json"), "Metadata .json file should exist"

        # Verify .pt file contains only state_dict
        state_dict = torch.load(temp_path, map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict), "PT file should contain state_dict"
        assert "fc.weight" in state_dict, "State dict should contain model weights"

        # Verify .json file contains metadata
        import json
        with open(base_path + ".json", 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["epoch"] == 5
        assert saved_metadata["agent_step"] == 100
        assert saved_metadata["evals"]["score"] == 0.95
        assert saved_metadata["avg_reward"] == 42.0
        assert "model_info" in saved_metadata
        assert saved_metadata["model_info"]["type"] == "MockPolicy"

        print("✅ New checkpoint format save verified!")

        # Test loading
        loaded_pr = policy_store.load_from_uri(f"file://{temp_path}")
        loaded_policy = loaded_pr.policy

        # Verify the loaded policy works
        test_input = torch.randn(1, 10)
        output = loaded_policy(test_input)

        assert output.shape == torch.Size([1, 10])
        assert loaded_pr.metadata["epoch"] == 5
        assert loaded_pr.metadata["agent_step"] == 100
        assert loaded_pr.metadata["evals"]["score"] == 0.95

        print("✅ New checkpoint format load verified!")

    finally:
        # Cleanup
        for path in [temp_path, base_path + ".json"]:
            if os.path.exists(path):
                os.remove(path)


def test_checkpoint_migration():
    """Test migration from old format to new format"""
    
    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
            },
            "data_dir": tempfile.mkdtemp(),
            "agent": {
                "type": "metta",
                "hidden_size": 256,
                "num_layers": 2,
            },
        }
    )

    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test policy
    policy = MockPolicy()
    metadata = PolicyMetadata(
        agent_step=200,
        epoch=10,
        generation=2,
        train_time=120.0,
    )

    # Save in old format
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        old_path = f.name

    try:
        # Create and save in old format (mimicking the old save method)
        pr = policy_store.create_empty_policy_record(name=old_path)
        pr.metadata = metadata
        pr.policy = policy

        # Save using old method (directly pickle PolicyRecord)
        pr._policy_store = None
        torch.save(pr, old_path)
        pr._policy_store = policy_store

        # Migrate to new format
        new_path = policy_store.migrate_checkpoint(old_path)

        # Verify new files exist
        assert os.path.exists(new_path)
        base_path = new_path[:-3]
        assert os.path.exists(base_path + ".json")

        # Load from new format
        loaded_pr = policy_store.load_from_uri(f"file://{new_path}")
        assert loaded_pr.metadata["epoch"] == 10
        assert loaded_pr.metadata["agent_step"] == 200

        # Verify policy works
        test_input = torch.randn(1, 10)
        output = loaded_pr.policy(test_input)
        assert output.shape == torch.Size([1, 10])

        print("✅ Checkpoint migration verified!")

    finally:
        # Cleanup
        for path in [old_path, new_path, new_path[:-3] + ".json"]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    test_policy_save_load_without_pydantic()
    test_policy_save_load_with_dict_metadata()
    test_load_old_format_checkpoint()
    test_policy_record_backwards_compatibility()
    test_new_checkpoint_format()
    test_checkpoint_migration()
    print("✅ All tests passed!")
