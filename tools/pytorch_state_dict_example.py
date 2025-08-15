import torch
import torch.nn as nn


# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.lstm = nn.LSTM(64, 32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 10)

        # Register a buffer (non-trainable but part of model state)
        self.register_buffer("running_mean", torch.zeros(16))

    def forward(self, x):
        return x


# Create model and examine its state_dict
model = SimpleNet()
state_dict = model.state_dict()

print("=== PyTorch state_dict() Contents ===\n")
print(f"Type: {type(state_dict)}")
print(f"Number of entries: {len(state_dict)}\n")

print("Keys and shapes:")
for key, tensor in state_dict.items():
    print(f"  '{key}': {tensor.shape} ({tensor.dtype})")

print("\n=== Sample Values ===")
# Show actual values from a few parameters
print(f"\nconv1.weight (first 3 values): {state_dict['conv1.weight'].flatten()[:3]}")
print(f"conv1.bias: {state_dict['conv1.bias']}")
print(f"running_mean buffer: {state_dict['running_mean']}")

# Show how to save and load just the state_dict
print("\n=== Save/Load Example ===")
torch.save(state_dict, "model_weights.pt")
loaded_state_dict = torch.load("model_weights.pt", map_location="cpu")

print(f"Loaded state_dict has {len(loaded_state_dict)} entries")
print("Keys match:", list(state_dict.keys()) == list(loaded_state_dict.keys()))

# Show how to load into a new model
new_model = SimpleNet()
new_model.load_state_dict(loaded_state_dict)
print("Successfully loaded weights into new model")

print("\n=== What's NOT in state_dict ===")
print("Model attributes that are NOT saved:")
print(f"  - Model class name: {type(model).__name__}")
print("  - Architecture info (layer sizes, etc.)")
print("  - Hyperparameters used during training")
print("  - Optimizer state")
print("  - Training history")

# Clean up
import os

if os.path.exists("model_weights.pt"):
    os.remove("model_weights.pt")


# Load and examine the object
data = torch.load(
    "/Users/localmini/github/metta/example.pt", map_location="cpu", weights_only=False
)  # map_location avoids GPU issues

# See the type
print(f"Type: {type(data)}")

# includes
#   file_path: str
#   metadata: PolicyMetadata
#   policy: MettaAgent
#   uri: str
#   _cached_policy: MettaAgent
#   _policy_store: PolicyStore (None)

# If it's a dict, see the keys
if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")

# Basic info about the object
print(f"Object: {data}")


# Test the new YAML methods for PolicyRecord
print("\n=== Testing PolicyRecord YAML Methods ===")

# Import the necessary classes
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent", "src"))

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord

# Create a test PolicyRecord
metadata = PolicyMetadata(
    agent_step=100, epoch=5, generation=1, train_time=60.0, score=0.85, additional_field="test_value"
)

policy_record = PolicyRecord(
    policy_store=None,  # None for standalone usage
    run_name="test_policy_v1",
    uri="file:///path/to/policy.pt",
    metadata=metadata,
)

# Test to_yaml method
print("\n--- PolicyRecord to YAML ---")
yaml_str = policy_record.to_yaml()
print(yaml_str)

# Test from_yaml method
print("\n--- PolicyRecord from YAML ---")
reconstructed_record = PolicyRecord.from_yaml(yaml_str)
print(f"Reconstructed run_name: {reconstructed_record.run_name}")
print(f"Reconstructed uri: {reconstructed_record.uri}")
print(f"Reconstructed metadata: {reconstructed_record.metadata}")

# Verify they are equivalent (excluding policy_store which is None)
print("\n--- Verification ---")
print(f"run_name matches: {policy_record.run_name == reconstructed_record.run_name}")
print(f"uri matches: {policy_record.uri == reconstructed_record.uri}")
print(f"metadata matches: {dict(policy_record.metadata) == dict(reconstructed_record.metadata)}")
