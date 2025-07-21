#!/usr/bin/env python3
"""
Example of loading a Metta checkpoint without the full Metta framework.

This demonstrates how the new checkpoint format enables loading models
in any PyTorch environment without Metta-specific dependencies.
"""

import json
import argparse
import torch
import torch.nn as nn
from typing import Dict, Any


class SimpleMettaAgent(nn.Module):
    """Minimal MettaAgent reconstruction for loading checkpoints."""
    
    def __init__(self, model_info: Dict[str, Any]):
        super().__init__()
        
        # Extract model architecture from metadata
        agent_attrs = model_info.get('agent_attributes', {})
        hidden_size = model_info.get('hidden_size', 256)
        
        # Recreate basic MettaAgent structure
        # This is a simplified version - real implementation would need
        # to match the exact architecture based on model_info
        self.components = nn.ModuleDict()
        
        # These would need to be reconstructed based on the actual architecture
        # This is just a placeholder to show the concept
        self.hidden_size = hidden_size
        
        # Placeholder components that would need proper reconstruction
        self.components['_core_'] = nn.Linear(100, hidden_size)
        self.components['_value_'] = nn.Linear(hidden_size, 1)
        self.components['_action_'] = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        # Simplified forward pass
        h = self.components['_core_'](x)
        value = self.components['_value_'](h)
        action = self.components['_action_'](h)
        return action, value


def load_metta_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a Metta checkpoint in the new format.
    
    Args:
        checkpoint_path: Path to the .pt file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load metadata
    base_path = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
    metadata_path = base_path + '.json'
    
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Display metadata
    print("\nCheckpoint Metadata:")
    print(f"  Epoch: {metadata.get('epoch', 'N/A')}")
    print(f"  Agent Steps: {metadata.get('agent_step', 'N/A')}")
    print(f"  Generation: {metadata.get('generation', 'N/A')}")
    print(f"  Train Time: {metadata.get('train_time', 'N/A')}s")
    if 'avg_reward' in metadata:
        print(f"  Average Reward: {metadata['avg_reward']}")
    
    # Load model state dict
    print(f"\nLoading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Display model info
    print("\nModel State Dict Keys:")
    for key in sorted(state_dict.keys())[:10]:  # Show first 10 keys
        param_shape = state_dict[key].shape
        print(f"  {key}: {param_shape}")
    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Note: To actually use the model, you would need to:
    # 1. Reconstruct the exact model architecture based on model_info
    # 2. Load the state dict into the reconstructed model
    # 3. Set the model to evaluation mode
    
    # For demonstration, we'll just return the loaded data
    return state_dict, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Load Metta checkpoints without the full framework'
    )
    parser.add_argument('checkpoint', help='Path to the .pt checkpoint file')
    parser.add_argument('--device', default='cpu', help='Device to load on')
    
    args = parser.parse_args()
    
    try:
        state_dict, metadata = load_metta_checkpoint(args.checkpoint, args.device)
        
        print("\n✅ Successfully loaded checkpoint!")
        print("\nThis checkpoint can now be used in any PyTorch environment")
        print("without requiring the full Metta framework.")
        
        # Example of how you might use it:
        print("\nExample usage:")
        print("  # 1. Reconstruct model architecture based on metadata")
        print("  model = YourModelClass(metadata['model_info'])")
        print("  # 2. Load the weights")
        print("  model.load_state_dict(state_dict)")
        print("  # 3. Use for inference")
        print("  model.eval()")
        print("  output = model(input_tensor)")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the checkpoint has been migrated to the new format.")
        print("Use 'python tools/migrate_checkpoints.py' to migrate old checkpoints.")
        return 1
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())