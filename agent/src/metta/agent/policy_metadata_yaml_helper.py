"""
Policy metadata YAML serialization utilities.

This module handles saving and loading policy records with sidecar pattern.
Initialize once with init_yaml_serializers() before use.
"""

import shutil
import tempfile
from typing import Any, Dict

import torch
import yaml
from gymnasium.spaces import MultiDiscrete
from safetensors.torch import load_file, save_file

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord

# Module-level initialization flag
_yaml_serializers_initialized = False


def multidiscrete_representer(dumper, data):
    return dumper.represent_mapping("!MultiDiscrete", {"nvec": data.nvec.tolist(), "dtype": str(data.dtype)})


def multidiscrete_constructor(loader, node):
    values = loader.construct_mapping(node)
    return MultiDiscrete(values["nvec"], dtype=values["dtype"])


def init_yaml_serializers():
    """Initialize YAML serializers once. Call this before using other functions."""
    global _yaml_serializers_initialized

    if not _yaml_serializers_initialized:
        yaml.SafeDumper.add_representer(MultiDiscrete, multidiscrete_representer)
        yaml.SafeLoader.add_constructor("!MultiDiscrete", multidiscrete_constructor)
        _yaml_serializers_initialized = True


def save_policy(policy_record: PolicyRecord, checkpoint_name: str, base_path: str) -> str:
    """Save policy record with sidecar pattern"""
    init_yaml_serializers()  # Ensure serializers are initialized

    weights_ptx_path = f"{base_path}/{checkpoint_name}.safetensors"
    metadata_path = f"{base_path}/{checkpoint_name}.yaml"

    # Extract state dict (handle DDP wrapper)
    state_dict = _get_state_dict(policy_record.policy)
    # state_dict = {key.replace("policy.", ""): value for key, value in state_dict.items()}

    # Extract metadata from the policy record and agent
    metadata_dict = _extract_metadata_from_record(policy_record)

    # Atomic save
    safetensors_path = _atomic_save_weights(state_dict, weights_ptx_path)
    _atomic_save_metadata(metadata_dict, metadata_path)

    return safetensors_path


def get_metadata(checkpoint_name: str, base_path: str) -> PolicyMetadata:
    """Fast metadata loading without touching weights"""
    init_yaml_serializers()  # Ensure serializers are initialized

    metadata_path = f"{base_path}/{checkpoint_name}.yaml"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_dict = yaml.safe_load(f)
        if metadata_dict:
            return PolicyMetadata(**metadata_dict)
        else:
            return PolicyMetadata()


def restore_weights(agent: PolicyAgent, path: str) -> None:
    """Restore agent weights from checkpoint"""
    init_yaml_serializers()  # Ensure serializers are initialized

    weights = load_file(path)

    # Load weights (handle DDP)
    if isinstance(agent, DistributedMettaAgent):
        agent.module.load_state_dict(weights)
    else:
        agent.load_state_dict(weights)


# Private helper functions
def _get_state_dict(agent: PolicyAgent) -> Dict[str, torch.Tensor]:
    """Extract state dict, handling DDP wrapper"""
    if isinstance(agent, DistributedMettaAgent):
        return agent.module.state_dict()
    return agent.state_dict()


def _extract_agent_attributes(agent: PolicyAgent) -> Dict[str, Any]:
    # Get the actual agent (unwrap DDP if needed)
    actual_agent = agent.module if isinstance(agent, DistributedMettaAgent) else agent

    # Check if policy has agent_attributes
    attr_dict = {}
    if hasattr(actual_agent.policy, "agent_attributes"):
        value = actual_agent.policy.agent_attributes
        if isinstance(value, dict):
            attr_dict["agent_attributes"] = value

    # Return empty dict if no valid attributes found
    return attr_dict


def _extract_metadata_from_record(policy_record: PolicyRecord) -> Dict[str, Any]:
    """Extract metadata from a PolicyRecord and the underlying agent."""
    agent = policy_record.policy

    metadata_dict: Dict[str, Any] = {
        "metadata": dict(policy_record.metadata),  # Convert PolicyMetadata to dict
        "agent": _extract_agent_attributes(agent),
    }

    # Add top-level summary fields for backward compatibility
    metadata_dict["model_type"] = type(agent).__name__
    metadata_dict["run_name"] = policy_record.run_name
    metadata_dict["uri"] = policy_record.uri

    return metadata_dict


def _atomic_save_weights(sd: Dict, target_path: str) -> str:
    """Atomically save weights to prevent corruption"""
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt", dir=os.path.dirname(target_path)) as tmp:
        # (recommended) normalize tensors before saving - should I do this?
        sd = {k: v.detach().cpu().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
        save_file(sd, tmp.name)
        tmp.flush()
        shutil.move(tmp.name, target_path)

        print(f"Saved model weights to {target_path}")
        return target_path


def _atomic_save_metadata(metadata: Dict, target_path: str):
    """Atomically save metadata YAML"""
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", dir=os.path.dirname(target_path), encoding="utf-8"
    ) as tmp:
        yaml.safe_dump(metadata, tmp, default_flow_style=False, allow_unicode=True)
        tmp.flush()
        shutil.move(tmp.name, target_path)


def _is_serializable(obj) -> bool:
    """Check if object is YAML serializable"""
    try:
        yaml.safe_dump(obj)
        return True
    except:
        print(f"Failed to serialize {obj}")
        return False
