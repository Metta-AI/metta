import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from gymnasium.spaces import MultiDiscrete

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent
from metta.agent.policy_record import PolicyRecord


def multidiscrete_representer(dumper, data):
    return dumper.represent_mapping("!MultiDiscrete", {"nvec": data.nvec.tolist(), "dtype": str(data.dtype)})


def multidiscrete_constructor(loader, node):
    values = loader.construct_mapping(node)
    return MultiDiscrete(values["nvec"], dtype=values["dtype"])


@dataclass
class CheckpointMetadata:
    """Structured metadata for checkpoints"""

    model_type: str
    training_step: int
    epoch: int
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    environment_info: Dict[str, str]
    timestamp: str
    git_hash: Optional[str] = None


class PolicyCheckpointManager:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        yaml.SafeDumper.add_representer(MultiDiscrete, multidiscrete_representer)
        yaml.SafeLoader.add_constructor("!MultiDiscrete", multidiscrete_constructor)

        space = MultiDiscrete([5, 2, 2])
        yaml_str = yaml.safe_dump({"action_space": space})
        loaded = yaml.safe_load(yaml_str)
        print(loaded)

    def save(self, policy_record: PolicyRecord, checkpoint_name: str):
        """Save policy record with sidecar pattern"""
        weights_ptx_path = self.base_path / f"{checkpoint_name}.ptx"
        metadata_path = self.base_path / f"{checkpoint_name}.yaml"

        # Extract state dict (handle DDP wrapper)
        state_dict = self._get_state_dict(policy_record.policy)

        # Extract metadata from the policy record and agent
        metadata_dict = self._extract_metadata_from_record(policy_record)

        # Atomic save
        self._atomic_save_weights(state_dict, weights_ptx_path)
        self._atomic_save_metadata(metadata_dict, metadata_path)

    def _get_state_dict(self, agent: PolicyAgent) -> Dict[str, torch.Tensor]:
        """Extract state dict, handling DDP wrapper"""
        if isinstance(agent, DistributedMettaAgent):
            return agent.module.state_dict()
        return agent.state_dict()

    def _extract_agent_metadata(self, agent: PolicyAgent) -> Dict[str, Any]:
        """Extract non-weight attributes from agent, reflecting the actual variable layout"""
        metadata = {}

        # Get the actual agent (unwrap DDP if needed)
        actual_agent = agent.module if isinstance(agent, DistributedMettaAgent) else agent

        # Extract all attributes that exist on the agent, organized by their actual structure
        # These are the actual attributes that exist on MettaAgent
        agent_attrs = [
            "agent_attributes",
            "_total_params",
            "active_features",
            "device",
            "feature_id_to_name",
            "feature_normalizations",
            "original_feature_mapping",
            "feature_id_remap",
            "action_max_params",
            "action_names",
            "active_actions",
            "cum_action_max_params",
            "action_index_tensor",
        ]

        # Extract attributes that exist on the agent
        for attr in agent_attrs:
            if hasattr(actual_agent, attr):
                value = getattr(actual_agent, attr)
                # Only serialize JSON-compatible types
                if self._is_serializable(value):
                    metadata[attr] = value

        return metadata

    def _extract_metadata_from_record(self, policy_record: PolicyRecord) -> Dict[str, Any]:
        """Extract metadata from a PolicyRecord and the underlying agent."""
        agent = policy_record.policy

        metadata_dict = {
            "metadata": dict(policy_record.metadata),  # Convert PolicyMetadata to dict
            "agent": self._extract_agent_metadata(agent),
        }

        # Add top-level summary fields for backward compatibility
        metadata_dict["model_type"] = type(agent).__name__
        metadata_dict["run_name"] = policy_record.run_name
        metadata_dict["uri"] = policy_record.uri

        return metadata_dict

    def _atomic_save_weights(self, state_dict: Dict, target_path: Path):
        """Atomically save weights to prevent corruption"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt", dir=target_path.parent) as tmp:
            torch.save(state_dict, tmp.name)
            tmp.flush()
            shutil.move(tmp.name, target_path)

    def _atomic_save_metadata(self, metadata: Dict, target_path: Path):
        """Atomically save metadata YAML"""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml", dir=target_path.parent, encoding="utf-8"
        ) as tmp:
            yaml.safe_dump(metadata, tmp, default_flow_style=False, allow_unicode=True)
            tmp.flush()
            shutil.move(tmp.name, target_path)

    def load_metadata_only(self, checkpoint_name: str) -> Dict[str, Any]:
        """Fast metadata loading without touching weights"""
        metadata_path = self.base_path / f"{checkpoint_name}.yaml"
        with open(metadata_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_weights_only(self, checkpoint_name: str) -> Dict[str, torch.Tensor]:
        """Load only model weights"""
        weights_path = self.base_path / f"{checkpoint_name}.pt"
        return torch.load(weights_path, map_location="cpu")

    def load_weights_ptx(self, checkpoint_name: str) -> Dict[str, torch.Tensor]:
        """Load model weights from .ptx file"""
        weights_ptx_path = self.base_path / f"{checkpoint_name}.ptx"
        return torch.load(weights_ptx_path, map_location="cpu")

    def load_full(self, checkpoint_name: str) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load both weights and metadata"""
        return self.load_weights_only(checkpoint_name), self.load_metadata_only(checkpoint_name)

    def restore_agent(self, agent: PolicyAgent, checkpoint_name: str):
        """Restore agent from checkpoint"""
        weights, metadata = self.load_full(checkpoint_name)

        # Load weights (handle DDP)
        if isinstance(agent, DistributedMettaAgent):
            agent.module.load_state_dict(weights)
            actual_agent = agent.module
        else:
            agent.load_state_dict(weights)
            actual_agent = agent

        # Restore metadata attributes
        for key, value in metadata.items():
            if hasattr(actual_agent, key) and key not in ["model_type", "timestamp"]:
                setattr(actual_agent, key, value)

    @staticmethod
    def _is_serializable(obj) -> bool:
        """Check if object is YAML serializable"""
        try:
            yaml.safe_dump(obj)
            return True
        except:
            print(f"Failed to serialize {obj}")
            return False
