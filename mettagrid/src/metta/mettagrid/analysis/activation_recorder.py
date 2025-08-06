"""
Activation recording utilities for mettagrid analysis.
"""

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ActivationMetadata:
    """Metadata for recorded activations."""

    policy_uri: str
    sequence_id: str
    sequence_type: str  # "extracted" or "procedural"
    sequence_length: int
    environment: str
    timestamp: str
    lstm_hidden_size: int
    lstm_cell_size: int


class ActivationRecorder:
    """
    Records LSTM activations at sequence completion points.

    This class captures the internal states of LSTM layers
    after executing specific sequences of actions/observations.
    """

    def __init__(self, storage_dir: str = "activations"):
        """
        Initialize the activation recorder.

        Args:
            storage_dir: Directory to store recorded activations
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Hook storage
        self.activations = {}
        self.hooks = []

    def record_activations(
        self, policy: nn.Module, sequences: List[Dict[str, Any]], policy_uri: str, environment: str
    ) -> Dict[str, Any]:
        """
        Record LSTM activations for a set of sequences.

        Args:
            policy: The policy model to record from
            sequences: List of sequences to execute
            policy_uri: Wandb URI of the policy
            environment: Environment name/configuration

        Returns:
            Dictionary containing recorded activations and metadata
        """
        # Register hooks for LSTM layers
        self._register_hooks(policy)

        activations_data = {
            "policy_uri": policy_uri,
            "environment": environment,
            "recorded_at": datetime.now().isoformat(),
            "sequences": [],
            "activations": {},
            "metadata": [],
        }

        try:
            for i, sequence in enumerate(sequences):
                sequence_id = f"seq_{i:04d}"

                # Execute sequence and record activations
                sequence_activations = self._execute_sequence(policy, sequence, sequence_id, policy_uri, environment)

                activations_data["sequences"].append(sequence)
                activations_data["activations"][sequence_id] = sequence_activations
                activations_data["metadata"].append(sequence_activations["metadata"])

        finally:
            # Clean up hooks
            self._remove_hooks()

        return activations_data

    def _register_hooks(self, policy: nn.Module):
        """Register hooks on LSTM layers."""
        self.activations = {}
        self.hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # LSTM returns (output, (hidden, cell))
                    hidden, cell = output[1]
                    self.activations[name] = {"hidden": hidden.detach().cpu(), "cell": cell.detach().cpu()}
                else:
                    # Single output
                    self.activations[name] = {"output": output.detach().cpu()}

            return hook

        # Find and hook LSTM layers
        for name, module in policy.named_modules():
            if isinstance(module, (nn.LSTM, nn.LSTMCell)):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _execute_sequence(
        self, policy: nn.Module, sequence: Dict[str, Any], sequence_id: str, policy_uri: str, environment: str
    ) -> Dict[str, Any]:
        """
        Execute a single sequence and record activations at completion.

        Args:
            policy: Policy to execute
            sequence: Sequence data (observations, actions, etc.)
            sequence_id: Unique identifier for this sequence
            policy_uri: Policy wandb URI
            environment: Environment name

        Returns:
            Recorded activation data
        """
        # Reset policy state
        policy.eval()

        # Execute sequence (placeholder - needs environment integration)
        # This is a simplified version - actual implementation needs
        # proper environment integration
        observations = sequence.get("observations", [])
        actions = sequence.get("actions", [])

        # Process sequence
        with torch.no_grad():
            for obs, _action in zip(observations, actions, strict=False):
                # Forward pass through policy
                # This is a placeholder - actual implementation depends on
                # how the policy processes observations
                if isinstance(obs, (list, tuple)):
                    _obs_tensor = torch.tensor(obs, dtype=torch.float32)
                else:
                    _obs_tensor = obs

                # Forward pass (placeholder)
                # output = policy(obs_tensor)

        # Record final activations
        final_activations = {}
        for name, activation_data in self.activations.items():
            final_activations[name] = {
                "hidden": activation_data.get("hidden", None),
                "cell": activation_data.get("cell", None),
                "output": activation_data.get("output", None),
            }

        # Create metadata
        metadata = ActivationMetadata(
            policy_uri=policy_uri,
            sequence_id=sequence_id,
            sequence_type=sequence.get("type", "unknown"),
            sequence_length=len(observations),
            environment=environment,
            timestamp=datetime.now().isoformat(),
            lstm_hidden_size=final_activations.get("lstm", {}).get("hidden", torch.zeros(1)).shape[-1]
            if final_activations
            else 0,
            lstm_cell_size=final_activations.get("lstm", {}).get("cell", torch.zeros(1)).shape[-1]
            if final_activations
            else 0,
        )

        return {"activations": final_activations, "metadata": asdict(metadata)}

    def save_activations(self, activations_data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save recorded activations to disk.

        Args:
            activations_data: Data from record_activations
            filename: Optional filename, auto-generated if None

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            policy_name = activations_data["policy_uri"].replace("/", "_")
            filename = f"activations_{policy_name}_{timestamp}.pkl"

        filepath = self.storage_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump(activations_data, f)

        # Also save metadata as JSON for easy inspection
        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "policy_uri": activations_data["policy_uri"],
                    "environment": activations_data["environment"],
                    "recorded_at": activations_data["recorded_at"],
                    "num_sequences": len(activations_data["sequences"]),
                    "metadata": activations_data["metadata"],
                },
                f,
                indent=2,
            )

        return filepath

    def load_activations(self, filepath: Path) -> Dict[str, Any]:
        """
        Load recorded activations from disk.

        Args:
            filepath: Path to saved activations file

        Returns:
            Loaded activations data
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
