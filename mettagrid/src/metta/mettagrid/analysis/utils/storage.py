"""
Storage management utilities for activation data.
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class StorageManager:
    """
    Manages storage and retrieval of activation data and analysis results.
    """

    def __init__(self, base_dir: str = "analysis_data"):
        """
        Initialize the storage manager.

        Args:
            base_dir: Base directory for storing data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.base_dir / "activations").mkdir(exist_ok=True)
        (self.base_dir / "models").mkdir(exist_ok=True)
        (self.base_dir / "results").mkdir(exist_ok=True)
        (self.base_dir / "metadata").mkdir(exist_ok=True)

    def save_activations(self, activations_data: Dict[str, Any], policy_uri: str, environment: str) -> Path:
        """
        Save activation data with metadata.

        Args:
            activations_data: Activation data to save
            policy_uri: Wandb URI of the policy
            environment: Environment name

        Returns:
            Path to saved file
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_name = policy_uri.replace("/", "_")
        filename = f"activations_{policy_name}_{environment}_{timestamp}.pkl"
        filepath = self.base_dir / "activations" / filename

        # Save activation data
        with open(filepath, "wb") as f:
            pickle.dump(activations_data, f)

        # Save metadata
        metadata = {
            "policy_uri": policy_uri,
            "environment": environment,
            "timestamp": timestamp,
            "filepath": str(filepath),
            "num_sequences": len(activations_data.get("sequences", [])),
            "num_activations": len(activations_data.get("activations", {})),
            "data_hash": self._compute_data_hash(activations_data),
        }

        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return filepath

    def load_activations(self, filepath: Path) -> Dict[str, Any]:
        """
        Load activation data from file.

        Args:
            filepath: Path to activation file

        Returns:
            Loaded activation data
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def save_model(self, model_data: Dict[str, Any], model_name: str, policy_uri: str) -> Path:
        """
        Save trained model with metadata.

        Args:
            model_data: Model data to save
            model_name: Name for the model
            policy_uri: Wandb URI of the policy

        Returns:
            Path to saved model
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_name = policy_uri.replace("/", "_")
        filename = f"{model_name}_{policy_name}_{timestamp}.pt"
        filepath = self.base_dir / "models" / filename

        # Save model data
        import torch

        torch.save(model_data, filepath)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "policy_uri": policy_uri,
            "timestamp": timestamp,
            "filepath": str(filepath),
            "model_type": type(model_data.get("model", None)).__name__,
            "config": model_data.get("config", {}).__dict__
            if hasattr(model_data.get("config", {}), "__dict__")
            else {},
            "training_metrics": {
                "final_loss": model_data.get("final_loss", None),
                "final_sparsity": model_data.get("final_sparsity", None),
                "num_active_neurons": model_data.get("num_active_neurons", None),
            },
        }

        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return filepath

    def load_model(self, filepath: Path) -> Dict[str, Any]:
        """
        Load model data from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model data
        """
        import torch

        return torch.load(filepath, map_location="cpu")

    def save_results(self, results: Dict[str, Any], results_name: str, policy_uri: str) -> Path:
        """
        Save analysis results with metadata.

        Args:
            results: Results to save
            results_name: Name for the results
            policy_uri: Wandb URI of the policy

        Returns:
            Path to saved results
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_name = policy_uri.replace("/", "_")
        filename = f"{results_name}_{policy_name}_{timestamp}.pkl"
        filepath = self.base_dir / "results" / filename

        # Save results
        with open(filepath, "wb") as f:
            pickle.dump(results, f)

        # Save metadata
        metadata = {
            "results_name": results_name,
            "policy_uri": policy_uri,
            "timestamp": timestamp,
            "filepath": str(filepath),
            "results_type": type(results).__name__,
            "results_keys": list(results.keys()) if isinstance(results, dict) else [],
        }

        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return filepath

    def load_results(self, filepath: Path) -> Dict[str, Any]:
        """
        Load results from file.

        Args:
            filepath: Path to results file

        Returns:
            Loaded results
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def list_activations(self) -> List[Dict[str, Any]]:
        """
        List all saved activation files with metadata.

        Returns:
            List of activation file metadata
        """
        activations_dir = self.base_dir / "activations"
        files = []

        for filepath in activations_dir.glob("*.pkl"):
            metadata_file = filepath.with_suffix(".json")
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    metadata["filepath"] = str(filepath)
                    files.append(metadata)

        return files

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all saved model files with metadata.

        Returns:
            List of model file metadata
        """
        models_dir = self.base_dir / "models"
        files = []

        for filepath in models_dir.glob("*.pt"):
            metadata_file = filepath.with_suffix(".json")
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    metadata["filepath"] = str(filepath)
                    files.append(metadata)

        return files

    def list_results(self) -> List[Dict[str, Any]]:
        """
        List all saved result files with metadata.

        Returns:
            List of result file metadata
        """
        results_dir = self.base_dir / "results"
        files = []

        for filepath in results_dir.glob("*.pkl"):
            metadata_file = filepath.with_suffix(".json")
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    metadata["filepath"] = str(filepath)
                    files.append(metadata)

        return files

    def _compute_data_hash(self, data: Dict[str, Any]) -> str:
        """Compute hash of data for integrity checking."""
        # Create a string representation of the data
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def cleanup_old_files(self, days_old: int = 30):
        """
        Clean up files older than specified days.

        Args:
            days_old: Age threshold in days
        """
        import time

        current_time = time.time()
        threshold = current_time - (days_old * 24 * 60 * 60)

        for subdir in ["activations", "models", "results"]:
            dir_path = self.base_dir / subdir
            for filepath in dir_path.glob("*"):
                if filepath.stat().st_mtime < threshold:
                    filepath.unlink()
                    # Also remove metadata file if it exists
                    metadata_file = filepath.with_suffix(".json")
                    if metadata_file.exists():
                        metadata_file.unlink()
