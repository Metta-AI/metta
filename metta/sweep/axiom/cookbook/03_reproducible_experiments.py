#!/usr/bin/env python3
"""
Recipe: Creating Perfectly Reproducible Experiments
====================================================

Problem: You need to ensure your experiments are perfectly reproducible,
allowing you to save, share, and re-run exact experiment configurations.

Solution: Use Axiom's spec system with deterministic controls, manifest tracking,
and experiment serialization to achieve perfect reproducibility.

This recipe demonstrates:
- Saving and loading experiment specs
- Ensuring deterministic execution
- Tracking all configuration and results
- Reproducing experiments from saved specs
- Comparing reproduced runs for validation
"""

import os
import sys
import json
import pickle
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import datetime

# Add parent directory to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from metta.sweep.axiom.core import Pipeline
from metta.sweep.axiom.experiment import AxiomExperiment, RunHandle
from metta.sweep.axiom.experiment_spec import ExperimentSpec, AxiomSpec, AxiomControls
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment
)


# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================

def compute_spec_hash(spec: AxiomSpec) -> str:
    """
    Compute a deterministic hash of an experiment spec.
    
    Args:
        spec: Experiment specification
        
    Returns:
        Hex string hash of the spec
    """
    # Convert spec to JSON for consistent hashing
    spec_dict = spec.dict()
    spec_json = json.dumps(spec_dict, sort_keys=True)
    return hashlib.sha256(spec_json.encode()).hexdigest()[:16]


def save_experiment_spec(spec: AxiomSpec, path: Path, 
                        include_metadata: bool = True) -> Dict[str, Any]:
    """
    Save an experiment spec to disk with metadata.
    
    Args:
        spec: Experiment specification to save
        path: Path to save to (will create directory if needed)
        include_metadata: Whether to include metadata like timestamp and hash
        
    Returns:
        Metadata dictionary
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Compute metadata
    metadata = {
        "spec_hash": compute_spec_hash(spec),
        "timestamp": datetime.datetime.now().isoformat(),
        "spec_type": spec.__class__.__name__,
        "name": spec.name,
    }
    
    # Save spec as JSON
    spec_file = path / "spec.json"
    with open(spec_file, "w") as f:
        json.dump(spec.dict(), f, indent=2, default=str)
    
    # Save metadata
    if include_metadata:
        metadata_file = path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Also save as pickle for exact Python object preservation
    pickle_file = path / "spec.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(spec, f)
    
    print(f"✓ Saved experiment spec to {path}")
    print(f"  Hash: {metadata['spec_hash']}")
    
    return metadata


def load_experiment_spec(path: Path, use_pickle: bool = False) -> AxiomSpec:
    """
    Load an experiment spec from disk.
    
    Args:
        path: Path to load from
        use_pickle: Whether to use pickle (exact) or JSON (portable)
        
    Returns:
        Loaded experiment spec
    """
    path = Path(path)
    
    if use_pickle:
        # Load from pickle for exact reproduction
        pickle_file = path / "spec.pkl"
        with open(pickle_file, "rb") as f:
            spec = pickle.load(f)
    else:
        # Load from JSON for portability
        spec_file = path / "spec.json"
        with open(spec_file, "r") as f:
            spec_dict = json.load(f)
        
        # Determine spec type from metadata or dict
        metadata_file = path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            spec_type = metadata.get("spec_type", "ExperimentSpec")
        else:
            spec_type = spec_dict.get("_spec_type", "ExperimentSpec")
        
        # Reconstruct appropriate spec type
        if spec_type == "TrainAndEvalSpec":
            spec = TrainAndEvalSpec(**spec_dict)
        else:
            spec = ExperimentSpec(**spec_dict)
    
    # Verify hash if metadata exists
    metadata_file = path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        current_hash = compute_spec_hash(spec)
        saved_hash = metadata["spec_hash"]
        
        if current_hash == saved_hash:
            print(f"✓ Loaded spec with verified hash: {saved_hash}")
        else:
            print(f"⚠ Hash mismatch! Saved: {saved_hash}, Current: {current_hash}")
    
    return spec


# =============================================================================
# REPRODUCIBLE EXPERIMENT RUNNER
# =============================================================================

class ReproducibleExperiment:
    """
    Wrapper for running perfectly reproducible experiments.
    """
    
    def __init__(self, spec: AxiomSpec):
        """
        Initialize reproducible experiment.
        
        Args:
            spec: Experiment specification
        """
        self.spec = spec
        self.spec_hash = compute_spec_hash(spec)
        self.runs: List[RunHandle] = []
        self.manifests: List[Dict[str, Any]] = []
        
    def ensure_determinism(self):
        """Ensure all determinism settings are properly configured."""
        import random
        import numpy as np
        
        # Ensure controls are set for determinism
        if not hasattr(self.spec, 'controls'):
            self.spec.controls = AxiomControls()
        
        self.spec.controls.enforce_determinism = True
        
        if self.spec.controls.seed is None:
            self.spec.controls.seed = 42
            print("⚠ No seed specified, using default seed=42")
        
        # Set all random seeds
        seed = self.spec.controls.seed
        random.seed(seed)
        np.random.seed(seed)
        
        # PyTorch determinism (if available)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"✓ PyTorch determinism enabled with seed={seed}")
        except ImportError:
            pass
        
        print(f"✓ Determinism ensured with seed={seed}")
    
    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """
        Create a reproducible pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Configured pipeline
        """
        pipeline = Pipeline(name="reproducible")
        
        # Deterministic initialization
        def init(state):
            import random
            import numpy as np
            
            # Re-seed at pipeline start
            seed = self.spec.controls.seed
            random.seed(seed)
            np.random.seed(seed)
            
            return {
                **state,
                "seed": seed,
                "spec_hash": self.spec_hash,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        
        pipeline.stage("init", init)
        
        # Simulated training (deterministic)
        def train(state):
            import numpy as np
            
            # Use deterministic random generator
            rng = np.random.RandomState(state["seed"])
            
            # Generate reproducible "training" results
            results = []
            for i in range(10):
                results.append(rng.randn())
            
            state["training_results"] = results
            state["final_score"] = float(np.mean(results))
            return state
        
        pipeline.stage("train", train)
        
        # Evaluation (also deterministic)
        def evaluate(state):
            import numpy as np
            
            rng = np.random.RandomState(state["seed"] + 1000)
            
            eval_scores = [state["final_score"] + rng.randn() * 0.1 
                          for _ in range(5)]
            
            state["eval_scores"] = eval_scores
            state["eval_mean"] = float(np.mean(eval_scores))
            state["eval_std"] = float(np.std(eval_scores))
            return state
        
        pipeline.stage("evaluate", evaluate)
        
        return pipeline
    
    def run(self, save_dir: Optional[Path] = None) -> RunHandle:
        """
        Run the experiment reproducibly.
        
        Args:
            save_dir: Directory to save results
            
        Returns:
            Run handle with results
        """
        print(f"\n{'='*60}")
        print(f"Running Reproducible Experiment: {self.spec.name}")
        print(f"Spec Hash: {self.spec_hash}")
        print(f"{'='*60}")
        
        # Ensure determinism
        self.ensure_determinism()
        
        # Create experiment
        if isinstance(self.spec, TrainAndEvalSpec):
            experiment = TrainAndEvalExperiment(self.spec)
        else:
            experiment = AxiomExperiment(self.spec, self.create_pipeline)
        
        experiment.prepare()
        
        # Run experiment
        run_handle = experiment.run(tag=f"repro_{self.spec_hash[:8]}")
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            self.save_run(run_handle, save_dir)
        
        self.runs.append(run_handle)
        self.manifests.append(run_handle.manifest())
        
        return run_handle
    
    def save_run(self, run_handle: RunHandle, save_dir: Path):
        """
        Save run results and manifest.
        
        Args:
            run_handle: Run handle to save
            save_dir: Directory to save to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest = run_handle.manifest()
        manifest_file = save_dir / f"manifest_{run_handle._tag}.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Save spec for this run
        spec_dir = save_dir / f"spec_{run_handle._tag}"
        save_experiment_spec(self.spec, spec_dir)
        
        print(f"✓ Saved run results to {save_dir}")
    
    def verify_reproducibility(self, runs: Optional[List[RunHandle]] = None) -> bool:
        """
        Verify that multiple runs produced identical results.
        
        Args:
            runs: Runs to compare (uses self.runs if None)
            
        Returns:
            True if all runs are identical
        """
        if runs is None:
            runs = self.runs
        
        if len(runs) < 2:
            print("⚠ Need at least 2 runs to verify reproducibility")
            return False
        
        print(f"\n{'='*60}")
        print("Verifying Reproducibility")
        print(f"{'='*60}")
        
        # Compare manifests
        base_manifest = runs[0].manifest()
        all_identical = True
        
        for i, run in enumerate(runs[1:], 1):
            manifest = run.manifest()
            
            # Compare key results
            differences = []
            
            # Check pipeline results
            base_results = base_manifest.get("pipeline_result", {})
            run_results = manifest.get("pipeline_result", {})
            
            for key in ["final_score", "eval_mean", "training_results"]:
                if key in base_results and key in run_results:
                    if base_results[key] != run_results[key]:
                        differences.append(f"  - {key}: {base_results[key]} vs {run_results[key]}")
            
            if differences:
                print(f"\n✗ Run {i+1} differs from baseline:")
                for diff in differences:
                    print(diff)
                all_identical = False
            else:
                print(f"✓ Run {i+1} matches baseline")
        
        if all_identical:
            print(f"\n✓✓ Perfect reproducibility achieved across {len(runs)} runs!")
        else:
            print(f"\n✗✗ Reproducibility check failed - results differ")
        
        return all_identical


# =============================================================================
# EXAMPLE: SAVE AND REPRODUCE
# =============================================================================

def example_save_and_reproduce():
    """
    Example: Save an experiment and reproduce it later.
    """
    print("="*70)
    print("EXAMPLE: Save and Reproduce Experiment")
    print("="*70)
    
    # Create original experiment
    original_spec = create_quick_experiment()
    original_spec.name = "original_experiment"
    original_spec.controls.seed = 12345
    original_spec.controls.enforce_determinism = True
    
    # Save the spec
    save_dir = Path("/tmp/axiom_reproducible")
    spec_dir = save_dir / "saved_spec"
    save_experiment_spec(original_spec, spec_dir)
    
    # Run original
    print("\n--- Running Original Experiment ---")
    original_runner = ReproducibleExperiment(original_spec)
    original_run = original_runner.run(save_dir / "original_run")
    
    # Load and reproduce
    print("\n--- Loading and Reproducing ---")
    loaded_spec = load_experiment_spec(spec_dir)
    loaded_spec.name = "reproduced_experiment"
    
    repro_runner = ReproducibleExperiment(loaded_spec)
    repro_run = repro_runner.run(save_dir / "reproduced_run")
    
    # Verify they match
    print("\n--- Verification ---")
    
    # Compare spec hashes
    original_hash = compute_spec_hash(original_spec)
    loaded_hash = compute_spec_hash(loaded_spec)
    
    if original_hash == loaded_hash:
        print(f"✓ Spec hashes match: {original_hash}")
    else:
        print(f"✗ Spec hashes differ: {original_hash} vs {loaded_hash}")
    
    # Run multiple times to verify determinism
    print("\n--- Multiple Run Verification ---")
    multi_runner = ReproducibleExperiment(loaded_spec)
    runs = []
    for i in range(3):
        print(f"\nRun {i+1}:")
        runs.append(multi_runner.run())
    
    multi_runner.verify_reproducibility(runs)


# =============================================================================
# EXAMPLE: EXPERIMENT VERSIONING
# =============================================================================

class ExperimentVersion:
    """Track versions of experiments for reproducibility."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize experiment versioning system.
        
        Args:
            base_dir: Base directory for storing versions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version history."""
        if self.version_file.exists():
            with open(self.version_file, "r") as f:
                return json.load(f)
        return {"versions": [], "current": None}
    
    def save_version(self, spec: AxiomSpec, tag: str = None) -> str:
        """
        Save a new version of an experiment.
        
        Args:
            spec: Experiment spec to save
            tag: Optional tag for this version
            
        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        spec_hash = compute_spec_hash(spec)[:8]
        version_id = f"v_{timestamp}_{spec_hash}"
        
        if tag:
            version_id = f"{version_id}_{tag}"
        
        # Save spec
        version_dir = self.base_dir / version_id
        metadata = save_experiment_spec(spec, version_dir)
        
        # Update version history
        version_info = {
            "id": version_id,
            "timestamp": timestamp,
            "spec_hash": spec_hash,
            "tag": tag,
            "name": spec.name,
            "metadata": metadata,
        }
        
        self.versions["versions"].append(version_info)
        self.versions["current"] = version_id
        
        # Save updated versions
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=2)
        
        print(f"✓ Saved version: {version_id}")
        return version_id
    
    def list_versions(self):
        """List all saved versions."""
        print("\nSaved Experiment Versions:")
        print("-" * 60)
        
        for v in self.versions["versions"]:
            current = " [CURRENT]" if v["id"] == self.versions["current"] else ""
            tag = f" ({v['tag']})" if v.get("tag") else ""
            print(f"{v['id']}{tag}{current}")
            print(f"  Name: {v['name']}")
            print(f"  Hash: {v['spec_hash']}")
            print(f"  Time: {v['timestamp']}")
    
    def load_version(self, version_id: str = None) -> AxiomSpec:
        """
        Load a specific version.
        
        Args:
            version_id: Version to load (current if None)
            
        Returns:
            Loaded experiment spec
        """
        if version_id is None:
            version_id = self.versions["current"]
        
        version_dir = self.base_dir / version_id
        return load_experiment_spec(version_dir)
    
    def compare_versions(self, v1: str, v2: str):
        """
        Compare two versions of an experiment.
        
        Args:
            v1: First version ID
            v2: Second version ID
        """
        print(f"\nComparing versions: {v1} vs {v2}")
        print("-" * 60)
        
        spec1 = self.load_version(v1)
        spec2 = self.load_version(v2)
        
        # Compare as dicts
        dict1 = spec1.dict()
        dict2 = spec2.dict()
        
        # Find differences
        self._print_dict_diff(dict1, dict2)
    
    def _print_dict_diff(self, d1: Dict, d2: Dict, prefix: str = ""):
        """Print differences between two dicts."""
        all_keys = set(d1.keys()) | set(d2.keys())
        
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in d1:
                print(f"  + {full_key}: {d2[key]} (added)")
            elif key not in d2:
                print(f"  - {full_key}: {d1[key]} (removed)")
            elif d1[key] != d2[key]:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    self._print_dict_diff(d1[key], d2[key], full_key)
                else:
                    print(f"  ~ {full_key}: {d1[key]} -> {d2[key]}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution demonstrating reproducibility patterns.
    """
    print("="*70)
    print("REPRODUCIBLE EXPERIMENTS COOKBOOK")
    print("="*70)
    print("\nThis recipe demonstrates how to create perfectly reproducible")
    print("experiments that can be saved, shared, and re-run exactly.\n")
    
    # Example 1: Save and reproduce
    example_save_and_reproduce()
    
    # Example 2: Version management
    print("\n" + "="*70)
    print("EXAMPLE: Experiment Version Management")
    print("="*70)
    
    version_manager = ExperimentVersion(Path("/tmp/axiom_versions"))
    
    # Create and save versions
    spec1 = create_quick_experiment()
    spec1.name = "experiment_v1"
    spec1.trainer_config.total_timesteps = 10000
    v1 = version_manager.save_version(spec1, tag="baseline")
    
    spec2 = create_quick_experiment()
    spec2.name = "experiment_v2"
    spec2.trainer_config.total_timesteps = 50000
    spec2.trainer_config.batch_size = 128
    v2 = version_manager.save_version(spec2, tag="optimized")
    
    # List versions
    version_manager.list_versions()
    
    # Compare versions
    version_manager.compare_versions(v1, v2)
    
    # Load and run a specific version
    print("\n--- Running Specific Version ---")
    loaded = version_manager.load_version(v1)
    runner = ReproducibleExperiment(loaded)
    runner.run()
    
    print("\n" + "="*70)
    print("REPRODUCIBILITY COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Always set seeds and enforce_determinism for reproducibility")
    print("2. Save both spec and manifest for complete experiment tracking")
    print("3. Use version management for experiment evolution")
    print("4. Verify reproducibility by running multiple times")
    print("5. Hash specs to detect any configuration changes")


if __name__ == "__main__":
    main()