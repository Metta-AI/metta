"""AxiomExperiment: Spec-driven experimental control."""

import hashlib
import json
import os
import platform
import random
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import torch

from metta.common.config.tool import Tool
from metta.sweep.axiom.core import Ctx, Pipeline
from metta.sweep.axiom.experiment_spec import AxiomControls, ExperimentSpec
from metta.sweep.axiom.manifest import diff_manifests


class RunHandle:
    """Handle to a completed experiment run."""
    
    def __init__(self, tag: str, manifest: dict[str, Any], run_dir: str):
        self.tag = tag
        self._manifest = manifest
        self.run_dir = run_dir
        self.manifest_path = os.path.join(run_dir, f"{tag}.manifest.json")
    
    def manifest(self) -> dict[str, Any]:
        """Get the run manifest."""
        return self._manifest


class AxiomExperiment:
    """Spec-driven experiment harness for reproducible ML/RL experiments.
    
    This class consumes an ExperimentSpec that contains ALL configuration
    needed to build and run experiments. The spec is the single source of
    truth for reproducibility.
    
    Key features:
    1. Spec-driven: All config comes from ExperimentSpec
    2. Pipeline factories: Tools are created from spec configs
    3. Controlled execution: Deterministic seeding and environment
    4. Auditable: Generates manifests for every run
    5. Composable: Supports hierarchical join points
    
    Note: Subclasses that need Tool functionality should inherit from both
    AxiomExperiment and Tool, with Tool last in the MRO.
    """
    
    def __init__(
        self,
        spec: ExperimentSpec,
        pipeline_factory: Optional[Callable[[dict], Pipeline]] = None,
    ):
        """Initialize experiment from spec.
        
        Args:
            spec: Complete experiment specification
            pipeline_factory: Optional factory to create pipeline from config
                             If not provided, will use default factories
        """
        self.spec = spec
        self.pipeline_factory = pipeline_factory or self._get_default_factory()
        
        # Track runs
        self.runs: dict[str, RunHandle] = {}
        
        # Ensure run directory exists
        os.makedirs(spec.run_dir, exist_ok=True)
        
        # Set up fingerprinting functions
        self._setup_fingerprinters()
        
        # Cache for built pipelines
        self._pipeline_cache: dict[str, Pipeline] = {}
    
    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Execute the experiment as a Tool.
        
        This is the Tool interface - subclasses should override this
        to implement their specific experiment logic.
        """
        # Default implementation runs the standard experiment flow
        self.prepare()
        handle = self.run(tag=args.get('tag', 'default'))
        
        # Return 0 for success
        return 0 if handle.manifest().get('status') == 'complete' else 1
    
    def _get_default_factory(self) -> Callable[[dict], Pipeline]:
        """Get default pipeline factory based on pipeline type."""
        if self.spec.pipeline_type == "training":
            return self._create_training_pipeline
        elif self.spec.pipeline_type == "sweep":
            return self._create_sweep_pipeline
        else:
            # Generic factory that expects a Tool with get_pipeline()
            def generic_factory(config: dict) -> Pipeline:
                # This would need to be implemented based on your Tool structure
                raise NotImplementedError(
                    f"No default factory for pipeline type: {self.spec.pipeline_type}"
                )
            return generic_factory
    
    def _create_training_pipeline(self, config: dict) -> Pipeline:
        """Create training pipeline from config."""
        from metta.tools.train_pipeline import TrainJobPipeline
        
        # Create TrainJobPipeline from config
        tool = TrainJobPipeline(**config)
        return tool.get_pipeline()
    
    def _create_sweep_pipeline(self, config: dict) -> Pipeline:
        """Create sweep pipeline from config."""
        from metta.sweep.axiom.sequential_sweep import SequentialSweepPipeline
        
        # Create sweep pipeline from config
        sweep = SequentialSweepPipeline(**config)
        return sweep.build_pipeline()
    
    def _setup_fingerprinters(self):
        """Set up dataset and environment fingerprinting functions."""
        # These would map string names to actual functions
        # For now, use simple defaults
        
        if self.spec.dataset_hasher:
            # Look up the hasher by name
            self.dataset_hasher = lambda: f"dataset_{self.spec.dataset_hasher}"
        else:
            self.dataset_hasher = None
        
        if self.spec.env_hasher:
            # Look up the hasher by name
            self.env_hasher = lambda: f"env_{self.spec.env_hasher}"
        else:
            self.env_hasher = None
    
    def prepare(self) -> None:
        """Prepare experiment by freezing control variables."""
        # Apply seed fan-out
        self._seed_fanout(
            self.spec.controls.seed,
            self.spec.controls.enforce_determinism
        )
        
        # Compute fingerprints
        self.fingerprints = {
            "dataset": self.dataset_hasher() if self.dataset_hasher else None,
            "env": self.env_hasher() if self.env_hasher else None,
        }
        
        # Capture environment
        self.captured_env = {
            var: os.environ.get(var) 
            for var in self.spec.controls.capture_env_vars
        }
    
    def build_pipeline(self) -> Pipeline:
        """Build the main pipeline from spec.
        
        This creates the pipeline and applies any provided joins.
        
        Returns:
            Configured Pipeline ready to run
        """
        # Check cache
        cache_key = json.dumps(self.spec.pipeline_config, sort_keys=True)
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        
        # Create base pipeline from config
        pipeline = self.pipeline_factory(self.spec.pipeline_config)
        
        # Apply provided joins if the pipeline supports them
        for join_name, impl_name in self.spec.provided_joins.items():
            # Get the join config if provided
            join_config = self.spec.join_configs.get(impl_name, {})
            
            # Create the join implementation
            join_pipeline = self._create_join_pipeline(impl_name, join_config)
            
            # Apply the join only if pipeline has required joins
            if hasattr(pipeline, '_required_joins') and join_name in pipeline._required_joins:
                pipeline = pipeline.provide_join(join_name, join_pipeline)
            # Otherwise, just skip (the pipeline doesn't support this join point)
        
        # Cache the built pipeline
        self._pipeline_cache[cache_key] = pipeline
        
        return pipeline
    
    def _create_join_pipeline(self, impl_name: str, config: dict) -> Pipeline:
        """Create a join implementation pipeline.
        
        Args:
            impl_name: Name of the implementation
            config: Configuration for the implementation
        
        Returns:
            Pipeline implementing the join
        """
        # This would use a registry of join implementations
        # For now, create simple examples
        
        if impl_name == "adam":
            return Pipeline().stage("adam", lambda s: {**s, "optimizer": "adam"})
        elif impl_name == "sgd":
            return Pipeline().stage("sgd", lambda s: {**s, "optimizer": "sgd"})
        elif impl_name == "muon":
            return Pipeline().stage("muon", lambda s: {**s, "optimizer": "muon"})
        else:
            raise ValueError(f"Unknown join implementation: {impl_name}")
    
    def run(
        self,
        tag: str = "default",
        override_joins: Optional[dict[str, str]] = None,
        **kwargs
    ) -> RunHandle:
        """Run the experiment.
        
        Args:
            tag: Unique identifier for this run
            override_joins: Override spec's provided_joins for this run
            **kwargs: Additional metadata for the run
        
        Returns:
            RunHandle with manifest and results
        """
        # Apply any join overrides for this run
        if override_joins:
            # Create a modified spec for this run
            run_spec = self.spec.model_copy(deep=True)
            run_spec.provided_joins.update(override_joins)
            
            # Build pipeline with overrides
            original_spec = self.spec
            self.spec = run_spec
            pipeline = self.build_pipeline()
            self.spec = original_spec
        else:
            # Use standard pipeline
            pipeline = self.build_pipeline()
        
        # Create context
        ctx = Ctx()
        ctx.metadata.update({
            "experiment": self.spec.name,
            "tag": tag,
            "run_dir": self.spec.run_dir,
            **kwargs,
        })
        
        # Run pipeline
        start_time = datetime.utcnow()
        result = pipeline.run(ctx)
        end_time = datetime.utcnow()
        
        # Build manifest
        manifest = self._build_manifest(tag, result, start_time, end_time, override_joins)
        
        # Save manifest
        manifest_path = os.path.join(self.spec.run_dir, f"{tag}.manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Save spec alongside manifest
        spec_path = os.path.join(self.spec.run_dir, f"{tag}.spec.json")
        with open(spec_path, "w") as f:
            json.dump(self.spec.model_dump(), f, indent=2)
        
        # Create handle
        handle = RunHandle(tag, manifest, self.spec.run_dir)
        self.runs[tag] = handle
        
        return handle
    
    def run_comparison(
        self,
        baseline_joins: Optional[dict[str, str]] = None,
        variants: Optional[dict[str, dict[str, str]]] = None,
    ) -> dict[str, RunHandle]:
        """Run baseline and variants for comparison.
        
        Args:
            baseline_joins: Join implementations for baseline
            variants: Dict of variant_name -> join implementations
        
        Returns:
            Dict mapping run names to RunHandles
        """
        results = {}
        
        # Run baseline
        self.prepare()
        baseline = self.run("baseline", override_joins=baseline_joins)
        results["baseline"] = baseline
        
        # Run variants
        if variants:
            for variant_name, variant_joins in variants.items():
                # Check single-factor if enabled
                if self.spec.controls.single_factor_enforce:
                    self._assert_single_factor_change(
                        baseline_joins or {},
                        variant_joins
                    )
                
                # Run variant
                self.prepare()  # Reset seeds for each variant
                variant = self.run(variant_name, override_joins=variant_joins)
                results[variant_name] = variant
        
        return results
    
    def diff(self, baseline: RunHandle, variant: RunHandle) -> str:
        """Generate human-readable diff between runs.
        
        Args:
            baseline: Baseline run
            variant: Variant run
        
        Returns:
            Formatted diff string
        """
        diff_result = diff_manifests(
            baseline.manifest(),
            variant.manifest(),
        )
        
        return self._format_diff(diff_result)
    
    def _seed_fanout(self, seed: int, deterministic: bool) -> None:
        """Apply seed to all RNGs and set determinism flags."""
        # Python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Deterministic algorithms
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set CUBLAS workspace config if not already set
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    def _build_manifest(
        self,
        tag: str,
        result: Any,
        start_time: datetime,
        end_time: datetime,
        override_joins: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Build experiment manifest."""
        # Determine actual joins used
        actual_joins = self.spec.provided_joins.copy()
        if override_joins:
            actual_joins.update(override_joins)
        
        manifest = {
            "experiment": self.spec.name,
            "description": self.spec.description,
            "tag": tag,
            "timestamp": start_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            
            # Full spec
            "spec": self.spec.model_dump(),
            
            # Control variables
            "controls": self.spec.controls.model_dump(),
            
            # Fingerprints
            "fingerprints": self.fingerprints,
            
            # Environment
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "captured_vars": self.captured_env,
            },
            
            # Hardware
            "hardware": self._capture_hardware(),
            
            # Code
            "code": self._capture_code(),
            
            # Joins
            "joins": {
                "exposed": self.spec.exposed_joins,
                "provided": list(actual_joins.keys()),
                "implementations": actual_joins,
            },
            
            # Pipeline result
            "pipeline_result": result if isinstance(result, dict) else str(result),
        }
        
        return manifest
    
    def _capture_hardware(self) -> dict[str, Any]:
        """Capture hardware information."""
        info = {
            "cpu_count": os.cpu_count(),
            "platform": platform.machine(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
            })
        
        return info
    
    def _capture_code(self) -> dict[str, Any]:
        """Capture code version information."""
        info = {}
        
        # Try to get git info
        try:
            import subprocess
            
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__),
            )
            if result.returncode == 0:
                info["git_commit"] = result.stdout.strip()
            
            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__),
            )
            if result.returncode == 0:
                info["git_branch"] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__),
            )
            if result.returncode == 0:
                info["has_uncommitted_changes"] = bool(result.stdout.strip())
        except Exception:
            pass
        
        return info
    
    def _format_diff(self, diff: dict[str, Any]) -> str:
        """Format diff for human consumption."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Experiment Diff: {self.spec.name}")
        lines.append("=" * 60)
        
        # Group by category
        categories = {
            "Spec": ["spec"],
            "Controls": ["controls", "fingerprints"],
            "Joins": ["joins"],
            "Environment": ["environment"],
            "Code": ["code"],
        }
        
        for category, keys in categories.items():
            category_changes = []
            for key in keys:
                if key in diff and diff[key]:
                    category_changes.append((key, diff[key]))
            
            if category_changes:
                lines.append(f"\n[{category}]")
                for key, changes in category_changes:
                    lines.append(f"  {key}:")
                    if isinstance(changes, dict):
                        for k, v in changes.items():
                            lines.append(f"    {k}: {v}")
                    else:
                        lines.append(f"    {changes}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _assert_single_factor_change(
        self,
        baseline_joins: dict[str, str],
        variant_joins: dict[str, str],
    ) -> None:
        """Assert that only one join changed between configurations.
        
        Args:
            baseline_joins: Baseline join implementations
            variant_joins: Variant join implementations
        
        Raises:
            ValueError: If multiple joins changed
        """
        # Find differences
        all_keys = set(baseline_joins.keys()) | set(variant_joins.keys())
        changes = []
        
        for key in all_keys:
            baseline_val = baseline_joins.get(key)
            variant_val = variant_joins.get(key)
            if baseline_val != variant_val:
                changes.append(key)
        
        if len(changes) > 1:
            raise ValueError(
                f"Single-factor enforcement failed: {len(changes)} joins changed: {changes}"
            )