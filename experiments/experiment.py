"""Base class for reproducible experiments."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from experiments.types import TrainingJob


class Experiment(ABC):
    """Base class for all experiments.

    Experiments encapsulate:
    1. Training configuration and launch
    2. Metadata tracking
    3. Analysis notebook generation
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize experiment.

        Args:
            name: Optional experiment name. If not provided, uses class name.
        """
        self.name = name or self.__class__.__name__
        self.metadata = {
            "experiment_class": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "user": os.environ.get("USER", "unknown"),
        }
        self.launch_results = []
        self.training_jobs: List[TrainingJob] = []

    @abstractmethod
    def launch_training_runs(self) -> Dict[str, Any]:
        """Launch training runs and return metadata.

        This method should:
        1. Launch one or more training runs
        2. Record run names and job IDs
        3. Return a summary dict

        Returns:
            Dictionary containing:
                - run_names: List of wandb run names
                - job_ids: List of sky job IDs
                - launch_results: Full launch result dicts
                - success: Overall success boolean
        """
        pass

    @abstractmethod
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get configuration for analysis notebook generation.

        Returns:
            Dictionary with analysis configuration like:
                - metrics_to_plot: List of metric names
                - eval_suites: List of eval suites to check
                - custom_analysis: Any experiment-specific config
        """
        pass

    def generate_notebook(self, output_dir: Optional[str] = None) -> str:
        """Generate analysis notebook for the experiment.

        Args:
            output_dir: Directory to save notebook. If None, uses experiments/log/.

        Returns:
            Path to generated notebook
        """
        if not self.training_jobs:
            raise ValueError("No training runs launched yet. Call launch_training_runs() first.")

        # Extract run names and job IDs from TrainingJob objects
        wandb_run_ids = [job.wandb_run_id for job in self.training_jobs]
        skypilot_job_ids = [job.skypilot_job_id for job in self.training_jobs if job.skypilot_job_id]

        if not wandb_run_ids:
            raise ValueError("No successful runs to analyze")

        # Add analysis config to metadata
        analysis_config = self.get_analysis_config()
        self.metadata["analysis_config"] = analysis_config

        # Default to experiments/log/ directory
        if output_dir is None:
            output_dir = os.path.join("experiments", "log")

        # Generate notebook
        from experiments.notebooks.generation import generate_notebook_from_template

        return generate_notebook_from_template(
            experiment_name=self.name,
            run_names=wandb_run_ids,
            sky_job_ids=skypilot_job_ids if skypilot_job_ids else None,
            additional_metadata=self.metadata,
            output_dir=output_dir,
        )

    def run(self, generate_notebook: bool = True) -> Dict[str, Any]:
        """Run the complete experiment workflow.

        Args:
            generate_notebook: Whether to generate analysis notebook

        Returns:
            Dictionary with experiment results
        """
        print(f"Running experiment: {self.name}")
        print("=" * 50)

        # Launch training runs
        launch_summary = self.launch_training_runs()

        # Generate notebook if requested and launches succeeded
        notebook_path = None
        if generate_notebook and launch_summary.get("success", False):
            try:
                notebook_path = self.generate_notebook()
                print(f"\nGenerated analysis notebook: {notebook_path}")
            except Exception as e:
                print(f"\nWarning: Failed to generate notebook: {str(e)}")

        return {
            "experiment_name": self.name,
            "launch_summary": launch_summary,
            "notebook_path": notebook_path,
            "metadata": self.metadata,
        }

    def save_metadata(self, filepath: Optional[str] = None) -> str:
        """Save experiment metadata to JSON file.

        Args:
            filepath: Path to save metadata. If None, auto-generates in experiments/log/.

        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}_metadata.json"
            filepath = os.path.join("experiments", "log", filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(
                {
                    "metadata": self.metadata,
                    "launch_results": self.launch_results,
                },
                f,
                indent=2,
            )

        print(f"Saved metadata to: {filepath}")
        return filepath
