"""Base class for reproducible experiments."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from experiments.types import TrainingJob, TrainingJobConfig, BaseExperimentConfig
from experiments.launch import launch_training_run
from experiments.monitoring import get_wandb_run_name_from_sky_job
import subprocess


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
    def launch_training_runs(self) -> List[TrainingJob]:
        """Launch training runs and return TrainingJob objects.

        This method should:
        1. Launch one or more training runs
        2. Create TrainingJob objects for successful launches
        3. Store jobs in self.training_jobs
        4. Return the list of TrainingJob objects

        Returns:
            List of TrainingJob objects for successful launches
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
    
    def launch_training_run_from_config(self, run_name: str, config: TrainingJobConfig) -> Optional[TrainingJob]:
        """Launch a training run from a TrainingJobConfig.
        
        Args:
            run_name: Name for the training run
            config: TrainingJobConfig with launch parameters
            
        Returns:
            TrainingJob if successful, None otherwise
        """
        result = launch_training_run(
            run_name=run_name,
            curriculum=config.curriculum,
            gpus=config.gpus,
            nodes=config.nodes,
            no_spot=config.no_spot,
            skip_git_check=config.skip_git_check,
            additional_args=config.additional_args,
            wandb_tags=config.wandb_tags,
        )
        
        self.launch_results.append(result)
        
        if result["success"]:
            job = TrainingJob(
                wandb_run_name=run_name,
                skypilot_job_id=result.get("job_id"),
                config=config,
            )
            self.training_jobs.append(job)
            return job
        
        return None

    def generate_notebook(self, output_dir: Optional[str] = None) -> str:
        """Generate analysis notebook for the experiment.

        Args:
            output_dir: Directory to save notebook. If None, uses experiments/scratch/.

        Returns:
            Path to generated notebook
        """
        if not self.training_jobs:
            raise ValueError("No training runs launched yet. Call launch_training_runs() first.")

        # Extract run names and job IDs from TrainingJob objects
        wandb_run_names = [job.wandb_run_name for job in self.training_jobs]
        skypilot_job_ids = [job.skypilot_job_id for job in self.training_jobs if job.skypilot_job_id]

        if not wandb_run_names:
            raise ValueError("No successful runs to analyze")

        # Add analysis config to metadata
        analysis_config = self.get_analysis_config()
        self.metadata["analysis_config"] = analysis_config

        # Default to experiments/scratch/ directory
        if output_dir is None:
            output_dir = os.path.join("experiments", "scratch")

        # Generate notebook
        from experiments.notebooks.generation import generate_notebook_from_template

        return generate_notebook_from_template(
            experiment_name=self.name,
            run_names=wandb_run_names,  # Maps to wandb_run_names internally
            sky_job_ids=skypilot_job_ids if skypilot_job_ids else None,  # Maps to skypilot_job_ids internally
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
        launched_jobs = self.launch_training_runs()

        # Generate notebook if requested and launches succeeded
        notebook_path = None
        if generate_notebook and launched_jobs:
            try:
                notebook_path = self.generate_notebook()
                print(f"\nGenerated analysis notebook: {notebook_path}")
            except Exception as e:
                print(f"\nWarning: Failed to generate notebook: {str(e)}")

        return {
            "experiment_name": self.name,
            "launched_jobs": launched_jobs,
            "notebook_path": notebook_path,
            "metadata": self.metadata,
        }

    def save_metadata(self, filepath: Optional[str] = None) -> str:
        """Save experiment metadata to JSON file.

        Args:
            filepath: Path to save metadata. If None, auto-generates in experiments/scratch/.

        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}_metadata.json"
            filepath = os.path.join("experiments", "scratch", filename)

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
    
    @classmethod
    def create_notebook(cls, config: BaseExperimentConfig) -> str:
        """Create a notebook from configuration, handling job loading and launching.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Path to generated notebook
        """
        # Create instance
        instance = cls(config.name)
        
        # Load existing jobs if provided
        if config.job_ids:
            print(f"Loading {len(config.job_ids)} existing jobs...")
            for job_id in config.job_ids:
                run_name = get_wandb_run_name_from_sky_job(job_id)
                if run_name:
                    job = TrainingJob(
                        wandb_run_name=run_name,
                        skypilot_job_id=job_id,
                    )
                    instance.training_jobs.append(job)
                    print(f"  ✓ Job {job_id} → {run_name}")
                else:
                    print(f"  ✗ Job {job_id} → Could not find run name")
        
        # Launch new runs if requested (and no jobs loaded)
        if config.launch and not instance.training_jobs:
            instance.launch_training_runs()
        
        # Generate notebook
        from experiments.notebooks.generation import generate_notebook
        
        wandb_run_names = [job.wandb_run_name for job in instance.training_jobs] if instance.training_jobs else None
        skypilot_job_ids = [job.skypilot_job_id for job in instance.training_jobs if job.skypilot_job_id] if instance.training_jobs else None
        
        notebook_path = generate_notebook(
            name=config.name,
            description=config.description or f"Notebook for {instance.__class__.__name__}",
            sections=config.sections,
            wandb_run_names=wandb_run_names,
            skypilot_job_ids=skypilot_job_ids,
            additional_metadata={
                **instance.metadata,
                "from_recipe": True,
            },
            output_dir=config.output_dir,
        )
        
        # Open notebook if requested
        if config.open_notebook:
            print("\nOpening notebook in Jupyter...")
            try:
                subprocess.Popen(
                    ["uv", "run", "jupyter", "notebook", notebook_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("✓ Jupyter notebook launched")
            except Exception as e:
                print(f"Failed to open notebook: {e}")
                print(f"\nTo open manually:\n  uv run jupyter notebook {notebook_path}")
        
        return notebook_path
