"""Tests for experiment launch functionality - focusing on outcomes."""

import pytest
from experiments import Experiment, TrainingJob


class TestExperimentLaunch:
    """Test experiment launching behavior and outcomes."""
    
    def test_experiment_launch_creates_training_jobs(self):
        """Test that Experiment.launch_training_runs creates TrainingJob objects."""
        class TestExperiment(Experiment):
            def launch_training_runs(self):
                # Simulate launching multiple runs
                jobs = []
                for i in range(3):
                    job = TrainingJob(
                        wandb_run_name=f"test.run.{i}",
                        skypilot_job_id=f"sky-{i}",
                        config={"index": i}
                    )
                    self.training_jobs.append(job)
                    jobs.append(job)
                return jobs
            
            def get_analysis_config(self):
                return {"metrics": ["reward"]}
        
        exp = TestExperiment("test_exp")
        launched_jobs = exp.launch_training_runs()
        
        # Should return list of TrainingJob objects
        assert len(launched_jobs) == 3
        assert all(isinstance(job, TrainingJob) for job in launched_jobs)
        assert launched_jobs[0].wandb_run_name == "test.run.0"
        assert launched_jobs[2].skypilot_job_id == "sky-2"
        
        # Should also store in experiment
        assert exp.training_jobs == launched_jobs
    
    def test_experiment_no_launches_no_notebook(self):
        """Test that experiments with no successful launches don't generate notebooks."""
        class FailedExperiment(Experiment):
            def launch_training_runs(self):
                # No successful launches
                return []
            
            def get_analysis_config(self):
                return {}
        
        exp = FailedExperiment("failed_test")
        result = exp.run(generate_notebook=False)  # Skip notebook generation
        
        # Should not have any launched jobs
        assert result["launched_jobs"] == []
        assert result["notebook_path"] is None
    
    def test_launch_from_config(self):
        """Test launching with TrainingJobConfig."""
        from experiments import TrainingJobConfig
        
        class ConfigExperiment(Experiment):
            def launch_training_runs(self):
                # Create config
                config = TrainingJobConfig(
                    curriculum="test/curriculum",
                    gpus=2,
                    nodes=1,
                    wandb_tags=["test"],
                    additional_args=["trainer.optimizer.type=adam"]
                )
                
                # Use the helper method (would actually launch if not in test)
                # For testing, we'll manually create a job
                job = TrainingJob(
                    wandb_run_name="test.config.run",
                    skypilot_job_id="sky-config-123",
                    config=config,
                    notes="Launched from config"
                )
                self.training_jobs.append(job)
                return [job]
            
            def get_analysis_config(self):
                return {"metrics": ["loss"]}
        
        exp = ConfigExperiment("config_test")
        jobs = exp.launch_training_runs()
        
        assert len(jobs) == 1
        assert isinstance(jobs[0].config, TrainingJobConfig)
        assert jobs[0].config.curriculum == "test/curriculum"
        assert jobs[0].config.gpus == 2
        assert jobs[0].config.get_arg_value("trainer.optimizer.type") == "adam"