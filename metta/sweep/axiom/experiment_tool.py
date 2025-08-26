"""Wrapper to make AxiomExperiment instances runnable as Tools."""

from typing import Any

from metta.common.config.tool import Tool
from metta.sweep.axiom.experiment import AxiomExperiment


class ExperimentTool(Tool):
    """Tool wrapper for AxiomExperiment instances.
    
    This allows experiments to be run via the standard Tool interface
    while maintaining the separation between experiment logic and Tool framework.
    """
    
    # The wrapped experiment
    experiment: Any = None  # Will be an AxiomExperiment instance
    
    # Tool interface - mark all create_experiment args as consumed
    consumed_args: list[str] = [
        "name", "total_timesteps", "num_agents", "rollout_workers",
        "batch_size", "minibatch_size", "learning_rate", "num_eval_episodes",
        "policy_path", "use_curriculum", "enable_combat"
    ]
    
    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Execute the wrapped experiment."""
        if self.experiment is None:
            raise ValueError("No experiment configured")
            
        # If the experiment has its own invoke, use it
        if hasattr(self.experiment, 'invoke'):
            return self.experiment.invoke(args, overrides)
        
        # Otherwise use the default experiment flow
        self.experiment.prepare()
        handle = self.experiment.run(tag=args.get('tag', 'default'))
        
        # Return 0 for success
        manifest = handle.manifest()
        if 'pipeline_result' in manifest:
            status = manifest['pipeline_result'].get('status', 'unknown')
        else:
            status = manifest.get('status', 'unknown')
            
        return 0 if status == 'complete' else 1


def create_train_and_eval_tool(**kwargs) -> ExperimentTool:
    """Factory function to create a TrainAndEval experiment as a Tool.
    
    This is what gets called by run.py.
    """
    from metta.sweep.axiom.train_and_eval import create_experiment, TrainAndEvalExperiment
    
    # Filter out None values from kwargs for cleaner creation
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Create the experiment spec
    spec = create_experiment(**filtered_kwargs)
    
    # Create the experiment
    experiment = TrainAndEvalExperiment(spec)
    
    # Wrap it in a tool  
    tool = ExperimentTool(
        experiment=experiment,
        system=spec.system_config
    )
    
    return tool