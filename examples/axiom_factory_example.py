"""Example of using the tAXIOM factory pattern for sweeps.

This demonstrates the proper factory pattern where:
1. Users provide configuration, not context
2. Factory returns pipelines, not results
3. Orchestration is handled externally
"""

from metta.common.wandb.wandb_context import WandbConfig
from metta.sweep.axiom import Ctx
from metta.sweep.axiom_sweep_factory import (
    get_multi_trial_sweep,
    get_single_trial_sweep,
)
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.sweep_config import SweepConfig
from metta.tools.train import TrainTool


def example_single_trial():
    """Example of running a single trial using the factory pattern."""
    print("\n" + "="*60)
    print("Single Trial Example")
    print("="*60)
    
    # 1. Create configuration (users only provide config)
    config = SweepConfig(
        sweep_name="single_trial_example",
        num_trials=1,
        protein=ProteinConfig(
            metric="navigation_score",
            search_space={
                "learning_rate": {"type": "log_uniform", "min": 1e-4, "max": 1e-2},
                "batch_size": {"type": "choice", "values": [32, 64, 128]},
            }
        ),
        wandb=WandbConfig(
            entity="example-entity",
            project="example-project",
            tags=["factory", "example"]
        ),
        train_tool_factory=lambda run_name: TrainTool(run_name=run_name),
        evaluation_simulations=[]
    )
    
    # 2. Get pipeline from factory (factory returns pipeline, not results)
    pipeline = get_single_trial_sweep(config)
    
    print("Created pipeline for single trial")
    print(f"Pipeline has {len(pipeline.stages)} operations")
    
    # 3. Run the pipeline (orchestration happens here)
    # Note: Context is created by pipeline.run(), users never see it
    result = pipeline.run()
    
    print(f"Single trial complete: {result}")


def example_multi_trial():
    """Example of running multiple trials using the factory pattern."""
    print("\n" + "="*60)
    print("Multi-Trial Example")
    print("="*60)
    
    # 1. Create configuration
    config = SweepConfig(
        sweep_name="multi_trial_example",
        num_trials=3,
        protein=ProteinConfig(
            metric="exploration_score",
            search_space={
                "learning_rate": {"type": "log_uniform", "min": 1e-4, "max": 1e-2},
                "hidden_dim": {"type": "choice", "values": [128, 256, 512]},
            }
        ),
        wandb=WandbConfig(
            entity="example-entity",
            project="example-project",
            tags=["factory", "multi-trial"]
        ),
        train_tool_factory=lambda run_name: TrainTool(run_name=run_name),
        evaluation_simulations=[]
    )
    
    # 2. Get factory (not results!)
    factory = get_multi_trial_sweep(config)
    
    print(f"Created factory for {config.num_trials} trials")
    
    # 3. Get pipelines from factory
    init_pipeline = factory.get_init_pipeline()
    trial_pipeline = factory.get_trial_pipeline()
    
    print("Got initialization and trial pipelines from factory")
    
    # 4. Orchestrate execution (external to factory)
    # This is where the actual execution happens
    
    # Initialize once
    print("\nRunning initialization...")
    init_result = init_pipeline.run()
    print(f"Initialization complete: {init_result}")
    
    # Run trials
    results = []
    for i in range(config.num_trials):
        print(f"\nRunning trial {i+1}/{config.num_trials}...")
        
        # Create context for this trial (orchestrator's responsibility)
        ctx = Ctx()
        ctx.metadata["trial_index"] = i
        
        # Pass previous result if available
        if results:
            ctx.set_stage_input("suggest", results[-1])
        
        # Run trial pipeline with context
        result = trial_pipeline.run(ctx)
        results.append(result)
        
        print(f"Trial {i+1} complete")
    
    print(f"\nAll {len(results)} trials complete")
    return results


def example_custom_orchestration():
    """Example of custom orchestration with the factory pattern."""
    print("\n" + "="*60)
    print("Custom Orchestration Example")
    print("="*60)
    
    # Configuration
    config = SweepConfig(
        sweep_name="custom_example",
        num_trials=5,
        protein=ProteinConfig(
            metric="custom_metric",
            search_space={
                "param1": {"type": "uniform", "min": 0, "max": 1},
                "param2": {"type": "choice", "values": ["A", "B", "C"]},
            }
        ),
        wandb=WandbConfig(
            entity="example-entity",
            project="example-project"
        ),
        train_tool_factory=lambda run_name: TrainTool(run_name=run_name),
    )
    
    # Get factory
    factory = get_multi_trial_sweep(config)
    
    # Custom orchestration: Run trials with early stopping
    init_pipeline = factory.get_init_pipeline()
    trial_pipeline = factory.get_trial_pipeline()
    
    # Initialize
    init_pipeline.run()
    
    # Run trials with custom logic
    best_score = float('-inf')
    patience = 2
    no_improvement = 0
    
    for i in range(config.num_trials):
        print(f"\nTrial {i+1}...")
        
        # Run trial
        ctx = Ctx()
        ctx.metadata["trial_index"] = i
        ctx.metadata["best_score"] = best_score
        
        result = trial_pipeline.run(ctx)
        
        # Custom early stopping logic
        score = result.get("score", 0)
        if score > best_score:
            best_score = score
            no_improvement = 0
            print(f"New best score: {best_score:.4f}")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} trials")
        
        # Early stopping
        if no_improvement >= patience:
            print(f"Early stopping after {i+1} trials")
            break
    
    print(f"Custom orchestration complete. Best score: {best_score:.4f}")


def main():
    """Run all examples."""
    print("""
tAXIOM Factory Pattern Examples
================================

The factory pattern ensures:
1. Users provide configuration, never context
2. Factories return pipelines, not results
3. Orchestration is external and flexible

Key concepts:
- SweepConfig: All configuration in one place
- get_single_trial_sweep(): Returns a pipeline for one trial
- get_multi_trial_sweep(): Returns a factory with init and trial pipelines
- Orchestration: External control of pipeline execution
    """)
    
    # Note: These would actually run if the dependencies were available
    # For demonstration, we'll just show the structure
    
    try:
        # Single trial example
        example_single_trial()
    except Exception as e:
        print(f"Single trial example (would run with dependencies): {e}")
    
    try:
        # Multi-trial example
        example_multi_trial()
    except Exception as e:
        print(f"Multi-trial example (would run with dependencies): {e}")
    
    try:
        # Custom orchestration example
        example_custom_orchestration()
    except Exception as e:
        print(f"Custom orchestration example (would run with dependencies): {e}")
    
    print("\n" + "="*60)
    print("Factory Pattern Summary")
    print("="*60)
    print("""
The factory pattern provides clean separation:

1. Configuration → SweepConfig
2. Factory → get_multi_trial_sweep(config)
3. Pipelines → factory.get_init_pipeline(), factory.get_trial_pipeline()
4. Orchestration → External control with Ctx

This design ensures users never handle context directly,
and factories return composable pipelines, not results.
    """)


if __name__ == "__main__":
    main()