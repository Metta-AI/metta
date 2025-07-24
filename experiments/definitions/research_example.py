#!/usr/bin/env python3
"""Example of using the research experiment workflow.

This shows how to:
1. Create a research notebook
2. Use the ResearchExperiment class for tracking
3. Launch iterative experiments based on results
"""

from experiments.research_experiment import ResearchExperiment, RunConfig


def example_research_workflow():
    """Example research workflow showing iterative experimentation."""
    
    # Create research experiment tracker
    research = ResearchExperiment(
        name="optimizer_comparison",
        description="Comparing different optimizers and learning rates for arena training"
    )
    
    # Iteration 1: Baseline
    print("=== Iteration 1: Establishing Baseline ===")
    research.add_hypothesis("Muon optimizer with lr=0.0045 provides good baseline performance")
    
    baseline = RunConfig(
        name="baseline",
        learning_rate=0.0045,
        optimizer="muon",
        num_gpus=2,
        num_nodes=1,
        notes="Baseline configuration from arena.sh"
    )
    
    result1 = research.launch_run(baseline)
    
    if result1["success"]:
        research.add_learning("Baseline launched successfully")
    
    # Move to next iteration
    research.next_iteration()
    
    # Iteration 2: Optimizer comparison
    print("\n=== Iteration 2: Optimizer Ablation ===")
    research.add_hypothesis("Adam optimizer might work better with different learning rates")
    
    # Launch ablation grid
    ablation_results = research.launch_ablation_grid(
        base_config=RunConfig(
            name="optimizer_ablation",
            num_gpus=2,
            num_nodes=1
        ),
        param_grids={
            "trainer.optimizer.type": ["adam", "muon", "sgd"],
            "trainer.optimizer.learning_rate": [0.001, 0.0045, 0.01]
        }
    )
    
    successful = sum(1 for r in ablation_results if r.get("success", False))
    research.add_learning(f"Launched {successful}/{len(ablation_results)} ablation runs")
    
    # Save research log
    log_path = research.save_research_log()
    
    # Print summary
    summary = research.get_summary()
    print(f"\n=== Research Summary ===")
    print(f"Total runs launched: {summary['total_runs']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Hypotheses tested: {summary['hypotheses']}")
    print(f"Learnings documented: {summary['learnings']}")
    print(f"\nResearch log saved to: {log_path}")
    
    return research


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Research experiment example")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("[DRY RUN MODE - Commands will be printed but not executed]")
        # Override launch function
        import experiments.launch
        original_launch = experiments.launch.launch_training_run
        
        def dry_run_launch(**kwargs):
            kwargs["dry_run"] = True
            return original_launch(**kwargs)
            
        experiments.launch.launch_training_run = dry_run_launch
    
    # First, generate a research notebook
    print("Generating research notebook...")
    from experiments.notebooks.research_generation import generate_research_notebook
    
    notebook_path = generate_research_notebook(
        name="optimizer_research",
        description="Research notebook for optimizer comparison experiments"
    )
    
    print(f"\nGenerated notebook: {notebook_path}")
    print("\nNow demonstrating ResearchExperiment workflow...")
    print("-" * 50)
    
    # Run example workflow
    research = example_research_workflow()
    
    print("\n" + "=" * 50)
    print("Research workflow complete!")
    print(f"\n1. Open the notebook: jupyter notebook {notebook_path}")
    print("2. Use the notebook to monitor and analyze your runs")
    print("3. The ResearchExperiment class tracks all your iterations")
    
    return 0


if __name__ == "__main__":
    exit(main())