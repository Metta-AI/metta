"""Unified notebook creation for experiments and research.

Usage:
    # Create empty notebook for research
    ./experiments/new_notebook.py my_research
    
    # Create notebook from experiment recipe
    ./experiments/new_notebook.py arena_test --recipe arena
    
    # Create notebook with custom launch config
    ./experiments/new_notebook.py my_custom --gpus 4 --curriculum env/mettagrid/curriculum/test
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add parent directory to path so imports work when run from experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.notebooks.generation import generate_notebook, AVAILABLE_SECTIONS
from experiments.types import TrainingJobConfig
from experiments.launch import launch_training_run


def create_research_notebook(name: str, description: str = "", sections: Optional[List[str]] = None) -> str:
    """Create an empty research notebook."""
    # Script is in experiments/, notebooks go in experiments/log/
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "log"
    return generate_notebook(
        name=name,
        description=description,
        sections=sections,
        output_dir=str(output_dir)
    )


def create_experiment_notebook(
    name: str,
    recipe: Optional[str] = None,
    config: Optional[TrainingJobConfig] = None,
    launch: bool = True,
    sections: Optional[List[str]] = None
) -> str:
    """Create a notebook from an experiment recipe or config.
    
    Args:
        name: Name for the notebook
        recipe: Name of a recipe to use (e.g., 'arena')
        config: Custom TrainingJobConfig
        launch: Whether to actually launch the training
        sections: Which notebook sections to include
        
    Returns:
        Path to generated notebook
    """
    if recipe:
        # Import and run the recipe
        if recipe == "arena":
            from experiments.recipes.arena_experiment import ArenaExperiment
            experiment = ArenaExperiment(name)
        else:
            raise ValueError(f"Unknown recipe: {recipe}")
        
        if launch:
            result = experiment.run(generate_notebook=True)
            return result.get("notebook_path", "")
        else:
            # For no-launch, we need to create an empty notebook since no runs were launched
            # Just create a research notebook with the experiment name
            return create_research_notebook(name, description=f"Notebook for {recipe} experiment", sections=sections)
    
    elif config:
        # Launch with custom config
        user = os.environ.get("USER", "unknown")
        date = datetime.now().strftime("%m-%d")
        run_name = f"{user}.notebook.{name}.{date}"
        
        wandb_run_names = []
        skypilot_job_ids = []
        
        if launch:
            print(f"Launching training run: {run_name}")
            result = launch_training_run(
                run_name=run_name,
                curriculum=config.curriculum,
                gpus=config.gpus,
                nodes=config.nodes,
                no_spot=config.no_spot,
                additional_args=config.additional_args,
                wandb_tags=config.wandb_tags,
            )
            
            if result["success"]:
                wandb_run_names = [run_name]
                skypilot_job_ids = [result["job_id"]] if result.get("job_id") else []
                print(f"✓ Launched successfully!")
            else:
                print("✗ Launch failed")
        
        # Generate notebook
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "log"
        return generate_notebook(
            name=name,
            description=f"Notebook for {name}",
            sections=sections,
            wandb_run_names=wandb_run_names if wandb_run_names else None,
            skypilot_job_ids=skypilot_job_ids if skypilot_job_ids else None,
            output_dir=str(output_dir)
        )
    
    else:
        # Just create empty research notebook
        return create_research_notebook(name, sections=sections)


def main():
    """Main entry point for unified notebook creation."""
    parser = argparse.ArgumentParser(
        description="Create notebooks for experiments and research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Create empty research notebook
  ./experiments/new_notebook.py my_research
  
  # Create notebook from arena recipe (launches training)
  ./experiments/new_notebook.py arena_test --recipe arena
  
  # Create notebook from recipe without launching
  ./experiments/new_notebook.py arena_analysis --recipe arena --no-launch
  
  # Create notebook with custom config
  ./experiments/new_notebook.py custom_exp --curriculum env/mettagrid/curriculum/test --gpus 2
  
  # Create minimal notebook
  ./experiments/new_notebook.py quick --sections setup,state,monitor

Available sections: {', '.join(AVAILABLE_SECTIONS.keys())}
        """
    )
    
    parser.add_argument("name", help="Name for the notebook")
    parser.add_argument("-d", "--description", help="Description of the notebook")
    parser.add_argument("-s", "--sections", help="Comma-separated list of sections to include")
    
    # Recipe or custom config
    recipe_group = parser.add_argument_group("recipe options")
    recipe_group.add_argument("-r", "--recipe", choices=["arena"], 
                             help="Use a predefined experiment recipe")
    recipe_group.add_argument("--no-launch", action="store_true",
                             help="Generate notebook without launching training")
    
    # Custom launch config
    config_group = parser.add_argument_group("launch configuration")
    config_group.add_argument("-c", "--curriculum", help="Path to curriculum config")
    config_group.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs")
    config_group.add_argument("-n", "--nodes", type=int, default=1, help="Number of nodes")
    config_group.add_argument("--no-spot", action="store_true", help="Disable spot instances")
    config_group.add_argument("-t", "--tags", nargs="+", help="WandB tags")
    config_group.add_argument("-a", "--args", nargs="+", help="Additional trainer args")
    
    args = parser.parse_args()
    
    # Parse sections
    sections = None
    if args.sections:
        sections = [s.strip() for s in args.sections.split(",")]
        invalid = [s for s in sections if s not in AVAILABLE_SECTIONS]
        if invalid:
            print(f"Error: Invalid sections: {', '.join(invalid)}")
            print(f"Available sections: {', '.join(AVAILABLE_SECTIONS.keys())}")
            return 1
    
    # Create config if custom params provided
    config = None
    if args.curriculum:
        config = TrainingJobConfig(
            curriculum=args.curriculum,
            gpus=args.gpus,
            nodes=args.nodes,
            no_spot=args.no_spot,
            wandb_tags=args.tags,
            additional_args=args.args or []
        )
    
    # Create notebook
    try:
        filepath = create_experiment_notebook(
            name=args.name,
            recipe=args.recipe,
            config=config,
            launch=not args.no_launch and (args.recipe or config),
            sections=sections
        )
        
        print(f"\nNotebook created: {filepath}")
        print(f"\nTo open:")
        print(f"  jupyter notebook {filepath}")
        print(f"  # or")
        print(f"  code {filepath}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())