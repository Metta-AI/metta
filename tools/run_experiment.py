#!/usr/bin/env -S uv run
"""Run experiments using the AxiomExperiment framework.

This is a variant of run.py that handles Axiom experiments instead of Tools.
Experiments are spec-driven, composable pipeline structures that can include
multiple tools and advanced features like variation points and manifests.

Usage:
    ./tools/run_experiment.py path.to.experiment_factory
    ./tools/run_experiment.py path.to.ExperimentClass --args key=value
    ./tools/run_experiment.py metta.sweep.axiom.train_and_eval:create_experiment --args name=test total_timesteps=10000
"""

import argparse
import inspect
import json
import logging
import os
import signal
import sys
import warnings
from typing import Any, Type, Union, cast

from omegaconf import OmegaConf

from metta.common.util.logging_helpers import init_logging
from metta.utils.module import load_function

# Import these later to avoid circular imports
# They'll be imported when needed

logger = logging.getLogger(__name__)


def init_mettagrid_system_environment() -> None:
    """Initialize environment variables for headless operation."""
    # Set CUDA launch blocking for better error messages in development
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set environment variables to run without display
    os.environ["GLFW_PLATFORM"] = "osmesa"  # Use OSMesa as the GLFW backend
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["DISPLAY"] = ""

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")


def create_experiment(
    make_experiment_path: str, 
    args_conf: dict[str, Any]
) -> "AxiomExperiment":
    """Create an experiment instance from a path and arguments.
    
    Args:
        make_experiment_path: Path to experiment factory function or class
        args_conf: Arguments to pass to the factory
        
    Returns:
        AxiomExperiment instance ready to run
    """
    # Import here to avoid circular imports
    from metta.sweep.axiom.experiment import AxiomExperiment
    from metta.sweep.axiom.experiment_spec import ExperimentSpec
    from metta.sweep.axiom.train_and_eval import TrainAndEvalExperiment, TrainAndEvalSpec
    
    make_experiment = load_function(make_experiment_path)
    
    # Check if it's an ExperimentSpec subclass
    if inspect.isclass(make_experiment):
        if issubclass(make_experiment, ExperimentSpec):
            # Create spec from class
            spec = make_experiment(**args_conf)
            
            # Map spec types to their experiment classes
            if isinstance(spec, TrainAndEvalSpec):
                return TrainAndEvalExperiment(spec)
            else:
                # Generic AxiomExperiment
                return AxiomExperiment(spec)
        elif issubclass(make_experiment, AxiomExperiment):
            # Direct experiment class - needs a spec
            # Try to find a create_spec method or use first arg as spec
            if "spec" in args_conf:
                spec_conf = args_conf.pop("spec")
                if isinstance(spec_conf, dict):
                    # Assume it's a TrainAndEvalSpec for now
                    spec = TrainAndEvalSpec(**spec_conf)
                else:
                    spec = spec_conf
                return make_experiment(spec, **args_conf)
            else:
                raise ValueError(
                    f"Experiment class {make_experiment.__name__} requires a 'spec' argument"
                )
    else:
        # It's a factory function
        # Only pass arguments that the function accepts
        make_experiment_args = {}
        sig = inspect.signature(make_experiment)
        for key in sig.parameters.keys():
            if key in args_conf:
                make_experiment_args[key] = args_conf[key]
        
        result = make_experiment(**make_experiment_args)
        
        # Check what was returned
        if isinstance(result, AxiomExperiment):
            return result
        elif isinstance(result, ExperimentSpec):
            # Create appropriate experiment from spec
            if isinstance(result, TrainAndEvalSpec):
                return TrainAndEvalExperiment(result)
            else:
                return AxiomExperiment(result)
        else:
            raise ValueError(
                f"Factory {make_experiment_path} must return an AxiomExperiment or ExperimentSpec, "
                f"got {type(result)}"
            )


def main():
    """Main entry point for running experiments."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Run Axiom experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "make_experiment_path", 
        type=str, 
        help="Path to the experiment factory function or class"
    )
    parser.add_argument(
        "--args", 
        nargs="*",
        help="Arguments to pass to the factory (key=value format)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Tag for this experiment run"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        default=False,
        help="Print experiment spec without running"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Run comparison mode with baseline and variants"
    )
    parser.add_argument(
        "--variants",
        type=str,
        help="JSON file with variant configurations for comparison mode"
    )
    args = parser.parse_args()

    # Initialize environment
    init_logging()
    init_mettagrid_system_environment()

    # Exit on ctrl+c
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

    # Parse arguments
    make_experiment_args = args.args or []
    assert isinstance(make_experiment_args, list)

    args_conf = OmegaConf.to_container(OmegaConf.from_cli(make_experiment_args))
    assert isinstance(args_conf, dict)
    args_conf = cast(dict[str, Any], args_conf)

    # Create the experiment
    try:
        experiment = create_experiment(args.make_experiment_path, args_conf)
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        sys.exit(1)

    # Handle dry run
    if args.dry_run:
        print("Dry run: printing experiment spec")
        print(experiment.spec.model_dump_json(indent=2))
        sys.exit(0)

    # Seed random number generators using experiment's control settings
    from metta.rl.system_config import SystemConfig, seed_everything
    
    if hasattr(experiment.spec, 'system_config'):
        seed_everything(experiment.spec.system_config)
    else:
        # Use default seeding
        seed_everything(SystemConfig())

    # Prepare experiment (seeds, determinism, etc.)
    experiment.prepare()

    # Run experiment
    try:
        if args.compare and args.variants:
            # Comparison mode - run baseline and variants
            with open(args.variants, 'r') as f:
                variants = json.load(f)
            
            logger.info(f"Running comparison with {len(variants)} variants")
            results = experiment.run_comparison(
                baseline_joins=variants.get("baseline"),
                variants={k: v for k, v in variants.items() if k != "baseline"}
            )
            
            # Print comparison results
            print("\n=== Comparison Results ===")
            for name, handle in results.items():
                manifest = handle.manifest()
                print(f"\n{name}:")
                print(f"  Status: {manifest.get('pipeline_result', {}).get('status', 'unknown')}")
                if 'metrics' in manifest:
                    print(f"  Metrics: {manifest['metrics']}")
        else:
            # Single run mode
            logger.info(f"Running experiment: {experiment.spec.name} (tag: {args.tag})")
            handle = experiment.run(tag=args.tag)
            
            # Get and print results
            manifest = handle.manifest()
            
            print("\n=== Experiment Complete ===")
            print(f"Name: {experiment.spec.name}")
            print(f"Tag: {args.tag}")
            print(f"Status: {manifest.get('pipeline_result', {}).get('status', 'unknown')}")
            
            # Print key metrics if available
            pipeline_result = manifest.get('pipeline_result', {})
            if 'eval_results' in pipeline_result:
                eval_results = pipeline_result['eval_results']
                if hasattr(eval_results, 'scores'):
                    print(f"\nEvaluation Results:")
                    print(f"  Average reward: {eval_results.scores.avg_simulation_score:.2f}")
                    print(f"  Category score: {eval_results.scores.avg_category_score:.2f}")
            
            # Save manifest path
            print(f"\nManifest saved to: {handle.manifest_path}")
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()