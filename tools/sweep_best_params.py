#!/usr/bin/env -S uv run
"""Find the best hyperparameters from a sweep and generate config patches."""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf

from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.common.wandb.sweep import sweep_id_from_name

logger = setup_mettagrid_logger("sweep_best_params")


def convert_to_yaml(data: Union[str, Dict[str, Any]]) -> str:
    """Convert a dict or string representation of dict to YAML format.

    Args:
        data: Either a dictionary or a string representation of a dictionary

    Returns:
        YAML formatted string
    """
    # If it's a string that looks like a dict, parse it first
    if isinstance(data, str):
        data = data.strip()
        if data.startswith("{") and data.endswith("}"):
            try:
                data = ast.literal_eval(data)
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse string as dict: {e}")
                return data

    # Convert to YAML
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def get_sweep_runs(sweep_id: str, entity: str, project: str) -> List[Any]:
    """Get all runs from a sweep sorted by score."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Get all runs and filter for successful ones
    runs = []
    for run in sweep.runs:
        if run.summary.get("protein.state") == "success":
            score = run.summary.get("score", run.summary.get("protein.objective", 0))
            if score is not None and score > 0:  # Filter out failed runs
                runs.append(run)

    # Sort by score (descending for reward metric)
    runs.sort(key=lambda r: r.summary.get("score", r.summary.get("protein.objective", 0)), reverse=True)
    return runs


def extract_hyperparameters_from_run(run: Any) -> Dict[str, Any]:
    """Extract the hyperparameters used in a run."""
    # Try to get from protein.suggestion first (most reliable)
    suggestion = run.summary.get("protein.suggestion", {})
    logger.debug(f"Raw protein.suggestion: type={type(suggestion)}, value={suggestion}")

    # If suggestion is a string (dict representation), try to parse it
    if isinstance(suggestion, str):
        try:
            import ast

            suggestion = ast.literal_eval(suggestion)
            logger.debug(f"Parsed suggestion from string: {suggestion}")
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse protein.suggestion string: {suggestion}")
            suggestion = {}

    # If we have a suggestion, use it
    if suggestion:
        params = suggestion
    else:
        # Fallback: extract trainer config from run config
        config_dict = dict(run.config)
        # We want to extract the trainer parameters which contain the hyperparameters
        params = {}
        if "trainer" in config_dict:
            params["trainer"] = config_dict["trainer"]

    # Recursively parse any string dict values
    def parse_string_dicts(obj: Any) -> Any:
        # Check if it's dict-like (including wandb SummarySubDict)
        if hasattr(obj, "items") and hasattr(obj, "keys"):
            result = {}
            for k, v in obj.items():
                if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                    try:
                        import ast

                        result[k] = parse_string_dicts(ast.literal_eval(v))
                    except (ValueError, SyntaxError):
                        result[k] = v
                else:
                    result[k] = parse_string_dicts(v)
            return result
        elif isinstance(obj, list):
            return [parse_string_dicts(item) for item in obj]
        else:
            return obj

    parsed = parse_string_dicts(params)

    # Ensure we return a dict
    return parsed if isinstance(parsed, dict) else {}


def load_local_hyperparameters(sweep_run: str, run_id: str, data_dir: str) -> Dict[str, Any]:
    """Load hyperparameters from local train_config_overrides.yaml."""
    run_dir = Path(data_dir) / "sweep" / sweep_run / "runs" / run_id
    override_path = run_dir / "train_config_overrides.yaml"

    if override_path.exists():
        cfg = OmegaConf.load(override_path)
        container = OmegaConf.to_container(cfg)
        # Extract just the trainer parameters
        if isinstance(container, dict) and "trainer" in container:
            return {"trainer": container["trainer"]}
        elif isinstance(container, dict):
            # Ensure all keys are strings
            return {str(k): v for k, v in container.items()}
        else:
            return {}
    else:
        logger.warning(f"Local override file not found: {override_path}")
        return {}


def format_hyperparameters_yaml(params: Dict[str, Any], indent: int = 0) -> str:
    """Format hyperparameters as YAML string."""
    lines = []
    indent_str = "  " * indent

    for key, value in params.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(format_hyperparameters_yaml(value, indent + 1))
        else:
            # Format numbers nicely
            if isinstance(value, float):
                if value < 0.01:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.6f}".rstrip("0").rstrip(".")
            else:
                formatted_value = str(value)
            lines.append(f"{indent_str}{key}: {formatted_value}")

    return "\n".join(lines)


def generate_config_patch(params: Dict[str, Any], output_path: str) -> None:
    """Generate a config patch file."""
    with open(output_path, "w") as f:
        f.write("# @package _global_\n")
        f.write("# Best hyperparameters from sweep\n")
        f.write("# Apply with: +trainer/patch=<filename_without_yaml>\n\n")
        # Use custom formatting for better readability
        yaml_content = format_hyperparameters_yaml(params)
        f.write(yaml_content)
    logger.info(f"Config patch saved to: {output_path}")


def generate_override_args(params: Dict[str, Any], prefix: str = "") -> List[str]:
    """Generate command-line override arguments."""
    args = []

    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            args.extend(generate_override_args(value, full_key))
        else:
            if isinstance(value, float):
                if value < 0.01:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.6f}".rstrip("0").rstrip(".")
            else:
                formatted_value = str(value)
            args.append(f"{full_key}={formatted_value}")

    return args


def parse_args():
    """Parse command-line arguments before Hydra takes over."""
    parser = argparse.ArgumentParser(
        description="Find best hyperparameters from a sweep",
        add_help=False,  # We'll handle help ourselves
    )
    parser.add_argument("--top-n", type=int, default=1, help="Number of top runs to analyze")
    parser.add_argument("--show-scores", action="store_true", help="Show scores for all runs")
    parser.add_argument("--no-generate-patch", action="store_true", help="Skip generating config patch file")
    parser.add_argument("--output-dir", default="configs/trainer/patch", help="Directory for patch files")
    parser.add_argument("--help", "-h", action="store_true", help="Show this help message and exit")

    # Parse only our known arguments, leave the rest for Hydra
    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        print("\nHydra arguments:")
        print("  sweep_run=<name>     Name of the sweep to analyze (required)")
        sys.exit(0)

    return args, remaining


# Global variable to store parsed args
PARSED_ARGS: Optional[argparse.Namespace] = None


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    global PARSED_ARGS
    args = PARSED_ARGS
    assert args is not None, "Arguments should have been parsed before main()"

    # The sweep name comes from cfg.sweep_run (set via command line: sweep_run=axel_remote_test_1)
    if not hasattr(cfg, "sweep_run") or not cfg.sweep_run:
        logger.error("No sweep_run specified. Use: sweep_run=<sweep_name>")
        return 1

    # Create output directory if generating patches
    if not args.no_generate_patch:
        os.makedirs(args.output_dir, exist_ok=True)

    # Always load from WandB
    logger.info(f"Loading best parameters from WandB sweep: {cfg.sweep_run}")

    # Get sweep ID
    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.wandb.entity, cfg.sweep_run)
    if not sweep_id:
        logger.error(f"Sweep not found: {cfg.sweep_run}")
        return 1

    # Get runs
    runs = get_sweep_runs(sweep_id, cfg.wandb.entity, cfg.wandb.project)

    if args.show_scores:
        logger.info("\nAll runs with scores:")
        for run in runs[:20]:  # Show top 20
            score = run.summary.get("score", run.summary.get("protein.objective", 0))
            logger.info(f"  {run.name}: {score:.4f}")

    if not runs:
        logger.error("No successful runs found in sweep")
        return 1

    # Get best run (first one since they're sorted by score)
    best_run = runs[0]
    score = best_run.summary.get("score", best_run.summary.get("protein.objective", 0))
    logger.info(f"\nBest run: {best_run.name} (score: {score:.4f})")

    # Extract hyperparameters from best run
    params = extract_hyperparameters_from_run(best_run)

    if not params:
        # If protein.suggestion is empty, try to extract from config
        logger.warning("No parameters found in protein.suggestion, checking run config...")
        if "trainer" in best_run.config:
            params = {"trainer": dict(best_run.config["trainer"])}

    if not params:
        logger.error("No hyperparameters found")
        return 1

    # Generate different output formats
    logger.info("\n" + "=" * 60)
    logger.info("BEST HYPERPARAMETERS:")
    logger.info("=" * 60)

    # 1. Display as YAML
    logger.info("\n1. As YAML config:")
    logger.info("-" * 40)
    print(format_hyperparameters_yaml(params))

    # 2. Generate config patch file (if requested)
    if not args.no_generate_patch:
        patch_name = f"{cfg.sweep_run}_best.yaml"
        patch_path = os.path.join(args.output_dir, patch_name)
        generate_config_patch(params, patch_path)

        logger.info(f"\n2. Config patch saved to: {patch_path}")
        logger.info(f"   Use with: +trainer/patch={patch_name[:-5]}")

    # 3. Generate command-line overrides
    override_args = generate_override_args(params)
    logger.info("\n3. As command-line overrides:")
    logger.info("-" * 40)
    logger.info("./devops/train.sh " + " ".join(override_args))

    # 4. Generate a complete training command
    logger.info("\n4. Complete training command:")
    logger.info("-" * 40)
    logger.info(f"./devops/train.sh run={cfg.sweep_run}_best " + " ".join(override_args))

    # 5. If multiple top runs requested, show comparison
    if args.top_n > 1:
        logger.info(f"\n5. Comparison of top {args.top_n} runs:")
        logger.info("-" * 40)

        # Get top N runs
        best_runs = runs[: args.top_n]

        # Collect all parameter keys
        all_params = {}
        for run in best_runs:
            run_params = extract_hyperparameters_from_run(run)
            for key, value in run_params.items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)

        # Show parameters that vary
        for key, values in all_params.items():
            if len(set(str(v) for v in values)) > 1:  # Parameters that differ
                logger.info(f"{key}:")
                for i, run in enumerate(best_runs):
                    run_params = extract_hyperparameters_from_run(run)
                    value = run_params.get(key, "N/A")
                    score = run.summary.get("score", run.summary.get("protein.objective", 0))
                    logger.info(f"  Run {i + 1} (score: {score:.4f}): {value}")

    return 0


if __name__ == "__main__":
    # Parse our custom arguments first
    PARSED_ARGS, remaining_args = parse_args()

    # Update sys.argv to only include remaining arguments for Hydra
    sys.argv = [sys.argv[0]] + remaining_args

    sys.exit(main())
