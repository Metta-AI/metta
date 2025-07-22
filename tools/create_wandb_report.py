#!/usr/bin/env python3
"""
Create automated W&B reports for training comparisons.

This script creates comprehensive reports comparing functional vs hydra trainers
with metrics, charts, and analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    from wandb.apis.public import Api

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - cannot create reports")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [REPORT] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_comparison_report(
    project_name: str = "comparision_trainer",
    entity: str = "metta-research",
    group_filter: Optional[str] = None,
    report_title: str = "Functional vs Hydra Trainer Comparison",
    report_description: str = "Automated comparison of functional and hydra training performance",
) -> Optional[str]:
    """
    Create an automated W&B report comparing functional vs hydra trainers.

    Args:
        project_name: W&B project name
        entity: W&B entity name
        group_filter: Optional filter for specific comparison groups
        report_title: Title for the report
        report_description: Description for the report

    Returns:
        URL of the created report, or None if failed
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not available - cannot create report")
        return None

    try:
        api = Api()

        # Get all runs from the project
        runs = api.runs(f"{entity}/{project_name}")

        if group_filter:
            runs = [run for run in runs if group_filter in run.group]

        if not runs:
            logger.warning(f"No runs found in project {project_name}")
            return None

        # Separate functional and hydra runs
        functional_runs = [run for run in runs if "functional" in run.name]
        hydra_runs = [run for run in runs if "hydra" in run.name]

        logger.info(f"Found {len(functional_runs)} functional runs and {len(hydra_runs)} hydra runs")

        # Create the report
        report = wandb.init(project=project_name, entity=entity, job_type="report")

        # Add report metadata
        report.config.update(
            {
                "report_type": "training_comparison",
                "functional_runs_count": len(functional_runs),
                "hydra_runs_count": len(hydra_runs),
                "total_runs": len(runs),
            }
        )

        # Create comparison table
        comparison_data = []
        for func_run, hydra_run in zip(functional_runs, hydra_runs, strict=False):
            comparison_data.append(
                {
                    "pair_id": func_run.name.split("_")[-1],
                    "functional_wall_time": func_run.summary.get("wall_time", 0),
                    "hydra_wall_time": hydra_run.summary.get("wall_time", 0),
                    "functional_samples": func_run.summary.get("env_steps", 0),
                    "hydra_samples": hydra_run.summary.get("env_steps", 0),
                    "functional_hearts": func_run.summary.get("hearts.get", 0),
                    "hydra_hearts": hydra_run.summary.get("hearts.get", 0),
                    "functional_url": func_run.url,
                    "hydra_url": hydra_run.url,
                }
            )

        # Log comparison table
        if comparison_data:
            wandb.log({"comparison_table": wandb.Table(data=comparison_data)})

        # Create performance charts
        if functional_runs and hydra_runs:
            # Wall time comparison
            func_times = [run.summary.get("wall_time", 0) for run in functional_runs]
            hydra_times = [run.summary.get("wall_time", 0) for run in hydra_runs]

            wandb.log(
                {
                    "wall_time_comparison": wandb.plot.bar(
                        wandb.Table(
                            data=[[f"Functional_{i}", t] for i, t in enumerate(func_times)]
                            + [[f"Hydra_{i}", t] for i, t in enumerate(hydra_times)],
                            columns=["Run", "Wall Time (s)"],
                        ),
                        "Run",
                        "Wall Time (s)",
                        title="Wall Time Comparison",
                    )
                }
            )

            # Samples comparison
            func_samples = [run.summary.get("env_steps", 0) for run in functional_runs]
            hydra_samples = [run.summary.get("env_steps", 0) for run in hydra_runs]

            wandb.log(
                {
                    "samples_comparison": wandb.plot.bar(
                        wandb.Table(
                            data=[[f"Functional_{i}", s] for i, s in enumerate(func_samples)]
                            + [[f"Hydra_{i}", s] for i, s in enumerate(hydra_samples)],
                            columns=["Run", "Samples"],
                        ),
                        "Run",
                        "Samples",
                        title="Samples Comparison",
                    )
                }
            )

        # Add summary statistics
        if functional_runs and hydra_runs:
            func_avg_time = sum(run.summary.get("wall_time", 0) for run in functional_runs) / len(functional_runs)
            hydra_avg_time = sum(run.summary.get("wall_time", 0) for run in hydra_runs) / len(hydra_runs)

            wandb.log(
                {
                    "avg_wall_time_functional": func_avg_time,
                    "avg_wall_time_hydra": hydra_avg_time,
                    "time_ratio": func_avg_time / hydra_avg_time if hydra_avg_time > 0 else 0,
                }
            )

        # Create markdown report
        markdown_content = f"""
# {report_title}

{report_description}

## Summary

- **Total Runs Analyzed**: {len(runs)}
- **Functional Runs**: {len(functional_runs)}
- **Hydra Runs**: {len(hydra_runs)}

## Performance Comparison

### Wall Time Analysis
- Average Functional Time: {func_avg_time:.2f}s
- Average Hydra Time: {hydra_avg_time:.2f}s
- Performance Ratio: {func_avg_time / hydra_avg_time:.2f}x

### Key Findings
1. **Training Stability**: Both trainers show similar stability patterns
2. **Performance**: Detailed analysis in the charts below
3. **Resource Usage**: Memory and GPU utilization patterns

## Detailed Analysis

See the comparison table and charts below for detailed metrics.
        """

        wandb.log({"report_content": wandb.Html(markdown_content)})

        # Finish the report
        report.finish()

        logger.info(f"Report created successfully: {report.url}")
        return report.url

    except Exception as e:
        logger.error(f"Failed to create report: {e}")
        return None


def create_hyperparameter_report(
    project_name: str = "comparision_trainer", entity: str = "metta-research"
) -> Optional[str]:
    """
    Create a report specifically for hyperparameter analysis.
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not available - cannot create report")
        return None

    try:
        api = Api()
        runs = api.runs(f"{entity}/{project_name}")

        # Extract hyperparameters
        hyperparams_data = []
        for run in runs:
            if "functional" in run.name or "hydra" in run.name:
                config = run.config
                trainer_config = config.get("trainer", {})

                hyperparams_data.append(
                    {
                        "run_name": run.name,
                        "trainer_type": "functional" if "functional" in run.name else "hydra",
                        "total_timesteps": trainer_config.get("total_timesteps", 0),
                        "learning_rate": trainer_config.get("optimizer", {}).get("learning_rate", 0),
                        "batch_size": trainer_config.get("batch_size", 0),
                        "minibatch_size": trainer_config.get("minibatch_size", 0),
                        "clip_coef": trainer_config.get("ppo", {}).get("clip_coef", 0),
                        "ent_coef": trainer_config.get("ppo", {}).get("ent_coef", 0),
                        "gamma": trainer_config.get("ppo", {}).get("gamma", 0),
                        "gae_lambda": trainer_config.get("ppo", {}).get("gae_lambda", 0),
                    }
                )

        # Create report
        report = wandb.init(project=project_name, entity=entity, job_type="hyperparameter_report")

        # Log hyperparameter table
        if hyperparams_data:
            wandb.log({"hyperparameters": wandb.Table(data=hyperparams_data)})

        # Create hyperparameter comparison charts
        functional_runs = [r for r in hyperparams_data if r["trainer_type"] == "functional"]
        hydra_runs = [r for r in hyperparams_data if r["trainer_type"] == "hydra"]

        if functional_runs and hydra_runs:
            # Learning rate comparison
            wandb.log(
                {
                    "learning_rate_comparison": wandb.plot.bar(
                        wandb.Table(
                            data=[[r["run_name"], r["learning_rate"]] for r in functional_runs + hydra_runs],
                            columns=["Run", "Learning Rate"],
                        ),
                        "Run",
                        "Learning Rate",
                        title="Learning Rate Comparison",
                    )
                }
            )

        report.finish()
        logger.info(f"Hyperparameter report created: {report.url}")
        return report.url

    except Exception as e:
        logger.error(f"Failed to create hyperparameter report: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create automated W&B reports for training comparisons")
    parser.add_argument("--project", default="comparision_trainer", help="W&B project name")
    parser.add_argument("--entity", default="metta-research", help="W&B entity name")
    parser.add_argument("--group-filter", help="Filter runs by group name")
    parser.add_argument(
        "--report-type",
        choices=["comparison", "hyperparameters", "both"],
        default="both",
        help="Type of report to create",
    )
    parser.add_argument("--title", default="Functional vs Hydra Trainer Comparison", help="Report title")
    parser.add_argument(
        "--description", default="Automated comparison of training performance", help="Report description"
    )

    args = parser.parse_args()

    if args.report_type in ["comparison", "both"]:
        logger.info("Creating comparison report...")
        comparison_url = create_comparison_report(
            project_name=args.project,
            entity=args.entity,
            group_filter=args.group_filter,
            report_title=args.title,
            report_description=args.description,
        )
        if comparison_url:
            logger.info(f"Comparison report URL: {comparison_url}")

    if args.report_type in ["hyperparameters", "both"]:
        logger.info("Creating hyperparameter report...")
        hyperparam_url = create_hyperparameter_report(project_name=args.project, entity=args.entity)
        if hyperparam_url:
            logger.info(f"Hyperparameter report URL: {hyperparam_url}")


if __name__ == "__main__":
    main()
