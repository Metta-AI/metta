#!/usr/bin/env -S uv run
"""
Master pipeline for analyzing top performing policies from observatory database.

This pipeline orchestrates the complete analysis workflow:
1. Extract top N policies from observatory database via API
2. Run comprehensive evaluations on all environments using sim.py
3. Perform factor analysis with EM and K-fold cross-validation
4. Generate visualizations and policy clusters

Usage:
    ./tools/policy_analysis_pipeline.py ++num_policies=100 ++skip_stage=extract
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from metta.common.util.logging import setup_mettagrid_logger


class PolicyAnalysisPipeline:
    """Master pipeline for policy analysis workflow."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_mettagrid_logger("policy_analysis_pipeline")
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(self, stage_name: str, skip_stages: List[str]) -> bool:
        """Run a pipeline stage if not in skip list."""
        if stage_name in skip_stages:
            self.logger.info(f"Skipping stage: {stage_name}")
            return True

        self.logger.info(f"Running stage: {stage_name}")

        try:
            if stage_name == "extract":
                return self._run_extract_stage()
            elif stage_name == "evaluate":
                return self._run_evaluate_stage()
            elif stage_name == "analyze":
                return self._run_analyze_stage()
            else:
                self.logger.error(f"Unknown stage: {stage_name}")
                return False
        except Exception as e:
            self.logger.error(f"Stage {stage_name} failed: {e}")
            return False

    def _run_extract_stage(self) -> bool:
        """Run the policy extraction stage using observatory API."""
        self.logger.info("Extracting top policies from observatory database...")

        cmd = [
            sys.executable,
            "tools/observatory_policy_analysis.py",
            "--num-policies",
            str(self.config["num_policies"]),
            "--output-dir",
            str(self.output_dir / "extracted_data"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Extract stage failed: {result.stderr}")
            return False

        self.logger.info("Extract stage completed successfully")
        return True

    def _run_evaluate_stage(self) -> bool:
        """Run the comprehensive evaluation stage."""
        self.logger.info("Running comprehensive evaluations...")

        # Load extracted policies
        policies_file = self.output_dir / "extracted_data" / "top_policies.json"
        if not policies_file.exists():
            self.logger.error(f"Policies file not found: {policies_file}")
            return False

        with open(policies_file, "r") as f:
            policies = json.load(f)

        # Create evaluation configuration
        eval_config = {
            "policies": policies,
            "environments": self.config.get("environments", "all"),
            "num_episodes": self.config.get("num_episodes", 10),
            "output_dir": str(self.output_dir / "evaluations"),
        }

        eval_config_file = self.output_dir / "eval_config.json"
        with open(eval_config_file, "w") as f:
            json.dump(eval_config, f, indent=2)

            # Run comprehensive evaluation
        cmd = [
            sys.executable,
            "tools/comprehensive_eval.py",
            "--policy-uris-file",
            str(self.output_dir / "extracted_data" / "policy_uris.json"),
            "--output-dir",
            str(self.output_dir / "evaluations"),
        ]

        # Add wandb flag if enabled
        if self.config.get("enable_wandb", False):
            cmd.append("--enable-wandb")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Evaluate stage failed: {result.stderr}")
            return False

        self.logger.info("Evaluate stage completed successfully")
        return True

    def _run_analyze_stage(self) -> bool:
        """Run the factor analysis stage."""
        self.logger.info("Running factor analysis...")

        # Check for evaluation results
        eval_results_file = self.output_dir / "evaluations" / "evaluation_results.json"
        if not eval_results_file.exists():
            self.logger.error(f"Evaluation results not found: {eval_results_file}")
            return False

        # Create analysis configuration
        analysis_config = {
            "input_file": str(eval_results_file),
            "output_dir": str(self.output_dir / "analysis"),
            "max_components": self.config.get("max_components", 20),
            "k_folds": self.config.get("k_folds", 5),
            "random_state": self.config.get("random_state", 42),
        }

        analysis_config_file = self.output_dir / "analysis_config.json"
        with open(analysis_config_file, "w") as f:
            json.dump(analysis_config, f, indent=2)

            # Run factor analysis
        cmd = [
            sys.executable,
            "tools/factor_analysis.py",
            "--performance-matrix",
            str(self.output_dir / "evaluations" / "comprehensive_performance_matrix.csv"),
            "--output-dir",
            str(self.output_dir / "analysis"),
        ]

        # Add wandb flag if enabled
        if self.config.get("enable_wandb", False):
            cmd.append("--enable-wandb")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Analyze stage failed: {result.stderr}")
            return False

        self.logger.info("Analyze stage completed successfully")
        return True

    def run_pipeline(self, skip_stages: List[str]) -> bool:
        """Run the complete pipeline."""
        self.logger.info("Starting policy analysis pipeline")

        stages = ["extract", "evaluate", "analyze"]

        for stage in stages:
            if not self.run_stage(stage, skip_stages):
                self.logger.error(f"Pipeline failed at stage: {stage}")
                return False

        self.logger.info("Pipeline completed successfully!")
        return True

    def generate_report(self) -> None:
        """Generate a summary report of the analysis."""
        report_file = self.output_dir / "analysis_report.md"

        with open(report_file, "w") as f:
            f.write("# Policy Analysis Pipeline Report\n\n")

            # Add configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- Number of policies: {self.config['num_policies']}\n")
            f.write(f"- Output directory: {self.config['output_dir']}\n")
            f.write(f"- Max components: {self.config.get('max_components', 20)}\n")
            f.write(f"- K-folds: {self.config.get('k_folds', 5)}\n\n")

            # Add file structure
            f.write("## Output Files\n\n")
            f.write("```\n")
            f.write(f"{self.output_dir}/\n")
            f.write("├── extracted_data/\n")
            f.write("│   ├── top_policies.csv\n")
            f.write("│   ├── top_policies.json\n")
            f.write("│   ├── policy_evaluations.csv\n")
            f.write("│   └── summary_stats.json\n")
            f.write("├── evaluations/\n")
            f.write("│   ├── evaluation_results.json\n")
            f.write("│   └── performance_matrix.csv\n")
            f.write("└── analysis/\n")
            f.write("    ├── factor_analysis_results.json\n")
            f.write("    ├── optimal_components.json\n")
            f.write("    ├── policy_clusters.csv\n")
            f.write("    └── visualizations/\n")
            f.write("        ├── scree_plot.png\n")
            f.write("        ├── factor_loadings.png\n")
            f.write("        └── policy_clusters.png\n")
            f.write("```\n\n")

            # Add usage instructions
            f.write("## Usage Instructions\n\n")
            f.write("1. **Extracted Data**: Contains top policies and their metadata\n")
            f.write("2. **Evaluations**: Comprehensive performance data across all environments\n")
            f.write("3. **Analysis**: Factor analysis results, optimal dimensionality, and visualizations\n\n")

            f.write("## Next Steps\n\n")
            f.write("- Review the factor analysis results to understand policy performance dimensions\n")
            f.write("- Use the policy clusters for targeted analysis or training\n")
            f.write("- Explore the visualizations to identify patterns in policy behavior\n")

        self.logger.info(f"Report generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Master pipeline for policy analysis")
    parser.add_argument("--num-policies", type=int, default=100, help="Number of top policies to analyze")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("policy_analysis_results"), help="Output directory for all results"
    )
    parser.add_argument(
        "--skip-stage",
        action="append",
        choices=["extract", "evaluate", "analyze"],
        help="Stages to skip (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-components", type=int, default=20, help="Maximum number of components for factor analysis"
    )
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--environments", default="all", help="Environments to evaluate (default: all)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per policy-environment pair")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable wandb logging to metta-analysis project")

    args = parser.parse_args()

    # Setup configuration
    config = {
        "num_policies": args.num_policies,
        "output_dir": str(args.output_dir),
        "max_components": args.max_components,
        "k_folds": args.k_folds,
        "random_state": args.random_state,
        "environments": args.environments,
        "num_episodes": args.num_episodes,
        "enable_wandb": args.enable_wandb,
    }

    skip_stages = args.skip_stage or []

    # Create and run pipeline
    pipeline = PolicyAnalysisPipeline(config)

    if pipeline.run_pipeline(skip_stages):
        pipeline.generate_report()
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📄 Report generated: {args.output_dir}/analysis_report.md")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
