#!/usr/bin/env -S uv run
"""
Tool to run comprehensive evaluations on policies across all environments.

This script:
1. Takes a list of policy URIs from top_policies_analysis.py
2. Runs sim.py evaluations on all available environments
3. Collects and aggregates performance data
4. Outputs results in formats suitable for factor analysis

Usage:
    ./tools/comprehensive_eval.py ++policy_uris_file=analysis_results/policy_uris.json
    ++output_dir=comprehensive_eval_results ++enable_wandb=true
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import wandb
import yaml

from metta.common.util.logging import setup_mettagrid_logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ComprehensiveEvaluator:
    """Runner for comprehensive policy evaluations across all environments."""

    def __init__(self, output_dir: Path, enable_wandb: bool = False):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_wandb = enable_wandb
        self.wandb_run = None

        # Define individual environments to run (all 80 environments)
        self.individual_environments = [
            # Navigation environments (26)
            "navigation/emptyspace_withinsight",
            "navigation/emptyspace_outofsight",
            "navigation/emptyspace_sparse",
            "navigation/walls_withinsight",
            "navigation/walls_outofsight",
            "navigation/walls_sparse",
            "navigation/cylinder",
            "navigation/obstacles0",
            "navigation/obstacles1",
            "navigation/obstacles2",
            "navigation/obstacles3",
            "navigation/corridors",
            "navigation/labyrinth",
            "navigation/radialmaze",
            "navigation/cylinder_easy",
            "navigation/honeypot",
            "navigation/knotty",
            "navigation/memory_palace",
            "navigation/radial_large",
            "navigation/radial_mini",
            "navigation/radial_small",
            "navigation/swirls",
            "navigation/thecube",
            "navigation/walkaround",
            "navigation/wanderout",
            # Object Use environments (13)
            "object_use/altar_use_free",
            "object_use/armory_use_free",
            "object_use/armory_use",
            "object_use/generator_use_free",
            "object_use/generator_use",
            "object_use/lasery_use_free",
            "object_use/lasery_use",
            "object_use/mine_use",
            "object_use/shoot_out",
            "object_use/swap_in",
            "object_use/swap_out",
            "object_use/temple_use_free",
            "object_use/full_sequence",
            # Memory environments (25)
            "memory/easy",
            "memory/medium",
            "memory/hard",
            "memory/access_cross",
            "memory/boxout",
            "memory/choose_wisely",
            "memory/corners",
            "memory/easy_sequence",
            "memory/hall_of_mirrors",
            "memory/hard_sequence",
            "memory/journey_home",
            "memory/little_landmark_easy",
            "memory/little_landmark_hard",
            "memory/lobster_legs_cues",
            "memory/lobster_legs",
            "memory/medium_sequence",
            "memory/memory_swirls_hard",
            "memory/memory_swirls",
            "memory/passing_things",
            "memory/spacey_memory",
            "memory/tease",
            "memory/venture_out",
            "memory/which_way",
            "memory/tease_small",
            "memory/you_shall_not_pass",
            # Navigation Sequence environments (16)
            "navigation_sequence/cylinder",
            "navigation_sequence/obstacles0",
            "navigation_sequence/obstacles1",
            "navigation_sequence/obstacles2",
            "navigation_sequence/obstacles3",
            "navigation_sequence/corridors",
            "navigation_sequence/cylinder_easy",
            "navigation_sequence/honeypot",
            "navigation_sequence/knotty",
            "navigation_sequence/memory_palace",
            "navigation_sequence/radial_large",
            "navigation_sequence/radial_mini",
            "navigation_sequence/radial_small",
            "navigation_sequence/swirls",
            "navigation_sequence/thecube",
            "navigation_sequence/walkaround",
        ]

        # Keep the original eval_suites for backward compatibility
        self.eval_suites = [
            "navigation",
            "object_use",
            "memory",
            "nav_sequence",
            "all",  # Comprehensive suite with all environments
        ]

    def init_wandb(self, run_name: str) -> None:
        """Initialize wandb run for logging."""
        if self.enable_wandb:
            self.wandb_run = wandb.init(
                project="metta-analysis",
                name=run_name,
                entity="metta-research",
                config={
                    "evaluation_type": "comprehensive_policy_eval",
                    "eval_suites": self.eval_suites,
                },
            )
            self.logger.info(f"Initialized wandb run: {self.wandb_run.name}")

    def log_evaluation_result(self, result: Dict) -> None:
        """Log evaluation result to wandb."""
        if self.enable_wandb and self.wandb_run:
            # Extract key metrics
            metrics = {
                "success": result["success"],
                "execution_time": result["execution_time"],
                "eval_suite": result["eval_suite"],
            }

            # Add performance metrics if available
            if result["success"] and "results" in result:
                eval_results = result["results"]
                if "policies" in eval_results and len(eval_results["policies"]) > 0:
                    policy_result = eval_results["policies"][0]
                    if "reward" in policy_result:
                        metrics["reward"] = policy_result["reward"]
                    if "success_rate" in policy_result:
                        metrics["success_rate"] = policy_result["success_rate"]
                    if "episode_count" in policy_result:
                        metrics["episode_count"] = policy_result["episode_count"]

            # Log with policy identifier
            policy_name = result["policy_uri"].replace("wandb://run/", "").replace("/", "_")
            self.wandb_run.log({f"{policy_name}_{result['eval_suite']}": metrics})

    def finish_wandb(self) -> None:
        """Finish wandb run."""
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.finish()
            self.logger.info("Finished wandb run")

    def load_policy_uris(self, uris_file: Path) -> List[str]:
        """Load policy URIs from file."""
        if uris_file.suffix == ".json":
            with open(uris_file, "r") as f:
                uris = json.load(f)
        else:
            with open(uris_file, "r") as f:
                uris = [line.strip() for line in f if line.strip()]

        # Fix wandb URIs to include artifact type if missing
        fixed_uris = []
        for uri in uris:
            if uri.startswith("wandb://"):
                parts = uri[8:].split("/")
                if len(parts) == 2:
                    # URI format: wandb://entity/name
                    # Need to add project and artifact type: wandb://entity/project/model/name
                    entity, name = parts
                    # Use entity as project name too (common pattern)
                    fixed_uri = f"wandb://{entity}/{entity}/model/{name}"
                    fixed_uris.append(fixed_uri)
                elif len(parts) == 3:
                    # URI format: wandb://entity/project/name
                    # Need to add artifact type: wandb://entity/project/model/name
                    entity, project, name = parts
                    fixed_uri = f"wandb://{entity}/{project}/model/{name}"
                    fixed_uris.append(fixed_uri)
                else:
                    fixed_uris.append(uri)
            else:
                fixed_uris.append(uri)

        return fixed_uris

    def run_single_environment_evaluation(self, policy_uri: str, env_name: str, run_id: str) -> Dict:
        """
        Run evaluation for a single policy on a specific individual environment.

        Args:
            policy_uri: Policy URI to evaluate
            env_name: Individual environment name (e.g., "navigation/cylinder")
            run_id: Unique run identifier

        Returns:
            Dictionary with evaluation results
        """
        # Create unique run name
        policy_name = policy_uri.replace("wandb://run/", "").replace("/", "_")
        run_name = f"{env_name.replace('/', '_')}_{policy_name}_{run_id}"

        # Extract the main suite name for the eval_db_uri
        suite_name = env_name.split("/")[0]

        # Create a temporary config file for this specific environment
        # Create a minimal config that only includes this environment
        temp_config = {
            "defaults": ["sim_suite", "_self_"],
            "name": suite_name,
            "simulations": {env_name: {"env": f"env/mettagrid/{env_name.replace('/', '/evals/')}"}},
        }

        # Write temporary config file in configs directory so Hydra can find it
        temp_config_name = f"temp_{env_name.replace('/', '_')}_{run_id}.yaml"
        temp_config_path = Path("configs/sim") / temp_config_name

        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)

        # Build sim.py command for individual environment
        cmd = [
            "./tools/sim.py",
            f"sim={temp_config_name.replace('.yaml', '')}",
            f"run={run_name}",
            f"policy_uri={policy_uri}",
            f"+eval_db_uri=wandb://stats/{suite_name}_comprehensive_db",
            "seed=42",  # Fixed seed for reproducibility
            "torch_deterministic=True",
            "device=cpu",  # Use CPU for consistency
        ]

        self.logger.info(f"Running evaluation: {' '.join(cmd)}")

        try:
            # Run the evaluation
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=Path.cwd())
            end_time = time.time()

            # Parse JSON output from sim.py
            output_lines = result.stdout.split("\n")
            json_start = None
            json_end = None

            for i, line in enumerate(output_lines):
                if line.strip() == "===JSON_OUTPUT_START===":
                    json_start = i + 1
                elif line.strip() == "===JSON_OUTPUT_END===":
                    json_end = i
                    break

            if json_start is not None and json_end is not None:
                json_str = "\n".join(output_lines[json_start:json_end])
                eval_results = json.loads(json_str)
            else:
                self.logger.warning(f"Could not parse JSON output for {policy_uri}")
                eval_results = {"error": "Could not parse output"}

            return {
                "policy_uri": policy_uri,
                "environment": env_name,
                "eval_suite": suite_name,
                "run_name": run_name,
                "success": True,
                "execution_time": end_time - start_time,
                "results": eval_results,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Evaluation failed for {policy_uri} on {env_name}: {e}")
            return {
                "policy_uri": policy_uri,
                "environment": env_name,
                "eval_suite": suite_name,
                "run_name": run_name,
                "success": False,
                "execution_time": 0,
                "results": {"error": str(e)},
                "stdout": e.stdout,
                "stderr": e.stderr,
            }
        finally:
            # Clean up temporary config file
            try:
                temp_config_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary config file {temp_config_path}: {e}")

    def run_single_evaluation(self, policy_uri: str, eval_suite: str, run_id: str) -> Dict:
        """
        Run evaluation for a single policy on a specific evaluation suite.

        Args:
            policy_uri: Policy URI to evaluate
            eval_suite: Evaluation suite name
            run_id: Unique run identifier

        Returns:
            Dictionary with evaluation results
        """
        # Create unique run name
        policy_name = policy_uri.replace("wandb://run/", "").replace("/", "_")
        run_name = f"{eval_suite}_{policy_name}_{run_id}"

        # Build sim.py command
        cmd = [
            "./tools/sim.py",
            f"sim={eval_suite}",
            f"run={run_name}",
            f"policy_uri={policy_uri}",
            f"+eval_db_uri=wandb://stats/{eval_suite}_comprehensive_db",
            "seed=42",  # Fixed seed for reproducibility
            "torch_deterministic=True",
            "device=cpu",  # Use CPU for consistency
        ]

        self.logger.info(f"Running evaluation: {' '.join(cmd)}")

        try:
            # Run the evaluation
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=Path.cwd())
            end_time = time.time()

            # Parse JSON output from sim.py
            output_lines = result.stdout.split("\n")
            json_start = None
            json_end = None

            for i, line in enumerate(output_lines):
                if line.strip() == "===JSON_OUTPUT_START===":
                    json_start = i + 1
                elif line.strip() == "===JSON_OUTPUT_END===":
                    json_end = i
                    break

            if json_start is not None and json_end is not None:
                json_str = "\n".join(output_lines[json_start:json_end])
                eval_results = json.loads(json_str)
            else:
                self.logger.warning(f"Could not parse JSON output for {policy_uri}")
                eval_results = {"error": "Could not parse output"}

            return {
                "policy_uri": policy_uri,
                "eval_suite": eval_suite,
                "run_name": run_name,
                "success": True,
                "execution_time": end_time - start_time,
                "results": eval_results,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Evaluation failed for {policy_uri} on {eval_suite}: {e}")
            return {
                "policy_uri": policy_uri,
                "eval_suite": eval_suite,
                "run_name": run_name,
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
            }

    def run_comprehensive_evaluation(self, policy_uris: List[str], max_concurrent: int = 1) -> Dict:
        """
        Run comprehensive evaluations for all policies on all evaluation suites.

        Args:
            policy_uris: List of policy URIs to evaluate
            max_concurrent: Maximum number of concurrent evaluations

        Returns:
            Dictionary with all evaluation results
        """
        all_results = {
            "evaluations": [],
            "summary": {
                "total_policies": len(policy_uris),
                "total_evaluations": len(policy_uris) * len(self.individual_environments),
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "total_execution_time": 0,
            },
        }

        run_id = str(int(time.time()))

        # Initialize wandb if enabled
        if self.enable_wandb:
            run_name = f"msb_compEval_{run_id}"
            self.init_wandb(run_name)

        try:
            for i, policy_uri in enumerate(policy_uris):
                self.logger.info(f"Evaluating policy {i + 1}/{len(policy_uris)}: {policy_uri}")

                for env_name in self.individual_environments:
                    self.logger.info(f"  Running {env_name} evaluation...")

                    result = self.run_single_environment_evaluation(policy_uri, env_name, run_id)
                    all_results["evaluations"].append(result)

                    # Log to wandb if enabled
                    self.log_evaluation_result(result)

                    # Update summary
                    if result["success"]:
                        all_results["summary"]["successful_evaluations"] += 1
                        all_results["summary"]["total_execution_time"] += result["execution_time"]
                    else:
                        all_results["summary"]["failed_evaluations"] += 1

                    # Add delay between evaluations to avoid overwhelming the system
                    time.sleep(1)
        finally:
            # Finish wandb run
            self.finish_wandb()

        return all_results

    def extract_performance_data(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Extract performance data from evaluation results for factor analysis.

        Args:
            evaluation_results: Results from run_comprehensive_evaluation

        Returns:
            DataFrame with performance matrix (policies × individual environments)
        """
        performance_data = []

        for eval_result in evaluation_results["evaluations"]:
            if not eval_result["success"]:
                continue

            policy_uri = eval_result["policy_uri"]
            environment = eval_result.get("environment", eval_result["eval_suite"])  # Use environment if available
            eval_suite = eval_result["eval_suite"]
            results = eval_result["results"]

            # Extract metrics from sim.py results
            if "policies" in results and len(results["policies"]) > 0:
                policy_result = results["policies"][0]

                if "checkpoints" in policy_result and len(policy_result["checkpoints"]) > 0:
                    checkpoint = policy_result["checkpoints"][0]

                    # Extract reward average if available
                    reward_avg = checkpoint.get("metrics", {}).get("reward_avg", 0.0)

                    performance_data.append(
                        {
                            "policy_uri": policy_uri,
                            "policy_name": policy_uri.replace("wandb://run/", ""),
                            "environment": environment,
                            "eval_suite": eval_suite,
                            "reward_avg": reward_avg,
                            "success": eval_result["success"],
                        }
                    )

        return pd.DataFrame(performance_data)

    def save_results(self, evaluation_results: Dict, performance_data: pd.DataFrame):
        """Save evaluation results to files."""

        # Save full evaluation results
        results_file = self.output_dir / "comprehensive_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, cls=NumpyEncoder)

        # Save performance data
        performance_file = self.output_dir / "performance_data.csv"
        performance_data.to_csv(performance_file, index=False)

        # Create performance matrix for factor analysis (policies × individual environments)
        if not performance_data.empty:
            performance_matrix = performance_data.pivot_table(
                index="policy_name", columns="environment", values="reward_avg", fill_value=0.0
            )

            matrix_file = self.output_dir / "comprehensive_performance_matrix.csv"
            performance_matrix.to_csv(matrix_file)

            # Save matrix metadata
            matrix_metadata = {
                "shape": performance_matrix.shape,
                "policies": list(performance_matrix.index),
                "environments": list(performance_matrix.columns),
                "missing_values": performance_matrix.isnull().sum().sum(),
                "total_environments": len(self.individual_environments),
                "environment_breakdown": {
                    "navigation": len([e for e in self.individual_environments if e.startswith("navigation/")]),
                    "object_use": len([e for e in self.individual_environments if e.startswith("object_use/")]),
                    "memory": len([e for e in self.individual_environments if e.startswith("memory/")]),
                    "navigation_sequence": len(
                        [e for e in self.individual_environments if e.startswith("navigation_sequence/")]
                    ),
                },
            }

            metadata_file = self.output_dir / "performance_matrix_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(matrix_metadata, f, indent=2, cls=NumpyEncoder)

        # Save summary statistics
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(evaluation_results["summary"], f, indent=2, cls=NumpyEncoder)

        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info(f"  - Full results: {results_file}")
        self.logger.info(f"  - Performance data: {performance_file}")
        if not performance_data.empty:
            self.logger.info(f"  - Performance matrix: {matrix_file}")
            self.logger.info(f"  - Matrix metadata: {metadata_file}")
        self.logger.info(f"  - Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluations for top policies")
    parser.add_argument(
        "--policy-uris-file", type=Path, required=True, help="File containing policy URIs (JSON or text)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comprehensive_eval_results"),
        help="Directory to save evaluation results",
    )
    parser.add_argument("--max-concurrent", type=int, default=1, help="Maximum number of concurrent evaluations")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_mettagrid_logger("comprehensive_eval")
    logger.info("Starting comprehensive evaluation")

    # Create evaluator
    evaluator = ComprehensiveEvaluator(args.output_dir, args.enable_wandb)

    try:
        # Load policy URIs
        policy_uris = evaluator.load_policy_uris(args.policy_uris_file)
        logger.info(f"Loaded {len(policy_uris)} policy URIs")

        if not policy_uris:
            logger.error("No policy URIs found")
            return 1

        # Run comprehensive evaluations
        evaluation_results = evaluator.run_comprehensive_evaluation(policy_uris, args.max_concurrent)

        # Extract performance data
        performance_data = evaluator.extract_performance_data(evaluation_results)

        # Save results
        evaluator.save_results(evaluation_results, performance_data)

        # Print detailed performance arrays for each policy
        print("\n" + "=" * 80)
        print("DETAILED PERFORMANCE ARRAYS FOR EACH POLICY")
        print("=" * 80)

        if not performance_data.empty:
            # Group by policy and show individual environment results
            for policy_uri in performance_data["policy_uri"].unique():
                policy_data = performance_data[performance_data["policy_uri"] == policy_uri]
                policy_name = policy_data["policy_name"].iloc[0]

                print(f"\nPolicy: {policy_name}")
                print(f"URI: {policy_uri}")
                print("-" * 80)

                # Create a dictionary of results by environment
                env_results = {}
                for _, row in policy_data.iterrows():
                    env_results[row["environment"]] = row["reward_avg"]

                # Print results grouped by suite
                suite_names = ["navigation", "object_use", "memory", "navigation_sequence"]
                for suite in suite_names:
                    print(f"\n  {suite.upper()} ENVIRONMENTS:")
                    suite_envs = [e for e in evaluator.individual_environments if e.startswith(f"{suite}/")]
                    for env in suite_envs:
                        reward = env_results.get(env, "FAILED")
                        print(f"    {env:35}: {reward}")

                # Calculate and print statistics
                successful_rewards = [r for r in env_results.values() if isinstance(r, (int, float))]
                if successful_rewards:
                    print("\n  Statistics:")
                    print(f"    Mean reward: {sum(successful_rewards) / len(successful_rewards):.4f}")
                    print(f"    Min reward: {min(successful_rewards):.4f}")
                    print(f"    Max reward: {max(successful_rewards):.4f}")
                    print(
                        f"    Successful environments: "
                        f"{len(successful_rewards)}/{len(evaluator.individual_environments)}"
                    )
                else:
                    print("\n  Statistics: No successful evaluations")

        # Also print summary
        summary = evaluation_results["summary"]
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"  Total policies: {summary['total_policies']}")
        print(f"  Total evaluations: {summary['total_evaluations']}")
        print(f"  Successful: {summary['successful_evaluations']}")
        print(f"  Failed: {summary['failed_evaluations']}")
        print(f"  Total execution time: {summary['total_execution_time']:.1f}s")

        if not performance_data.empty:
            print(
                f"  Performance matrix shape: "
                f"{performance_data.pivot_table(index='policy_name', columns='environment', values='reward_avg').shape}"
            )
            print(f"  Total individual environments: {len(evaluator.individual_environments)}")
            print("  Environment breakdown:")
            for suite in ["navigation", "object_use", "memory", "navigation_sequence"]:
                count = len([e for e in evaluator.individual_environments if e.startswith(f"{suite}/")])
                print(f"    {suite}: {count}")

        return 0

    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
