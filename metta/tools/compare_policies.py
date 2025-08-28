#!/usr/bin/env python3
"""Simple policy comparison tool using stats database."""

import argparse
import json
import logging
from pathlib import Path

from metta.eval.eval_stats_db import EvalStatsDB

logger = logging.getLogger(__name__)


def compare_policies(stats_db_path: str, policy_specs: list[str], metric: str = "score") -> dict:
    """Compare multiple policies using evaluation stats database.

    Policy specs format: "checkpoint_path:epoch" or just "checkpoint_path" (uses epoch 0)
    """
    comparison_results = {"metric": metric, "policies": []}

    try:
        with EvalStatsDB.from_uri(stats_db_path) as stats_db:
            for policy_spec in policy_specs:
                # Parse policy specification
                if ":" in policy_spec:
                    checkpoint_path, epoch_str = policy_spec.rsplit(":", 1)
                    epoch = int(epoch_str)
                else:
                    checkpoint_path = policy_spec
                    epoch = 0

                # Get policy performance from stats database
                policy_scores = stats_db.simulation_scores(checkpoint_path, epoch, metric)

                # Calculate overall metrics
                total_samples = stats_db.sample_count(checkpoint_path, epoch)
                avg_score = sum(policy_scores.values()) / len(policy_scores) if policy_scores else 0.0

                comparison_results["policies"].append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "epoch": epoch,
                        "scores_by_simulation": dict(policy_scores),
                        "average_score": avg_score,
                        "total_samples": total_samples,
                    }
                )

                logger.info(f"Policy {Path(checkpoint_path).stem}:{epoch} - Average {metric}: {avg_score:.3f}")

    except Exception as e:
        logger.error(f"Policy comparison failed: {e}")
        comparison_results["error"] = str(e)

    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Compare policy performance using stats database")
    parser.add_argument("--stats-db", required=True, help="Path to evaluation stats database")
    parser.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help="Policy specs in format 'checkpoint_path:epoch' or 'checkpoint_path'",
    )
    parser.add_argument("--metric", default="score", help="Metric to compare (default: score)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Compare policies
    results = compare_policies(args.stats_db, args.policies, args.metric)

    # Output results
    output_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Comparison results written to {args.output}")
    else:
        print(output_json)

    # Summary to stderr for easy parsing
    if "error" not in results:
        policies = results["policies"]
        if policies:
            best_policy = max(policies, key=lambda p: p["average_score"])
            print(
                f"\nBest performing policy: {Path(best_policy['checkpoint_path']).stem}:{best_policy['epoch']} "
                f"(avg {args.metric}: {best_policy['average_score']:.3f})",
                file=__import__("sys").stderr,
            )


if __name__ == "__main__":
    main()
