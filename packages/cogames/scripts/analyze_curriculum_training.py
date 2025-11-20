#!/usr/bin/env -S uv run
"""Analyze curriculum training runs to extract insights for iteration.

This script helps answer:
1. Which missions/variants are agents learning?
2. Where are they getting stuck (hearts but no deposits)?
3. What reward configurations are working?
4. How should we adjust the curriculum buckets?

Usage:
    # Analyze a specific W&B run
    uv run python packages/cogames/scripts/analyze_curriculum_training.py --run-id <wandb_run_id>

    # Analyze multiple runs for comparison
    uv run python packages/cogames/scripts/analyze_curriculum_training.py --run-ids <id1> <id2> <id3>

    # Generate curriculum recommendations
    uv run python packages/cogames/scripts/analyze_curriculum_training.py --run-id <id> --recommend
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MissionDiagnostic:
    """Diagnostic for a single mission/variant."""

    mission_variant: str
    hearts_obtained_rate: float  # % of episodes where agents get hearts
    hearts_deposited_rate: float  # % of episodes where hearts are deposited
    avg_reward: float
    curriculum_level: int  # Which bucket level agents are at
    learning_progress: float  # Rate of improvement
    stuck: bool  # True if hearts obtained >> hearts deposited


@dataclass
class VerificationResult:
    """Results from running verification episodes."""

    mission: str
    episodes_run: int
    hearts_obtained_rate: float
    hearts_deposited_rate: float
    wandb_hearts_obtained_rate: float
    wandb_hearts_deposited_rate: float
    matches_wandb: bool
    discrepancy: float  # Difference between verification and W&B


@dataclass
class CurriculumAnalysis:
    """Overall curriculum analysis."""

    total_episodes: int
    agent_steps: int  # Total training steps
    epochs: int  # Total epochs completed
    missions_mastered: list[str]  # High deposit rates
    missions_stuck: list[MissionDiagnostic]  # Getting hearts but not depositing
    missions_failing: list[str]  # Not even getting hearts
    reward_bucket_utilization: dict[str, list[float]]  # Which buckets are used
    recommended_changes: list[str]
    verification_results: Optional[list[VerificationResult]] = None  # Verification episode results


def verify_policy_behavior(
    run_id: str,
    project: str,
    missions_to_test: list[str],
    num_episodes: int = 10,
) -> list[VerificationResult]:
    """Run verification episodes to verify policy behavior matches W&B logs.

    Uses Rollout class directly to run episodes and extract metrics.

    Args:
        run_id: W&B run ID
        project: W&B project name
        missions_to_test: List of mission names to test
        num_episodes: Number of episodes to run per mission

    Returns:
        List of VerificationResult comparing actual behavior with W&B logs
    """
    try:
        import wandb
    except ImportError as err:
        raise ImportError("wandb not installed. Install with: uv add wandb") from err

    api = wandb.Api(timeout=120)
    run = api.run(f"{project}/{run_id}")

    # Construct S3 checkpoint path from run ID
    # Pattern: s3://softmax-public/policies/{run_id}:latest
    policy_uri = f"s3://softmax-public/policies/{run_id}:latest"

    print(f"\n{'=' * 80}")
    print(f"VERIFICATION: Running {num_episodes} episodes per mission")
    print(f"Policy URI: {policy_uri}")
    print(f"Missions to test: {len(missions_to_test)}")
    print(f"{'=' * 80}\n")

    # Get W&B metrics for comparison
    history = run.history()
    global_hearts_obtained = 0.0
    global_hearts_deposited = 0.0

    heart_gained_col = "env_agent/heart.gained"
    chest_deposited_col = "env_game/chest.heart.deposited"

    if heart_gained_col in history.columns:
        global_hearts_obtained = (history[heart_gained_col] > 0).mean()

    if chest_deposited_col in history.columns:
        global_hearts_deposited = (history[chest_deposited_col] > 0).mean()

    verification_results = []

    # Import here to avoid circular dependencies
    try:
        from cogames.cogs_vs_clips.missions import MISSIONS
        from metta.rl.checkpoint_manager import CheckpointManager
        from mettagrid.policy.loader import initialize_or_load_policy
        from mettagrid.policy.policy_env_interface import PolicyEnvInterface
        from mettagrid.simulator.rollout import Rollout
    except ImportError as e:
        print(f"⚠ Warning: Could not import required modules for verification: {e}")
        print("  Skipping verification - install required packages")
        return []

    for mission in missions_to_test[:5]:  # Limit to 5 missions for testing
        print(f"Testing {mission}...")
        try:
            # Find mission config
            mission_obj = next((m for m in MISSIONS if m.name == mission), None)
            if not mission_obj:
                print(f"  ⚠ {mission}: Mission not found, skipping")
                continue

            env_config = mission_obj.make_env()
            policy_env_info = PolicyEnvInterface.from_mg_cfg(env_config)

            # Load policy
            policy_spec = CheckpointManager.policy_spec_from_uri(policy_uri, device="cpu")
            policy = initialize_or_load_policy(policy_env_info, policy_spec)
            policy.eval()
            agent_policies = [policy.agent_policy(i) for i in range(env_config.game.num_agents)]

            # Run episodes
            episodes_with_hearts = 0
            episodes_with_deposits = 0

            for episode_idx in range(num_episodes):
                rollout = Rollout(
                    env_config,
                    agent_policies,
                    render_mode="none",
                    seed=42 + episode_idx,
                    pass_sim_to_policies=True,
                )
                rollout.run_until_done()

                # Extract metrics from episode_stats
                episode_stats = rollout._sim.episode_stats
                heart_gained = 0.0

                if "agent" in episode_stats:
                    agent_stats_list = episode_stats["agent"]
                    for agent_stats in agent_stats_list:
                        heart_gained += float(agent_stats.get("heart.gained", 0.0))

                # Total reward indicates hearts assembled (deposited)
                # In cogs vs clips, reward = hearts assembled = hearts deposited
                total_reward = float(sum(rollout._sim.episode_rewards))

                # Note: Heart stats may not be in episode_stats during evaluation
                # Use total_reward as proxy (hearts assembled = deposits)

                # Check if hearts were gained
                if heart_gained > 0:
                    episodes_with_hearts += 1

                # Check if hearts were deposited (total_reward > 0 means hearts assembled/deposited)
                if total_reward > 0:
                    episodes_with_deposits += 1

            # Calculate rates
            hearts_obtained_rate = episodes_with_hearts / num_episodes if num_episodes > 0 else 0.0
            hearts_deposited_rate = episodes_with_deposits / num_episodes if num_episodes > 0 else 0.0

            discrepancy = abs(hearts_deposited_rate - global_hearts_deposited)
            matches = discrepancy < 0.2  # Consider match if within 20%

            verification_results.append(
                VerificationResult(
                    mission=mission,
                    episodes_run=num_episodes,
                    hearts_obtained_rate=hearts_obtained_rate,
                    hearts_deposited_rate=hearts_deposited_rate,
                    wandb_hearts_obtained_rate=global_hearts_obtained,
                    wandb_hearts_deposited_rate=global_hearts_deposited,
                    matches_wandb=matches,
                    discrepancy=discrepancy,
                )
            )

            print(f"  ✓ {mission}: {hearts_deposited_rate:.1%} deposits (W&B: {global_hearts_deposited:.1%})")

        except Exception as e:
            print(f"  ✗ {mission}: Error - {e}")
            import traceback

            traceback.print_exc()

    return verification_results


def analyze_wandb_run(run_id: str, project: str = "metta", verify: bool = False) -> CurriculumAnalysis:
    """Analyze a W&B run to extract curriculum insights.

    Args:
        run_id: W&B run ID
        project: W&B project name

    Returns:
        CurriculumAnalysis with recommendations
    """
    try:
        import wandb
    except ImportError as err:
        raise ImportError("wandb not installed. Install with: uv add wandb") from err

    api = wandb.Api(timeout=120)
    run = api.run(f"{project}/{run_id}")

    # Extract metrics from run history
    history = run.history()

    # Extract training progress metrics
    summary = run.summary
    agent_steps = summary.get("metric/agent_step", 0)
    epochs = summary.get("metric/epoch", 0)

    # Analyze per-mission performance
    mission_diagnostics = _analyze_missions(history)

    # Analyze reward bucket utilization
    bucket_utilization = _analyze_reward_buckets(history)

    # Generate recommendations
    recommendations = _generate_recommendations(mission_diagnostics, bucket_utilization)

    # Categorize missions
    mastered = [d.mission_variant for d in mission_diagnostics if d.hearts_deposited_rate > 0.8]
    stuck = [d for d in mission_diagnostics if d.hearts_obtained_rate > 0.5 and d.hearts_deposited_rate < 0.2]
    failing = [d.mission_variant for d in mission_diagnostics if d.hearts_obtained_rate < 0.2]

    # Run verification if requested
    verification_results = None
    if verify:
        # Test a sample of missions (mix of mastered, stuck, and failing)
        missions_to_test = []
        if mastered:
            missions_to_test.extend(mastered[:2])  # 2 mastered
        if stuck:
            missions_to_test.extend([d.mission_variant for d in stuck[:1]])  # 1 stuck
        if failing:
            missions_to_test.extend(failing[:2])  # 2 failing

        if missions_to_test:
            try:
                verification_results = verify_policy_behavior(run_id, project, missions_to_test, num_episodes=5)
            except Exception as e:
                print(f"⚠ Warning: Verification failed: {e}")
                verification_results = []

    return CurriculumAnalysis(
        total_episodes=len(history),
        agent_steps=int(agent_steps),
        epochs=int(epochs),
        missions_mastered=mastered,
        missions_stuck=stuck,
        missions_failing=failing,
        reward_bucket_utilization=bucket_utilization,
        recommended_changes=recommendations,
        verification_results=verification_results,
    )


def _analyze_missions(history: pd.DataFrame) -> list[MissionDiagnostic]:
    """Analyze per-mission performance from W&B history.

    Key metrics to extract:
    - env_per_label_rewards/<mission> - mission-specific rewards
    - env_agent/heart.gained - global hearts obtained
    - env_game/chest.heart.deposited - global hearts deposited
    """
    diagnostics = []

    # Find missions from reward columns (env_per_label_rewards/<mission>)
    mission_reward_cols = [col for col in history.columns if col.startswith("env_per_label_rewards/")]

    missions = set()
    for col in mission_reward_cols:
        # Extract mission name from "env_per_label_rewards/mission_name" or "env_per_label_rewards/mission_name.avg"
        mission = col.split("env_per_label_rewards/")[1].split(".")[0]
        missions.add(mission)

    # Get global heart/chest metrics (fallback if per-label metrics not available)
    heart_gained_col = "env_agent/heart.gained"
    chest_deposited_col = "env_game/chest.heart.deposited"

    # Calculate global rates
    global_hearts_obtained = 0.0
    global_hearts_deposited = 0.0

    if heart_gained_col in history.columns:
        global_hearts_obtained = (history[heart_gained_col] > 0).mean()

    if chest_deposited_col in history.columns:
        global_hearts_deposited = (history[chest_deposited_col] > 0).mean()

    # For each mission, analyze its reward column
    for mission in missions:
        reward_col = f"env_per_label_rewards/{mission}"
        reward_col_avg = f"env_per_label_rewards/{mission}.avg"

        avg_reward = 0.0
        curriculum_level = 0
        learning_progress = 0.0

        # Use .avg column if available, otherwise use base column
        if reward_col_avg in history.columns:
            avg_reward = history[reward_col_avg].mean()
        elif reward_col in history.columns:
            avg_reward = history[reward_col].mean()

        # Try to use per-mission chest deposit metrics if available (new feature)
        # Falls back to global rates if not available (for backward compatibility)
        chest_deposited_per_label_col = f"env_per_label_chest_deposits/{mission}"
        if chest_deposited_per_label_col in history.columns:
            # Use per-mission chest deposit rate
            hearts_deposited = (history[chest_deposited_per_label_col] > 0).mean()
        else:
            # Fall back to global rate (limitation of older runs)
            hearts_deposited = global_hearts_deposited

        # For hearts obtained, we still use global (could be added per-label in future)
        hearts_obtained = global_hearts_obtained

        # Determine if stuck (getting hearts but not depositing)
        stuck = hearts_obtained > 0.5 and hearts_deposited < 0.2

        diagnostics.append(
            MissionDiagnostic(
                mission_variant=mission,
                hearts_obtained_rate=hearts_obtained,
                hearts_deposited_rate=hearts_deposited,
                avg_reward=avg_reward,
                curriculum_level=curriculum_level,
                learning_progress=learning_progress,
                stuck=stuck,
            )
        )

    return diagnostics


def _analyze_reward_buckets(history: pd.DataFrame) -> dict[str, list[float]]:
    """Analyze which reward bucket levels are being used.

    Returns dict mapping bucket name to list of utilization rates per level.
    """
    bucket_utilization = defaultdict(list)

    # Look for curriculum bucket columns
    bucket_cols = [col for col in history.columns if "curriculum/bucket/" in col]

    for col in bucket_cols:
        # Extract bucket name (e.g., "game.agent.rewards.inventory.heart")
        bucket_name = col.split("curriculum/bucket/")[1].split("/")[0]

        # Calculate how often this bucket level is active
        utilization = (history[col] > 0).mean()
        bucket_utilization[bucket_name].append(utilization)

    return dict(bucket_utilization)


def _generate_recommendations(
    diagnostics: list[MissionDiagnostic], bucket_utilization: dict[str, list[float]]
) -> list[str]:
    """Generate curriculum recommendations based on diagnostics."""
    recommendations = []

    # Check for stuck missions (getting hearts but not depositing)
    stuck_missions = [d for d in diagnostics if d.stuck]
    if stuck_missions:
        recommendations.append(
            f"\n## CRITICAL: {len(stuck_missions)} missions where agents get hearts but don't deposit:"
        )
        for d in stuck_missions:
            recommendations.append(
                f"  - {d.mission_variant}: {d.hearts_obtained_rate:.1%} hearts, {d.hearts_deposited_rate:.1%} deposits"
            )
        recommendations.append("\nRecommended fixes:")
        recommendations.append("  1. Add chest.heart.deposited reward bucket: [0.0, 0.1, 0.5, 1.0, 2.0]")
        recommendations.append("  2. Increase chest.heart.amount reward relative to heart inventory reward")
        recommendations.append("  3. Add diagnostic missions that only reward deposit, not obtaining hearts")

    # Check reward bucket utilization
    if "inventory.heart" in bucket_utilization:
        heart_bucket_use = bucket_utilization["inventory.heart"]
        if len(heart_bucket_use) > 0 and max(heart_bucket_use) < 0.1:
            recommendations.append("\n## Heart reward bucket underutilized - consider lowering initial bucket values")

    # Check resource reward buckets
    resource_buckets = [k for k in bucket_utilization.keys() if "stats" in k and "gained" in k]
    for bucket in resource_buckets:
        utilization = bucket_utilization[bucket]
        if len(utilization) > 0 and max(utilization) > 0.8:
            recommendations.append(f"\n## Resource bucket '{bucket}' highly utilized - consider adding higher levels")

    # Check for missions with no learning progress
    no_progress = [d for d in diagnostics if abs(d.learning_progress) < 0.001 and d.curriculum_level == 0]
    if no_progress:
        recommendations.append(f"\n## {len(no_progress)} missions showing no learning progress:")
        for d in no_progress[:5]:  # Show top 5
            recommendations.append(f"  - {d.mission_variant}")
        recommendations.append("  Consider: easier variants, more shaping rewards, or longer max_steps")

    return recommendations


def compare_runs(run_ids: list[str], project: str = "metta") -> pd.DataFrame:
    """Compare multiple curriculum runs side-by-side.

    Returns DataFrame with key metrics for comparison.
    """
    results = []

    for run_id in run_ids:
        analysis = analyze_wandb_run(run_id, project)

        results.append(
            {
                "run_id": run_id,
                "missions_mastered": len(analysis.missions_mastered),
                "missions_stuck": len(analysis.missions_stuck),
                "missions_failing": len(analysis.missions_failing),
                "total_episodes": analysis.total_episodes,
            }
        )

    return pd.DataFrame(results)


def extract_replay_diagnostics(replay_dir: Path) -> dict:
    """Extract behavioral diagnostics from replay files.

    Analyzes agent behavior to determine WHY they're not depositing hearts:
    - Do they navigate to chest?
    - Do they have the default vibe when near chest?
    - Are they attempting deposit action?

    Args:
        replay_dir: Directory containing replay files

    Returns:
        Dictionary of behavioral diagnostics
    """
    diagnostics = {
        "episodes_analyzed": 0,
        "navigated_to_chest": 0,
        "had_hearts_near_chest": 0,
        "attempted_deposit": 0,
        "successful_deposits": 0,
        "common_failure_modes": defaultdict(int),
    }

    # This would need to parse actual replay files
    # For now, return structure showing what we'd extract
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Analyze curriculum training runs")
    parser.add_argument("--run-id", type=str, help="Single W&B run ID to analyze")
    parser.add_argument("--run-ids", nargs="+", help="Multiple run IDs to compare")
    parser.add_argument("--project", type=str, default="metta-research/metta", help="W&B project name")
    parser.add_argument("--recommend", action="store_true", help="Generate curriculum recommendations")
    parser.add_argument("--output", type=str, help="Output file for analysis (JSON)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification episodes to verify behavior matches W&B logs (requires S3 checkpoint path)",
    )

    args = parser.parse_args()

    if args.run_ids:
        # Compare multiple runs
        print("=" * 80)
        print("CURRICULUM RUN COMPARISON")
        print("=" * 80)
        comparison = compare_runs(args.run_ids, args.project)
        print(comparison.to_string(index=False))

    elif args.run_id:
        # Analyze single run
        print("=" * 80)
        print(f"CURRICULUM ANALYSIS: {args.run_id}")
        print("=" * 80)

        analysis = analyze_wandb_run(args.run_id, args.project, verify=args.verify)

        print("\nTraining Progress:")
        print(f"  Total agent steps: {analysis.agent_steps:,}")
        print(f"  Epochs completed: {analysis.epochs:,}")
        if analysis.agent_steps > 0 and analysis.epochs > 0:
            print(f"  Avg steps/epoch: {analysis.agent_steps / analysis.epochs:,.0f}")
        print(f"  Total episode samples logged: {analysis.total_episodes:,}")

        print(f"\n✓ Missions mastered ({len(analysis.missions_mastered)}):")
        for mission in analysis.missions_mastered:
            print(f"  - {mission}")

        print(f"\n⚠ Missions stuck ({len(analysis.missions_stuck)}) - getting hearts but not depositing:")
        for d in analysis.missions_stuck:
            print(
                f"  - {d.mission_variant}: {d.hearts_obtained_rate:.1%} hearts → {d.hearts_deposited_rate:.1%} deposits"
            )

        print(f"\n✗ Missions failing ({len(analysis.missions_failing)}) - not getting hearts:")
        for mission in analysis.missions_failing[:10]:  # Limit to 10
            print(f"  - {mission}")

        if args.verify and analysis.verification_results:
            print("\n" + "=" * 80)
            print("VERIFICATION RESULTS")
            print("=" * 80)
            for v in analysis.verification_results:
                match_status = "✓" if v.matches_wandb else "✗"
                print(f"{match_status} {v.mission}:")
                print(f"  Verification: {v.hearts_deposited_rate:.1%} deposits, {v.hearts_obtained_rate:.1%} hearts")
                print(
                    f"  W&B logs:     {v.wandb_hearts_deposited_rate:.1%} deposits, "
                    f"{v.wandb_hearts_obtained_rate:.1%} hearts"
                )
                print(f"  Discrepancy:  {v.discrepancy:.1%} ({'matches' if v.matches_wandb else 'mismatch'})")
                print()

        if args.recommend and analysis.recommended_changes:
            print("\n" + "=" * 80)
            print("RECOMMENDATIONS")
            print("=" * 80)
            for rec in analysis.recommended_changes:
                print(rec)

        # Save to file if requested
        if args.output:
            output_data = {
                "run_id": args.run_id,
                "agent_steps": analysis.agent_steps,
                "epochs": analysis.epochs,
                "total_episodes": analysis.total_episodes,
                "missions_mastered": analysis.missions_mastered,
                "missions_stuck": [
                    {
                        "mission": d.mission_variant,
                        "hearts_obtained_rate": d.hearts_obtained_rate,
                        "hearts_deposited_rate": d.hearts_deposited_rate,
                        "avg_reward": d.avg_reward,
                    }
                    for d in analysis.missions_stuck
                ],
                "missions_failing": analysis.missions_failing,
                "recommendations": analysis.recommended_changes,
                "verification_results": (
                    [
                        {
                            "mission": v.mission,
                            "episodes_run": v.episodes_run,
                            "hearts_obtained_rate": v.hearts_obtained_rate,
                            "hearts_deposited_rate": v.hearts_deposited_rate,
                            "wandb_hearts_obtained_rate": v.wandb_hearts_obtained_rate,
                            "wandb_hearts_deposited_rate": v.wandb_hearts_deposited_rate,
                            "matches_wandb": v.matches_wandb,
                            "discrepancy": v.discrepancy,
                        }
                        for v in (analysis.verification_results or [])
                    ]
                    if analysis.verification_results
                    else None
                ),
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\n✓ Analysis saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
