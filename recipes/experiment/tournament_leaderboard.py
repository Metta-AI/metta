"""Tournament-style leaderboard evaluation for policies.

Evaluates a policy against a pool of diverse agents (scripted and learned) across
many random team compositions. Returns a single aggregate "Value Above Replacement"
score based on mean hearts across all games.

The tournament works as follows:
1. Creates a policy pool with the input policy + 4 scripted baselines
2. Generates N random 4-agent games where each agent uses a unique policy
3. The input policy participates in every game (always included)
4. Other 3 slots are randomly sampled from the remaining pool
5. All games use machina_1 map with pack_rat + heart_chorus + lonely_hearts variants
6. Final score is mean total hearts (chest + agent inventories) across all games

Note: Policies are loaded with strict=False, allowing policies trained on different
action spaces to participate. Mismatched weights (e.g., different action heads) are
skipped during loading, while shared weights (e.g., observation encoders) are reused.

Usage:
    # Evaluate a policy with default 30 games
    ./tools/run.py recipes.experiment.tournament_leaderboard.run \\
        policy_uri='s3://softmax-public/policies/local.nishadsingh.20251114.124019/local.nishadsingh.20251114.124019:v74.mpt'

    # Run more games for better statistics
    ./tools/run.py recipes.experiment.tournament_leaderboard.run \\
        policy_uri='file://./train_dir/my_run/checkpoints/:latest' \\
        num_games=50

    # Quick test with fewer games
    ./tools/run.py recipes.experiment.tournament_leaderboard.run \\
        policy_uri='s3://...' \\
        num_games=5
"""

import logging
import random
from typing import Sequence

from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.runner import SimulationRunConfig, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.policy.policy import PolicySpec

try:
    from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
    from cogames.cogs_vs_clips.variants import (
        HeartChorusVariant,
        LonelyHeartVariant,
        PackRatVariant,
    )
except ImportError:
    # Fallback to direct import if cogames not installed as package
    import sys
    from pathlib import Path

    cogames_path = Path(__file__).parent.parent.parent / "packages" / "cogames" / "src"
    if cogames_path.exists():
        sys.path.insert(0, str(cogames_path))
    from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
    from cogames.cogs_vs_clips.variants import (
        HeartChorusVariant,
        LonelyHeartVariant,
        PackRatVariant,
    )

logger = logging.getLogger(__name__)


def run(
    policy_uri: str,
    num_games: int = 30,
    seed: int = 42,
    replay_dir: str | None = None,
) -> dict[str, float]:
    """Run tournament evaluation and return leaderboard scores for all policies.

    Args:
        policy_uri: URI of policy to evaluate (always included in every game)
        num_games: Number of games to run with random team compositions
        seed: Random seed for reproducibility
        replay_dir: Optional directory to save replays

    Returns:
        Dict mapping policy names to their leaderboard scores (mean hearts across all games)
    """
    if replay_dir is None:
        replay_dir = auto_replay_dir()

    # Define policy pool
    policy_specs = _create_policy_pool(policy_uri)
    logger.info(f"Created policy pool with {len(policy_specs)} policies")
    for i, spec in enumerate(policy_specs):
        logger.info(f"  [{i}] {spec.name}")

    # Create environment with all variants applied
    mission = Machina1OpenWorldMission.with_variants(
        [
            PackRatVariant(),
            HeartChorusVariant(),
            LonelyHeartVariant(),
        ]
    )
    mission.num_cogs = 4
    env = mission.make_env()
    logger.info("Created machina_1 environment with pack_rat, heart_chorus, lonely_hearts variants")

    # Generate random team compositions
    # Input policy (index 0) is in every game, other 3 slots sampled from pool
    simulations = _generate_random_teams(
        env=env,
        num_games=num_games,
        num_policies=len(policy_specs),
        seed=seed,
    )
    logger.info(f"Generated {len(simulations)} random team compositions")

    # Run all simulations
    logger.info("Running tournament simulations...")
    simulation_results = run_simulations(
        policy_specs=policy_specs,
        simulations=simulations,
        replay_dir=replay_dir,
        seed=seed,
    )

    # Compute leaderboard scores for all policies
    all_scores = _compute_all_leaderboard_scores(
        simulation_results, policy_specs=policy_specs, num_policies=len(policy_specs)
    )

    # Display leaderboard
    _display_leaderboard(all_scores)

    return all_scores


def _create_policy_pool(input_policy_uri: str) -> Sequence[PolicySpec]:
    """Create pool of diverse policies for tournament.

    Pool includes:
    - Input policy (being evaluated)
    - 4 scripted baselines (ladybug, thinky, race_car, random)
    Note: RL policies from S3 are currently disabled due to action space mismatches

    Returns:
        List of policy specs with input policy first (index 0)
    """
    # Input policy (being evaluated) - always first
    # Check if it's a class path (scripted policy) or a checkpoint URI
    if input_policy_uri.startswith(("s3://", "file://", "/", "./")):
        # It's a checkpoint URI
        input_spec = CheckpointManager.policy_spec_from_uri(input_policy_uri, device="cpu", strict=False)
    else:
        # It's a class path (scripted policy)
        input_spec = PolicySpec(class_path=input_policy_uri, data_path=None)

    # Scripted baselines - diverse behaviors
    scripted_specs = [
        PolicySpec(
            class_path="cogames.policy.nim_agents.agents.LadybugAgentsMultiPolicy",
            data_path=None,
        ),
        PolicySpec(
            class_path="cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
            data_path=None,
        ),
        PolicySpec(
            class_path="cogames.policy.nim_agents.agents.RaceCarAgentsMultiPolicy",
            data_path=None,
        ),
        PolicySpec(
            class_path="cogames.policy.nim_agents.agents.RandomAgentsMultiPolicy",
            data_path=None,
        ),
    ]

    # Trained RL policies from S3 (examples from codebase)
    # NOTE: Commented out for now as these policies are trained on different action spaces
    # TODO: Find policies trained on machina_1 or similar environments
    rl_specs = []
    # try:
    #     # Policy 1: Teacher policy from ABES kickstart experiments
    #     # Use strict=False to allow loading policies trained on different action spaces
    #     rl_spec_1 = CheckpointManager.policy_spec_from_uri(
    #         "s3://softmax-public/policies/av.teach.24checks.11.10.10/av.teach.24checks.11.10.10:v8016.mpt",
    #         device="cpu",
    #         strict=False,
    #     )
    #     rl_specs.append(rl_spec_1)
    #     logger.info(f"Added RL policy: {rl_spec_1.name}")
    # except Exception as e:
    #     logger.warning(f"Failed to load RL teacher policy: {e}")

    # try:
    #     # Policy 2: ICL resource chain policy from assembly lines
    #     # Use strict=False to allow loading policies trained on different action spaces
    #     rl_spec_2 = CheckpointManager.policy_spec_from_uri(
    #         "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt",
    #         device="cpu",
    #         strict=False,
    #     )
    #     rl_specs.append(rl_spec_2)
    #     logger.info(f"Added RL policy: {rl_spec_2.name}")
    # except Exception as e:
    #     logger.warning(f"Failed to load RL ICL policy: {e}")

    return [input_spec] + scripted_specs + rl_specs


def _generate_random_teams(
    env,
    num_games: int,
    num_policies: int,
    seed: int,
) -> list[SimulationRunConfig]:
    """Generate random 4-policy team compositions.

    Input policy (index 0) is in every game. Other 3 slots randomly sampled
    from remaining policies without replacement.

    Args:
        env: Environment configuration
        num_games: Number of games to generate
        num_policies: Total number of policies in pool
        seed: Random seed

    Returns:
        List of simulation configs, each with a unique 4-policy team
    """
    if num_policies < 4:
        raise ValueError(f"Need at least 4 policies in pool, got {num_policies}")

    rng = random.Random(seed)
    simulations = []

    other_policy_indices = list(range(1, num_policies))

    for game_idx in range(num_games):
        # Input policy + 3 random others (sample up to min(3, available))
        num_to_sample = min(3, len(other_policy_indices))
        team_indices = [0] + rng.sample(other_policy_indices, num_to_sample)

        # Create proportions: only the 4 selected policies participate
        proportions = [1 if i in team_indices else 0 for i in range(num_policies)]

        simulations.append(
            SimulationRunConfig(
                env=env,
                num_episodes=1,
                proportions=proportions,
                episode_tags={
                    "category": "tournament",
                    "name": f"game_{game_idx:03d}",
                    "team": ",".join(str(i) for i in sorted(team_indices)),
                },
            )
        )

    return simulations


def _compute_all_leaderboard_scores(
    simulation_results,
    policy_specs: Sequence[PolicySpec],
    num_policies: int,
) -> dict[str, float]:
    """Compute mean hearts metric for all policies in the pool.

    Hearts = chest hearts + sum of agent inventory hearts at episode end.

    Args:
        simulation_results: Results from run_simulations
        policy_specs: List of all policy specs in the pool
        num_policies: Total number of policies

    Returns:
        Dict mapping policy names to mean total hearts across their games
    """
    # Track hearts per episode for each policy
    policy_hearts: dict[int, list[float]] = {i: [] for i in range(num_policies)}

    for sim_result in simulation_results:
        for episode in sim_result.results.episodes:
            # Extract total hearts for this episode
            total_hearts = _extract_hearts_from_episode(episode)

            # Record this score for all policies that participated
            for policy_idx in range(num_policies):
                if policy_idx in episode.assignments:
                    policy_hearts[policy_idx].append(total_hearts)

    # Compute mean for each policy
    scores = {}
    for policy_idx, hearts_list in policy_hearts.items():
        if hearts_list:
            mean_hearts = sum(hearts_list) / len(hearts_list)
            scores[policy_specs[policy_idx].name] = mean_hearts
            logger.info(
                f"Policy {policy_specs[policy_idx].name}: {mean_hearts:.2f} hearts "
                f"(participated in {len(hearts_list)} games)"
            )
        else:
            # Policy didn't participate in any games
            scores[policy_specs[policy_idx].name] = 0.0
            logger.warning(f"Policy {policy_specs[policy_idx].name} didn't participate in any games!")

    return scores


def _display_leaderboard(scores: dict[str, float]) -> None:
    """Display leaderboard in a formatted table.

    Args:
        scores: Dict mapping policy names to their scores
    """
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"\n{'=' * 70}")
    logger.info("TOURNAMENT LEADERBOARD - Mean Total Hearts")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Rank':<6} {'Policy':<35} {'Score':<10}")
    logger.info(f"{'-' * 70}")

    for rank, (policy_name, score) in enumerate(sorted_scores, start=1):
        # Highlight the input policy
        marker = ">>> " if policy_name == "input_policy" else "    "
        logger.info(f"{marker}{rank:<2} {policy_name:<35} {score:>10.2f}")

    logger.info(f"{'=' * 70}")
    logger.info(f"Input policy score: {scores.get('input_policy', 0.0):.2f} hearts")
    logger.info(f"Best score: {sorted_scores[0][1]:.2f} hearts ({sorted_scores[0][0]})")
    logger.info(f"{'=' * 70}\n")


def _extract_hearts_from_episode(episode) -> float:
    """Extract total hearts (chest + all agent inventories) from episode stats.

    Args:
        episode: EpisodeRolloutResult

    Returns:
        Total hearts at episode end
    """
    # Try to get chest hearts from game stats
    chest_hearts = 0.0
    if "game" in episode.stats:
        chest_hearts = episode.stats["game"].get("chest.heart.amount", 0.0)

    # Sum hearts from all agent inventories
    agent_hearts = 0.0
    if "agent" in episode.stats:
        for agent_stats in episode.stats["agent"]:
            # Look for inventory.heart or similar stat
            agent_hearts += agent_stats.get("inventory.heart", 0.0)

    total_hearts = chest_hearts + agent_hearts
    return float(total_hearts)
