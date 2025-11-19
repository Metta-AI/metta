import ast
import logging
from typing import Any, Sequence, Union, cast

import numpy as np
from rich.console import Console
from rich.table import Table

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.runner import SimulationRunConfig, run_simulations
from metta.tools.multi_policy_eval import MultiPolicyEvalTool
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger(__name__)


class LeaderboardEvalTool(MultiPolicyEvalTool):
    policy_display_names: Sequence[str] | None = None

    def invoke(self, args: dict[str, str]) -> int | None:
        console = Console()

        # Run simulations
        simulation_results = run_simulations(
            policy_specs=self.policy_specs,
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
        )

        # Process results
        # We want to track stats per candidate policy
        # The policy_specs are [Candidate1, Candidate2, ..., Random, Baseline]

        category_names = [
            "Self-Play",
            "With-Random",
            "With-Baseline",
            "With-Thinky",
            "With-Ladybug",
        ]

        results_by_policy = {}

        for sim_result in simulation_results:
            sim_cfg = sim_result.run
            rollout_result = sim_result.results

            # Determine which policies were active in this simulation
            # proportions tells us the mix
            proportions = sim_cfg.proportions
            if proportions is None:
                continue
            active_indices = [i for i, p in enumerate(proportions) if p > 0]

            # Extract stats
            chest_hearts_per_episode = []
            total_hearts_per_episode = []
            for episode in rollout_result.episodes:
                stats = episode.stats
                game_stats = stats.get("game", {})
                chest_hearts = float(game_stats.get("chest.heart.amount", 0.0))

                inventory_hearts = 0.0
                for agent_stats in stats.get("agent", []):
                    inventory_hearts += float(agent_stats.get("inventory.heart", 0.0))

                total_hearts = chest_hearts + inventory_hearts

                chest_hearts_per_episode.append(chest_hearts)
                total_hearts_per_episode.append(total_hearts)

            mean_chest_hearts = np.mean(chest_hearts_per_episode) if chest_hearts_per_episode else 0.0
            mean_total_hearts = np.mean(total_hearts_per_episode) if total_hearts_per_episode else 0.0

            # Attribute this result to the candidate policy in the mix
            for idx in active_indices:
                # Get display name if available, else use spec name
                if self.policy_display_names and idx < len(self.policy_display_names):
                    policy_name = self.policy_display_names[idx]
                else:
                    policy_name = self.policy_specs[idx].name

                if policy_name in ["Random", "Baseline", "Thinky", "Ladybug"]:
                    # If we are benchmarking baselines themselves (no candidates provided), keep them
                    # Otherwise skip them to focus on candidates
                    pass

                # This is a candidate
                if policy_name not in results_by_policy:
                    results_by_policy[policy_name] = {
                        category: {"chest": [], "total": []} for category in category_names
                    }

                # Determine category based on the OTHER active agents
                other_indices = [i for i in active_indices if i != idx]
                if not other_indices:
                    category = "Self-Play"
                else:
                    # Get other policy name
                    if self.policy_display_names and other_indices[0] < len(self.policy_display_names):
                        other_name = self.policy_display_names[other_indices[0]]
                    else:
                        other_name = self.policy_specs[other_indices[0]].name

                    if other_name == "Random":
                        category = "With-Random"
                    elif other_name == "Baseline":
                        category = "With-Baseline"
                    elif other_name == "Thinky":
                        category = "With-Thinky"
                    elif other_name == "Ladybug":
                        category = "With-Ladybug"
                    else:
                        category = "Other"

                results_by_policy[policy_name][category]["chest"].append(mean_chest_hearts)
                results_by_policy[policy_name][category]["total"].append(mean_total_hearts)

        # Filter out baselines from display if we have actual candidates
        # Identify actual candidates (policies that are not in the baseline list)
        baseline_names = ["Random", "Baseline", "Thinky", "Ladybug"]
        candidates = [p for p in results_by_policy.keys() if p not in baseline_names]

        if candidates:
            # If we have candidates, remove baselines from the results table
            for baseline in baseline_names:
                if baseline in results_by_policy:
                    del results_by_policy[baseline]

        # Display Leaderboard (Chest Hearts)
        table = Table(title="Leaderboard Score (Machina 1 Open World)")
        table.add_column("Policy", justify="left", style="cyan", no_wrap=True)
        table.add_column("Score (Avg Hearts)", justify="right", style="magenta")
        table.add_column("Self-Play", justify="right")
        table.add_column("With Random", justify="right")
        table.add_column("With Baseline", justify="right")
        table.add_column("With Thinky", justify="right")
        table.add_column("With Ladybug", justify="right")

        leaderboard_rows: list[tuple[str, float]] = []

        def summarize_metrics(
            category_data: dict[str, dict[str, list[float]]], metric_key: str
        ) -> tuple[float, dict[str, float]]:
            component_scores: dict[str, float] = {}
            valid_scores = []
            for category in category_names:
                values = category_data[category][metric_key]
                score = float(np.mean(values)) if values else 0.0
                component_scores[category] = score
                if values:
                    valid_scores.append(score)
            aggregate_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            return aggregate_score, component_scores

        for policy_name, categories in results_by_policy.items():
            agg_score, components = summarize_metrics(categories, "chest")
            leaderboard_rows.append((policy_name, agg_score))

            table.add_row(
                policy_name,
                f"{agg_score:.2f}",
                f"{components['Self-Play']:.2f}",
                f"{components['With-Random']:.2f}",
                f"{components['With-Baseline']:.2f}",
                f"{components['With-Thinky']:.2f}",
                f"{components['With-Ladybug']:.2f}",
            )

        console.print(table)

        if leaderboard_rows:
            console.print("\n[bold cyan]Final Ranking (Chest Hearts)[/bold cyan]")
            ranked = sorted(leaderboard_rows, key=lambda x: x[1], reverse=True)
            for idx, (policy_name, score) in enumerate(ranked, start=1):
                console.print(f"{idx}. {policy_name}: {score:.2f}")

        # Secondary leaderboard for total hearts (chest + inventory)
        total_table = Table(title="Total Hearts (Chest + Inventory)")
        total_table.add_column("Policy", justify="left", style="cyan", no_wrap=True)
        total_table.add_column("Score (Avg Hearts)", justify="right", style="magenta")
        total_table.add_column("Self-Play", justify="right")
        total_table.add_column("With Random", justify="right")
        total_table.add_column("With Baseline", justify="right")
        total_table.add_column("With Thinky", justify="right")
        total_table.add_column("With Ladybug", justify="right")

        total_rows: list[tuple[str, float]] = []
        for policy_name, categories in results_by_policy.items():
            agg_score, components = summarize_metrics(categories, "total")
            total_rows.append((policy_name, agg_score))
            total_table.add_row(
                policy_name,
                f"{agg_score:.2f}",
                f"{components['Self-Play']:.2f}",
                f"{components['With-Random']:.2f}",
                f"{components['With-Baseline']:.2f}",
                f"{components['With-Thinky']:.2f}",
                f"{components['With-Ladybug']:.2f}",
            )

        console.print(total_table)

        if total_rows:
            console.print("\n[bold cyan]Final Ranking (Total Hearts)[/bold cyan]")
            ranked_total = sorted(total_rows, key=lambda x: x[1], reverse=True)
            for idx, (policy_name, score) in enumerate(ranked_total, start=1):
                console.print(f"{idx}. {policy_name}: {score:.2f}")

        return 0


def run(
    policy_uris: Union[str, Sequence[str]] = [],
    num_cogs: int = 4,
    num_episodes_per_config: int = 1,
    seed: int = 42,
) -> LeaderboardEvalTool:
    """
    Run a leaderboard evaluation for the provided policies.

    Args:
        policy_uris: Single URI or list of URIs for policies to evaluate.
        num_cogs: Number of cogs in the game (default 4).
        num_episodes_per_config: Number of episodes to run for each configuration (default 1).
        seed: Random seed for reproducibility (default 42).
    """
    if isinstance(policy_uris, str):
        # Handle string input that might be a list representation
        if policy_uris.startswith("[") and policy_uris.endswith("]"):
            try:
                policy_uris = ast.literal_eval(policy_uris)
            except (ValueError, SyntaxError):
                policy_uris = [policy_uris]
        else:
            policy_uris = [policy_uris]

    # If policy_uris is None or empty list (default), treat as empty list
    if not policy_uris:
        policy_uris = []

    # Setup Policy Specs
    policy_specs = []
    display_names = []

    # 1. Candidate Policies
    for i, uri in enumerate(policy_uris):
        # Use a readable name if possible, else index
        name = f"Candidate_{i + 1}"
        # Try to extract a name from URI if it looks like a file path or wandb run
        if "://" in uri:
            name = uri.split("/")[-1].replace(".mpt", "").replace(":", "_")

        spec = CheckpointManager.policy_spec_from_uri(uri, device="cpu")
        policy_specs.append(spec)
        display_names.append(name)

    # 2. Baseline Policies
    random_spec = PolicySpec(
        class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy",
        data_path=None,
    )
    baseline_spec = PolicySpec(
        class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        data_path=None,
    )
    thinky_spec = PolicySpec(
        class_path="cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
        data_path=None,
    )
    ladybug_spec = PolicySpec(
        class_path="cogames.policy.nim_agents.agents.LadyBugAgentsMultiPolicy",
        data_path=None,
    )
    policy_specs.extend([random_spec, baseline_spec, thinky_spec, ladybug_spec])
    display_names.extend(["Random", "Baseline", "Thinky", "Ladybug"])

    random_idx = len(policy_specs) - 4
    baseline_idx = len(policy_specs) - 3
    thinky_idx = len(policy_specs) - 2
    ladybug_idx = len(policy_specs) - 1

    # If no candidate policies provided, treat baselines as candidates for comparison
    # When policy_uris is empty, candidate_indices will range over the baselines
    candidate_indices = range(len(policy_uris)) if policy_uris else range(len(policy_specs))

    # Setup Environment
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env_config = mission.make_env()
    map_builder = getattr(env_config.game, "map_builder", None)
    if map_builder is not None and hasattr(map_builder, "seed"):
        cast(Any, map_builder).seed = seed

    # Setup Simulations
    simulations = []

    # For each candidate, create the 5 scenarios
    for i in candidate_indices:
        # Candidate index in policy_specs is i
        candidate_name = display_names[i]

        # Scenario A: Self-Play (All agents are Candidate i)
        props_sp = [0.0] * len(policy_specs)
        props_sp[i] = 1.0

        simulations.append(
            SimulationRunConfig(
                env=env_config,
                num_episodes=num_episodes_per_config,
                proportions=props_sp,
                episode_tags={"type": "self_play", "candidate": candidate_name},
            )
        )

        # Scenario B: With Random (Mix of Candidate i and Random)
        props_rand = [0.0] * len(policy_specs)
        props_rand[i] = 1.0
        props_rand[random_idx] = 1.0

        simulations.append(
            SimulationRunConfig(
                env=env_config,
                num_episodes=num_episodes_per_config,
                proportions=props_rand,
                episode_tags={"type": "with_random", "candidate": candidate_name},
            )
        )

        # Scenario C: With Baseline (Mix of Candidate i and Baseline)
        props_base = [0.0] * len(policy_specs)
        props_base[i] = 1.0
        props_base[baseline_idx] = 1.0

        simulations.append(
            SimulationRunConfig(
                env=env_config,
                num_episodes=num_episodes_per_config,
                proportions=props_base,
                episode_tags={"type": "with_baseline", "candidate": candidate_name},
            )
        )

        # Scenario D: With Thinky
        props_thinky = [0.0] * len(policy_specs)
        props_thinky[i] = 1.0
        props_thinky[thinky_idx] = 1.0

        simulations.append(
            SimulationRunConfig(
                env=env_config,
                num_episodes=num_episodes_per_config,
                proportions=props_thinky,
                episode_tags={"type": "with_thinky", "candidate": candidate_name},
            )
        )

        # Scenario E: With Ladybug
        props_ladybug = [0.0] * len(policy_specs)
        props_ladybug[i] = 1.0
        props_ladybug[ladybug_idx] = 1.0

        simulations.append(
            SimulationRunConfig(
                env=env_config,
                num_episodes=num_episodes_per_config,
                proportions=props_ladybug,
                episode_tags={"type": "with_ladybug", "candidate": candidate_name},
            )
        )

    # Create the tool with the specified seed in the system config
    tool = LeaderboardEvalTool(
        policy_specs=policy_specs,
        simulations=simulations,
        policy_display_names=display_names,
    )
    # Override the system seed if provided
    if seed is not None:
        tool.system.seed = seed

    return tool
