"""Evaluate missions against thinky agents to validate difficulty estimates.

Runs missions with thinky agents and compares:
- Estimated difficulty (fast/lightweight)
- Actual hearts achieved by agents
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cogames.cogs_vs_clips.difficulty_estimator import DifficultyReport, estimate_difficulty
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.variant_shuffler import VariantCombination

if TYPE_CHECKING:
    pass


@dataclass
class EvalResult:
    """Result of evaluating a single mission."""

    mission_name: str
    variant_names: list[str]

    # Estimated (from lightweight estimator)
    estimated_difficulty: float
    estimated_steps_per_heart: int
    estimated_max_hearts: int | None
    estimated_success_prob: float

    # Actual (from agent run)
    actual_hearts: int = 0
    actual_steps: int = 0
    actual_hearts_per_1k_steps: float = 0.0

    # Comparison
    difficulty_ratio: float = 0.0  # actual vs estimated

    def to_dict(self) -> dict:
        return {
            "mission": self.mission_name,
            "variants": self.variant_names,
            "estimated": {
                "difficulty": self.estimated_difficulty,
                "steps_per_heart": self.estimated_steps_per_heart,
                "max_hearts": self.estimated_max_hearts,
                "success_prob": self.estimated_success_prob,
            },
            "actual": {
                "hearts": self.actual_hearts,
                "steps": self.actual_steps,
                "hearts_per_1k": self.actual_hearts_per_1k_steps,
            },
            "ratio": self.difficulty_ratio,
        }


@dataclass
class EvalSuite:
    """Collection of evaluation results."""

    results: list[EvalResult] = field(default_factory=list)
    total_runs: int = 0
    max_steps: int = 10000

    def add_result(self, result: EvalResult) -> None:
        self.results.append(result)
        self.total_runs += 1

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "DIFFICULTY EVALUATION SUMMARY",
            "=" * 70,
            "",
            f"{'Mission':<30} | {'Est.D':>6} | {'Act.♥':>5} | {'♥/1k':>5} | {'Ratio':>6}",
            "-" * 70,
        ]

        for r in sorted(self.results, key=lambda x: x.estimated_difficulty):
            variants = "+".join(r.variant_names[:2]) if r.variant_names else "base"
            if len(r.variant_names) > 2:
                variants += "..."
            name = f"{r.mission_name[:15]}({variants[:12]})"
            lines.append(
                f"{name:<30} | {r.estimated_difficulty:>6.3f} | {r.actual_hearts:>5} | "
                f"{r.actual_hearts_per_1k_steps:>5.1f} | {r.difficulty_ratio:>6.2f}"
            )

        lines.extend([
            "-" * 70,
            f"Total runs: {self.total_runs}",
        ])

        return "\n".join(lines)

    def to_json(self, path: Path | str) -> None:
        data = {
            "total_runs": self.total_runs,
            "max_steps": self.max_steps,
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def create_eval_result(
    mission: Mission,
    variant_names: list[str],
    report: DifficultyReport,
) -> EvalResult:
    """Create an EvalResult from a mission and difficulty report."""
    return EvalResult(
        mission_name=mission.name,
        variant_names=variant_names,
        estimated_difficulty=report.difficulty_score,
        estimated_steps_per_heart=report.steady_state_steps,
        estimated_max_hearts=report.max_hearts,
        estimated_success_prob=report.success_probability,
    )


THINKY_AGENT_PATH = "cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy"


def run_thinky_evaluation(
    mission: Mission,
    max_steps: int = 10000,
    num_cogs: int = 4,
    seed: int = 42,
) -> tuple[float, int]:
    """Run thinky agents on a mission and return (hearts_per_agent, steps).

    Args:
        mission: Mission to evaluate
        max_steps: Maximum steps per episode
        num_cogs: Number of agents
        seed: Random seed

    Returns:
        Tuple of (hearts_per_agent, total_steps)
    """
    try:
        from mettagrid.policy.loader import initialize_or_load_policy
        from mettagrid.policy.policy import PolicySpec
        from mettagrid.policy.policy_env_interface import PolicyEnvInterface
        from mettagrid.simulator.rollout import Rollout
    except ImportError as e:
        print(f"Warning: Could not import mettagrid components: {e}")
        return 0.0, 0

    try:
        # Create environment config
        env_cfg = mission.make_env()
        env_cfg.game.max_steps = max_steps

        # Create policy and rollout
        pei = PolicyEnvInterface.from_mg_cfg(env_cfg)
        policy = initialize_or_load_policy(
            pei,
            PolicySpec(class_path=THINKY_AGENT_PATH, data_path=None),
        )
        agent_policies = [policy.agent_policy(i) for i in range(num_cogs)]

        rollout = Rollout(
            env_cfg,
            agent_policies,
            render_mode="none",
            seed=seed,
        )
        rollout.run_until_done()

        # Get ACTUAL hearts deposited from game stats (not rewards!)
        # Rewards can be inflated by reward shaping (e.g., HeartChorusVariant)
        stats = rollout._sim.episode_stats
        game_stats = stats.get("game", {})
        actual_hearts = game_stats.get("chest.heart.deposited", 0.0)

        total_steps = rollout._sim.current_step

        return actual_hearts, total_steps

    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        return 0.0, 0


def evaluate_combination(
    base_mission: Mission,
    combo: VariantCombination,
    max_steps: int = 10000,
    num_cogs: int = 4,
    seed: int = 42,
) -> EvalResult:
    """Evaluate a variant combination against thinky agents."""
    # Create mission with variants
    from cogames.cogs_vs_clips.mission import NumCogsVariant

    mission = base_mission.with_variants(combo.variants + [NumCogsVariant(num_cogs=num_cogs)])

    # Get difficulty estimate
    report = combo.difficulty_report or estimate_difficulty(mission)

    # Create result
    result = create_eval_result(mission, combo.variant_names, report)

    # Run thinky agents
    actual_hearts, steps = run_thinky_evaluation(mission, max_steps, num_cogs, seed)

    # Update result with actual performance (actual_hearts is already total, not per-agent)
    result.actual_hearts = int(actual_hearts)
    result.actual_steps = steps
    result.actual_hearts_per_1k_steps = (actual_hearts / steps * 1000) if steps > 0 else 0

    # Calculate ratio: how does actual compare to estimated?
    # Estimated difficulty = steps_per_heart / 1000
    # Actual "difficulty" = 1000 / hearts_per_1k = steps_per_heart
    if result.actual_hearts_per_1k_steps > 0:
        actual_steps_per_heart = 1000 / result.actual_hearts_per_1k_steps
        if result.estimated_steps_per_heart > 0:
            result.difficulty_ratio = actual_steps_per_heart / result.estimated_steps_per_heart
    elif result.estimated_difficulty == float("inf"):
        result.difficulty_ratio = 1.0  # Both impossible

    return result


def run_evaluation_suite(
    base_mission: Mission,
    combinations: list[VariantCombination],
    max_steps: int = 10000,
    num_cogs: int = 4,
    seed: int = 42,
) -> EvalSuite:
    """Run a full evaluation suite."""
    suite = EvalSuite(max_steps=max_steps)

    for i, combo in enumerate(combinations):
        print(f"Evaluating {i+1}/{len(combinations)}: {combo.variant_names or ['base']}")
        result = evaluate_combination(base_mission, combo, max_steps, num_cogs, seed + i)
        suite.add_result(result)
        print(f"  → {result.actual_hearts} hearts in {result.actual_steps} steps")

    return suite


# Quick demo/test function
def demo_evaluation():
    """Demo the evaluation pipeline without running agents."""
    from cogames.cogs_vs_clips.missions import HelloWorldOpenWorldMission
    from cogames.cogs_vs_clips.variant_shuffler import create_difficulty_spectrum, print_combination

    print("=== Difficulty Spectrum Demo ===\n")

    spectrum = create_difficulty_spectrum(
        HelloWorldOpenWorldMission,
        n_per_bucket=2,
        seed=42,
    )

    for bucket_name, combos in spectrum.items():
        print(f"\n{bucket_name.upper()}:")
        for combo in combos:
            print_combination(combo)


if __name__ == "__main__":
    demo_evaluation()

