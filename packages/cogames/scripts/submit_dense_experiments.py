#!/usr/bin/env -S uv run
"""Submit dense curriculum experiments.

Based on analysis recommendations (see run_analysis/DENSE_CURRICULUM_RECOMMENDATIONS.md), this script submits:

1. Dense resource levels [7,8,9,10] experiments:
   - Easier maps, with diagnostics
   - Easier maps, without diagnostics
   - Difficult maps, with diagnostics
   - Difficult maps, without diagnostics

2. Full range [1-10] experiments:
   - Easier maps, with diagnostics
   - Easier maps, without diagnostics
   - Difficult maps, with diagnostics
   - Difficult maps, without diagnostics

Key improvements:
- Chest deposit reward buckets: [1.5, 2.0, 2.5, 3.0]
- Adjusted inventory rewards: [0.1, 0.2, 0.3, 0.5]
- Max_uses bucketing: [1, 2, 3, 5, 10, 255] instead of single-use
- Scaled max_steps by resource level
- Map selection (easier vs difficult)
- Diagnostic mission inclusion/exclusion
"""

from recipes.experiment.cvc import dense_curriculum

# Map groupings
EASIER_MAPS = ["dense_training_4agents", "dense_training_4agentsbase"]
DIFFICULT_MAPS = ["dense_training_big", "dense_training_small"]


def submit_dense_experiments():
    """Submit dense curriculum experiments with different configurations."""
    print("=" * 80)
    print("Submitting Dense Curriculum Experiments")
    print("=" * 80)
    print("\nExperiments test:")
    print("  - Resource level ranges: [7,8,9,10] (dense) vs [1-10] (full range)")
    print("  - Map sets: Easier (4agents, 4agentsbase) vs Difficult (big, small)")
    print("  - Diagnostic missions: Included vs Excluded")
    print("\nKey improvements:")
    print("  - Chest deposit reward buckets: [1.5, 2.0, 2.5, 3.0]")
    print("  - Max_uses bucketing: [1, 2, 3, 5, 10, 255]")
    print("  - Scaled max_steps by resource level")
    print("  - Adjusted reward buckets to match successful curricula")

    experiments = []

    # Dense resource levels [7,8,9,10]
    for map_set_name, maps in [("easier", EASIER_MAPS), ("difficult", DIFFICULT_MAPS)]:
        for include_diag in [True, False]:
            diag_suffix = "diag" if include_diag else "nodiag"
            run_name = f"dense_rlvl7-10_maps_{map_set_name}_{diag_suffix}_v2"
            experiments.append(
                {
                    "run_name": run_name,
                    "resource_levels": [7, 8, 9, 10],
                    "maps_to_use": maps,
                    "include_diagnostics": include_diag,
                }
            )

    # Full range [1-10]
    for map_set_name, maps in [("easier", EASIER_MAPS), ("difficult", DIFFICULT_MAPS)]:
        for include_diag in [True, False]:
            diag_suffix = "diag" if include_diag else "nodiag"
            run_name = f"dense_rlvl1-10_maps_{map_set_name}_{diag_suffix}_v2"
            experiments.append(
                {
                    "run_name": run_name,
                    "resource_levels": list(range(1, 11)),
                    "maps_to_use": maps,
                    "include_diagnostics": include_diag,
                }
            )

    print(f"\nSubmitting {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"\n{i}. {exp['run_name']}")
        print(f"   Resource levels: {exp['resource_levels']}")
        print(f"   Maps: {', '.join(exp['maps_to_use'])}")
        print(f"   Diagnostics: {'Included' if exp['include_diagnostics'] else 'Excluded'}")

        dense_curriculum.experiment(
            run_name=exp["run_name"],
            num_cogs=4,
            resource_levels=exp["resource_levels"],
            maps_to_use=exp["maps_to_use"],
            include_diagnostics=exp["include_diagnostics"],
            skip_git_check=True,
        )

    print("\n" + "=" * 80)
    print(f"✓ All {len(experiments)} dense curriculum experiments submitted!")
    print("=" * 80)


if __name__ == "__main__":
    submit_dense_experiments()

