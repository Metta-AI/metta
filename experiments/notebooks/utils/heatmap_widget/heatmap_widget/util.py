from typing import List

from metta.common.client.metta_client import MettaAPIClient

from .HeatmapWidget import HeatmapWidget, create_heatmap_widget


async def fetch_real_heatmap_data(
    training_run_names: List[str],
    metrics: List[str],
    policy_selector: str = "best",
    api_base_url: str = "http://localhost:8000",
    max_policies: int = 20,
) -> HeatmapWidget:
    """
    Fetch real evaluation data using the metta HTTP API (same as repo.ts).

    Args:
        training_run_names: List of training run names (e.g., ["daveey.arena.rnd.16x4.2"])
        metrics: List of metrics to include (e.g., ["reward", "heart.get"])
        policy_selector: "best" or "latest" policy selection strategy
        api_base_url: Base URL for the stats server
        max_policies: Maximum number of policies to display

    Returns:
        HeatmapWidget with real data
    """
    client = MettaAPIClient(api_base_url)

    # Step 1: Get available policies to find training run IDs
    # TODO: backend should be doing the filtering, not frontend
    policies_data = await client.get_policies(page_size=100)

    # Find training run IDs that match our training run names
    training_run_ids = []
    for policy in policies_data.policies:
        if policy.type == "training_run" and any(
            run_name in policy.name for run_name in training_run_names
        ):
            training_run_ids.append(policy.id)

    if not training_run_ids:
        print(f"❌ No training runs found matching: {training_run_names}")
        raise Exception(
            f"No training runs found matching: {training_run_names}. This may be due to a limitation in the backend. \
            We're working on a fix!"
        )

    # Step 2: Get available evaluations for these training runs
    eval_names = await client.get_eval_names(training_run_ids, [])
    if not eval_names:
        print("❌ No evaluations found for selected training runs")
        raise Exception("No evaluations found for selected training runs")

    # Step 3: Get available metrics
    available_metrics = await client.get_available_metrics(
        training_run_ids, run_free_policy_ids=[], eval_names=eval_names
    )
    if not available_metrics:
        print("❌ No metrics found")
        raise Exception("No metrics found for selected training runs and evaluations")

    # Filter to requested metrics that actually exist
    valid_metrics = [m for m in metrics if m in available_metrics]
    if not valid_metrics:
        print(f"❌ None of the requested metrics {metrics} are available")
        raise Exception(f"None of the requested metrics {metrics} are available")

    # Step 4: Generate heatmap for the first metric
    primary_metric = valid_metrics[0]
    keys = eval_names
    heatmap_data = await client.generate_heatmap(
        training_run_ids, [], keys, primary_metric, policy_selector
    )

    if not heatmap_data.policyNames:
        raise Exception("❌ No heatmap policyNames. No heatmap data generated")

    # Limit policies if requested
    policy_names = heatmap_data.policyNames
    if len(policy_names) > max_policies:
        # Sort by average score and take top N
        avg_scores = heatmap_data.policyAverageScores
        top_policies = sorted(
            avg_scores.keys(), key=lambda p: avg_scores[p], reverse=True
        )[:max_policies]

        # Filter the data
        filtered_cells = {
            p: heatmap_data.cells[p] for p in top_policies if p in heatmap_data.cells
        }
        heatmap_data.policyNames = top_policies
        heatmap_data.cells = filtered_cells
        heatmap_data.policyAverageScores = {
            p: avg_scores[p] for p in top_policies if p in avg_scores
        }

    # Step 5: Convert to widget format
    cells = {}
    for policy_name in heatmap_data.policyNames:
        cells[policy_name] = {}
        for eval_name in heatmap_data.evalNames:
            cell = heatmap_data.cells.get(policy_name, {}).get(eval_name)
            if cell:
                cells[policy_name][eval_name] = {
                    "metrics": {primary_metric: cell.value},
                    "replayUrl": cell.replayUrl,
                    "evalName": eval_name,
                }
            else:
                cells[policy_name][eval_name] = {
                    "metrics": {primary_metric: 0.0},
                    "replayUrl": None,
                    "evalName": eval_name,
                }

    # Create widget
    widget = create_heatmap_widget()
    widget.set_multi_metric_data(
        cells=cells,
        eval_names=heatmap_data.evalNames,
        policy_names=heatmap_data.policyNames,
        metrics=[primary_metric],
        selected_metric=primary_metric,
    )

    return widget
