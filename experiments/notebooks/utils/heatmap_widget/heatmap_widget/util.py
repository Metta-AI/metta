from typing import List

from experiments.notebooks.utils.heatmap_widget.heatmap_widget.HeatmapWidget import (
    HeatmapWidget,
    create_heatmap_widget,
)
from metta.app_backend.clients.scorecard_client import ScorecardClient
from metta.app_backend.routes.heatmap_routes import HeatmapData
from metta.setup.utils import info, warning
from typing_extensions import Literal


async def get_available_metrics(
    client: ScorecardClient,
    search_texts: List[str] = [],
    include_run_free_policies: bool = False,
) -> List[str]:
    """
    Get available metrics for the given search texts without generating a heatmap.
    Useful for exploring what metrics are available before calling fetch_real_heatmap_data.

    Args:
        client: ScorecardClient instance
        search_texts: List of search texts to use to find training runs
        include_run_free_policies: Whether to include run-free policies

    Returns:
        List of available metric names
    """
    # Find training run IDs that match our training run names
    training_run_ids = []
    run_free_policy_ids = []
    if search_texts:
        for search_text in search_texts:
            policies_data = await client.get_policies(
                search_text=search_text, page_size=100
            )
            for policy in policies_data.policies:
                if policy.type == "training_run" and search_text in policy.name:
                    training_run_ids.append(policy.id)
                elif policy.type == "policy" and include_run_free_policies:
                    run_free_policy_ids.append(policy.id)
    else:
        raise Exception(
            "No search_texts provided. Please provide at least one so we can search for policies."
        )

    if not training_run_ids:
        raise Exception(f"No training runs found matching: {search_texts}.")

    # Get available evaluations for these training runs
    eval_names_tuples = await client.get_eval_names(
        training_run_ids, run_free_policy_ids
    )

    if not eval_names_tuples:
        warning("No evaluations found for selected training runs")
        return []

    # Flatten eval_names structure - it comes as [('category', ['eval1', 'eval2', ...])]
    flat_eval_names = []
    for item in eval_names_tuples:
        category, evals = item
        flat_eval_names.extend(evals)

    available_metrics = await client.get_available_metrics(
        training_run_ids, run_free_policy_ids, flat_eval_names
    )

    return sorted(available_metrics)


async def fetch_real_heatmap_data(
    client: ScorecardClient,
    metrics: List[str],
    search_texts: List[str] = [],
    policy_selector: Literal["best", "latest"] = "best",
    max_policies: int = 30,
    include_run_free_policies: bool = False,
) -> HeatmapWidget:
    """
    Fetch real evaluation data using the metta HTTP API (same as repo.ts).

    Args:
        metrics: List of metrics to include (e.g., ["reward", "heart.get"])
        search_texts: List of search texts to use to find training runs (e.g., ["relh.skypilot", "daveey.arena.rnd"])
        policy_selector: "best" or "latest" policy selection strategy
        api_base_url: Base URL for the stats server
        max_policies: Maximum number of policies to display

    Returns:
        HeatmapWidget with real data
    """
    # Step 1: Get available policies to find training run IDs
    # TODO: backend should be doing the filtering, not frontend

    # Find training run IDs that match our training run names
    training_run_ids = []
    run_free_policy_ids = []
    if search_texts:
        for search_text in search_texts:
            policies_data = await client.get_policies(
                search_text=search_text, page_size=100
            )
            for policy in policies_data.policies:
                if policy.type == "training_run" and search_text in policy.name:
                    training_run_ids.append(policy.id)
                elif policy.type == "policy" and include_run_free_policies:
                    run_free_policy_ids.append(policy.id)
    else:
        raise Exception(
            "No search_texts provided. Please provide at least one so we can search for policies."
        )

    if not training_run_ids:
        raise Exception(
            f"No training runs found matching: {search_texts}. This may be due to a \
                limitation in the backend. We're working on a fix!"
        )
    info(f"Training run IDs: {len(training_run_ids)}")
    info(f"Run free policy IDs: {len(run_free_policy_ids)}")

    # Step 2: Get available evaluations for these training runs
    eval_names = await client.get_eval_names(training_run_ids, run_free_policy_ids)

    if not eval_names:
        warning("No evaluations found for selected training runs")
        raise Exception("No evaluations found for selected training runs")

    # Step 3: Get available metrics
    # Flatten eval_names structure - it comes as [('category', ['eval1', 'eval2', ...])]
    flat_eval_names = []
    for category, evals in eval_names:
        flat_eval_names.extend(evals)

    available_metrics = await client.get_available_metrics(
        training_run_ids, run_free_policy_ids, flat_eval_names
    )
    if not available_metrics:
        warning("No metrics found")
        raise Exception("No metrics found for selected training runs and evaluations")

    # Filter to requested metrics that actually exist
    valid_metrics = [m for m in metrics if m in available_metrics]
    if not valid_metrics:
        info(f"Available metrics: {sorted(available_metrics)}")
        warning(f"None of the requested metrics {metrics} are available")
        warning(f"Available metrics are: {sorted(available_metrics)}")
        raise Exception(
            f"None of the requested metrics {metrics} are available. Available metrics: {sorted(available_metrics)}"
        )

    # Step 4: Generate heatmap for the first metric
    primary_metric = valid_metrics[0]
    heatmap_data: HeatmapData = await client.generate_heatmap(
        training_run_ids=training_run_ids,
        run_free_policy_ids=[],
        eval_names=flat_eval_names,
        metric=primary_metric,
        policy_selector=policy_selector,
    )

    if not heatmap_data.policyNames:
        raise Exception("No heatmap policyNames. No heatmap data generated")

    # Limit policies if requested
    policy_names = list(heatmap_data.policyNames)
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
        metrics=valid_metrics,  # Use only the valid metrics that were found
        selected_metric=primary_metric,
    )

    return widget
