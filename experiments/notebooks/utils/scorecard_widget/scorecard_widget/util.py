from experiments.notebooks.utils.scorecard_widget.scorecard_widget.ScorecardWidget import (
    ScorecardWidget,
    create_scorecard_widget,
)
from metta.app_backend.clients.scorecard_client import ScorecardClient
from metta.app_backend.routes.scorecard_routes import ScorecardData
from metta.setup.utils import info, warning
from typing_extensions import Literal


async def fetch_real_scorecard_data(
    client: ScorecardClient,
    restrict_to_metrics: list[str] | None = None,
    restrict_to_policy_names: list[str] | None = None,
    policy_selector: Literal["best", "latest"] = "best",
    max_policies: int = 30,
    include_run_free_policies: bool = False,
) -> ScorecardWidget:
    """
    Fetch real evaluation data using the metta HTTP API (same as repo.ts).

    Args:
        metrics: List of metrics to include (e.g., ["reward", "heart.get"])
        search_texts: List of search texts to use to find training runs (e.g., ["relh.skypilot", "daveey.arena.rnd"])
        policy_selector: "best" or "latest" policy selection strategy
        api_base_url: Base URL for the stats server
        max_policies: Maximum number of policies to display

    Returns:
        ScorecardWidget with real data
    """
    # Step 1: Get available policies to find training run IDs
    # TODO: backend should be doing the filtering, not frontend

    # Find training run IDs that match our training run names
    training_run_ids = []
    run_free_policy_ids = []
    policies_data = await client.get_policies()
    for policy in policies_data.policies:
        if policy.type == "training_run" and (
            not restrict_to_policy_names
            or any(
                filter_policy_name in policy.name
                for filter_policy_name in restrict_to_policy_names
            )
        ):
            training_run_ids.append(policy.id)
        elif policy.type == "policy" and include_run_free_policies:
            run_free_policy_ids.append(policy.id)

    if not training_run_ids:
        raise Exception("No training runs found")
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
    valid_metrics = list(
        filter(
            lambda m: (not restrict_to_metrics or m in restrict_to_metrics),
            available_metrics,
        )
    )
    if not valid_metrics:
        info(f"Available metrics: {sorted(available_metrics)}")
        if restrict_to_metrics:
            warning(
                f"None of the requested metrics {restrict_to_metrics} are available"
            )
        warning(f"Available metrics are: {sorted(available_metrics)}")
        raise Exception("No valid metrics found")

    # Step 4: Generate scorecard for the first metric
    primary_metric = valid_metrics[0]
    scorecard_data: ScorecardData = await client.generate_scorecard(
        training_run_ids=training_run_ids,
        run_free_policy_ids=[],
        eval_names=flat_eval_names,
        metric=primary_metric,
        policy_selector=policy_selector,
    )

    if not scorecard_data.policyNames:
        raise Exception("No scorecard policyNames. No scorecard data generated")

    # Limit policies if requested
    policy_names = list(scorecard_data.policyNames)
    if len(policy_names) > max_policies:
        # Sort by average score and take top N
        avg_scores = scorecard_data.policyAverageScores
        top_policies = sorted(
            avg_scores.keys(), key=lambda p: avg_scores[p], reverse=True
        )[:max_policies]

        # Filter the data
        filtered_cells = {
            p: scorecard_data.cells[p]
            for p in top_policies
            if p in scorecard_data.cells
        }
        scorecard_data.policyNames = top_policies
        scorecard_data.cells = filtered_cells
        scorecard_data.policyAverageScores = {
            p: avg_scores[p] for p in top_policies if p in avg_scores
        }

    # Step 5: Convert to widget format
    cells = {}
    for policy_name in scorecard_data.policyNames:
        cells[policy_name] = {}
        for eval_name in scorecard_data.evalNames:
            cell = scorecard_data.cells.get(policy_name, {}).get(eval_name)
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
    widget = create_scorecard_widget()
    widget.set_multi_metric_data(
        cells=cells,
        eval_names=scorecard_data.evalNames,
        policy_names=scorecard_data.policyNames,
        metrics=valid_metrics,  # Use only the valid metrics that were found
        selected_metric=primary_metric,
    )

    return widget
