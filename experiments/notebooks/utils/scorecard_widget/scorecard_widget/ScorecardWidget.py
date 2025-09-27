"""
Scorecard Widget for Jupyter Notebooks

An anywidget-based implementation of the observatory scorecard component.
Provides interactive policy evaluation scorecards with hover and click functionality.
"""

import pathlib
from logging import warning
from typing import Any, Callable, Dict, List, Literal

import anywidget
import traitlets
from softmax.orchestrator.clients.scorecard_client import ScorecardClient
from softmax.orchestrator.routes.scorecard_routes import ScorecardData

# FIXME: we need something like `dotenv` and `.env.local` files up in here.
_DEV = False
# _DEV = True

bundled_assets_dir = pathlib.Path(__file__).parent / "static"
src_dir = pathlib.Path(__file__).parent / "../src"

if _DEV:
    ESM = "http://localhost:5174/src/index.js?anywidget"
else:
    ESM = (bundled_assets_dir / "index.js").read_text()

CSS = (src_dir / "styles.css").read_text()


class ScorecardWidget(anywidget.AnyWidget):
    """
    Interactive scorecard widget for policy evaluation data.

    Displays a scorecard of policy performance across different evaluations,
    with interactive features like hover, click, and replay URL opening.
    """

    # AnyWidget requires _esm property for JavaScript code
    _esm = ESM
    _css = CSS
    name = traitlets.Unicode("ScorecardWidget").tag(sync=True)

    # Widget traits (data that syncs between Python and JavaScript)
    scorecard_data = traitlets.Dict({}).tag(sync=True)
    selected_metric = traitlets.Unicode("reward").tag(sync=True)
    num_policies_to_show = traitlets.Int(20).tag(sync=True)
    selected_cell = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    replay_opened = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, client: ScorecardClient | None = None, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {
            "selected_cell": [],
            "replay_opened": [],
            "metric_changed": [],
        }
        self.client = client
        if not self.client:
            self.client = ScorecardClient()

        # This print should work now!
        print("🚀 ScorecardWidget initialized successfully!")

        # Set up observers
        self.observe(self._on_cell_selected, names="selected_cell")
        self.observe(self._on_replay_opened, names="replay_opened")
        self.observe(self._on_metric_changed, names="selected_metric")

    def _on_cell_selected(self, change):
        """Handle cell selection events from JavaScript."""
        if change["new"]:
            for callback in self._callbacks["selected_cell"]:
                callback(change["new"])

    def _on_replay_opened(self, change):
        """Handle replay opening events from JavaScript."""
        if change["new"]:
            for callback in self._callbacks["replay_opened"]:
                callback(change["new"])

    def _on_metric_changed(self, change):
        """Handle metric change events."""
        for callback in self._callbacks["metric_changed"]:
            callback(change["new"])

    def on_cell_selected(self, callback: Callable[[Dict[str, str]], None]):
        """Register a callback for when a cell is selected.

        Args:
            callback: Function that receives {'policyUri': str, 'evalName': str}
        """
        self._callbacks["selected_cell"].append(callback)

    def on_replay_opened(self, callback: Callable[[Dict[str, str]], None]):
        """Register a callback for when a replay is opened.

        Args:
            callback: Function that receives {'policyUri': str, 'evalName': str, 'replayUrl': str}
        """
        self._callbacks["replay_opened"].append(callback)

    def on_metric_changed(self, callback: Callable[[str], None]):
        """Register a callback for when the metric is changed.

        Args:
            callback: Function that receives the new metric name as a string
        """
        self._callbacks["metric_changed"].append(callback)

    def set_data(
        self,
        cells: Dict[str, Dict[str, Dict[str, Any]]],
        eval_names: List[str],
        policy_names: List[str],
        policy_average_scores: Dict[str, float],
        selected_metric: str = "reward",
    ):
        """Set the scorecard data.

        Args:
            cells: Nested dict of {policy_name: {eval_name: {value, replayUrl, evalName}}}
            eval_names: List of evaluation names
            policy_names: List of policy names
            policy_average_scores: Dict mapping policy names to average scores
            selected_metric: Name of the selected metric
        """
        self.scorecard_data = {
            "cells": cells,
            "evalNames": eval_names,
            "policyNames": policy_names,
            "policyAverageScores": policy_average_scores,
        }
        self.selected_metric = selected_metric
        print(
            f"📊 Data set with {len(policy_names)} policies and {len(eval_names)} evaluations"
        )
        print(f"📈 Selected metric: {selected_metric}")

    def set_multi_metric_data(
        self,
        cells: Dict[str, Dict[str, Dict[str, Any]]],
        eval_names: List[str],
        policy_names: List[str],
        metrics: List[str],
        selected_metric: str | None = None,
    ):
        """Set scorecard data with multiple metrics per cell.

        Args:
            cells: Nested dict of {policy_name: {eval_name: {metrics: {metric_name: value}, replayUrl, evalName}}}
            eval_names: List of evaluation names
            policy_names: List of policy names
            metrics: List of available metric names
            selected_metric: Name of the selected metric (defaults to first metric)
        """
        if not selected_metric and metrics:
            selected_metric = metrics[0]
        elif selected_metric and selected_metric not in metrics:
            raise ValueError(
                f"Selected metric '{selected_metric}' not found in available metrics: {metrics}"
            )

        # Calculate policy averages for the selected metric
        policy_average_scores = {}
        for policy_name in policy_names:
            total = 0
            count = 0
            for eval_name in eval_names:
                cell = cells.get(policy_name, {}).get(eval_name, {})
                if cell and "metrics" in cell and selected_metric in cell["metrics"]:
                    total += cell["metrics"][selected_metric]
                    count += 1
            policy_average_scores[policy_name] = total / count if count > 0 else 0

        self.scorecard_data = {
            "cells": cells,
            "evalNames": eval_names,
            "policyNames": policy_names,
            "policyAverageScores": policy_average_scores,
            "availableMetrics": metrics,
        }
        self.selected_metric = selected_metric or ""

    def update_metric(self, metric: str):
        """Update the selected metric.

        This will trigger a re-render of the scorecard with the new metric
        displayed in titles and labels.

        Args:
            metric: Name of the metric to display
        """
        if metric != self.selected_metric:
            self.selected_metric = metric
        else:
            print(f"📈 Metric already set to: {metric}")

    def set_num_policies(self, num: int):
        """Set the number of policies to show."""
        if num > 0:
            self.num_policies_to_show = num
            print(f"🔢 Number of policies to show set to: {num}")

    def get_current_metric(self) -> str:
        """Get the currently selected metric.

        Returns:
            The name of the currently selected metric
        """
        return self.selected_metric

    async def fetch_real_scorecard_data(
        self,
        search_term: str | None = None,
        restrict_to_policy_ids: list[str] | None = None,
        restrict_to_metrics: list[str] | None = None,
        restrict_to_policy_names: list[str] | None = None,
        restrict_to_eval_names: list[str] | None = None,
        policy_selector: Literal["best", "latest"] = "best",
        max_policies: int = 30,
        primary_metric: str | None = None,
        include_run_free_policies: bool = False,
    ):
        """
        Fetch real evaluation data using the metta HTTP API (same as repo.ts).

        Args:
            client: ScorecardClient instance
            search_term: Search term to filter policies by name
            restrict_to_policy_ids: List of policy IDs to include (e.g., ["123", "456"])
            restrict_to_metrics: List of metrics to include (e.g., ["reward", "heart.get"])
            restrict_to_policy_names: List of policy name filters (e.g., ["relh.skypilot", "daveey.arena.rnd"])
            restrict_to_eval_names: List of specific evaluation names to include (e.g., ["navigation/labyrinth", "memory/easy"])
            policy_selector: "best" or "latest" policy selection strategy
            max_policies: Maximum number of policies to display
            include_run_free_policies: Whether to include standalone policies

        Returns:
            ScorecardWidget with real data
        """
        if (
            restrict_to_policy_ids == []
            or restrict_to_metrics == []
            or restrict_to_policy_names == []
            or restrict_to_eval_names == []
        ):
            return

        if not primary_metric:
            if restrict_to_metrics:
                primary_metric = restrict_to_metrics[0]
            else:
                primary_metric = "reward"
        if restrict_to_metrics and primary_metric not in restrict_to_metrics:
            raise ValueError(
                f"Primary metric {primary_metric} not found in restrict_to_metrics {restrict_to_metrics}"
            )

        if not self.client:
            raise ValueError("client is required to fetch scorecard data")

        if search_term:
            policies_data = await self.client.search_policies(search=search_term)
        else:
            policies_data = await self.client.get_policies()

        # Find training run IDs that match our training run names
        training_run_ids = []
        run_free_policy_ids = []
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

        if restrict_to_policy_ids:
            training_run_ids = [
                policy_id
                for policy_id in restrict_to_policy_ids
                if policy_id in training_run_ids
            ]
            run_free_policy_ids = [
                policy_id
                for policy_id in restrict_to_policy_ids
                if policy_id in run_free_policy_ids
            ]

        if not training_run_ids:
            raise Exception("No training runs found")

        if restrict_to_eval_names:
            # Use the specific eval names provided
            eval_names = restrict_to_eval_names
        else:
            # Get all available evaluations for these policies
            eval_names = await self.client.get_eval_names(
                training_run_ids, run_free_policy_ids
            )
            if not eval_names:
                raise Exception("No evaluations found for selected training runs")
            print(f"Found {len(eval_names)} available evaluations")

        available_metrics = await self.client.get_available_metrics(
            training_run_ids, run_free_policy_ids, eval_names
        )
        if not available_metrics:
            raise Exception("No metrics found for selected policies and evaluations")

        # Filter to requested metrics that actually exist
        valid_metrics = list(
            filter(
                lambda m: (not restrict_to_metrics or m in restrict_to_metrics),
                available_metrics,
            )
        )
        if not valid_metrics:
            print(f"Available metrics: {sorted(available_metrics)}")
            if restrict_to_metrics:
                warning(
                    f"None of the requested metrics {restrict_to_metrics} are available"
                )
            warning(f"Available metrics are: {sorted(available_metrics)}")
            raise Exception("No valid metrics found")

        scorecard_data: ScorecardData = await self.client.generate_scorecard(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
            metric=primary_metric,
            policy_selector=policy_selector,
        )

        all_policies = training_run_ids + run_free_policy_ids
        if len(all_policies) != len(scorecard_data.policyNames):
            warning(
                f"Number of policies in scorecard data ({len(scorecard_data.policyNames)}) does not match number of policies in your query ({len(training_run_ids) + len(run_free_policy_ids)})"
            )
            raise Exception(
                "Number of policies in scorecard data does not match number of policies in your query"
            )

        if not scorecard_data.policyNames:
            warning("No scorecard data found in the database for your query:")
            warning(f"  training_run_ids={training_run_ids}")
            warning(f"  run_free_policy_ids={run_free_policy_ids}")
            warning(f"  eval_names={eval_names}")
            warning(f"  primary_metric={primary_metric}")
            raise Exception("No scorecard data found in database for your query")

        cells = self._make_cells_from_scorecard_data(
            scorecard_data=scorecard_data,
            max_policies=max_policies,
            primary_metric=primary_metric,
        )

        self.set_multi_metric_data(
            cells=cells,
            eval_names=scorecard_data.evalNames,
            policy_names=scorecard_data.policyNames,
            metrics=valid_metrics,  # Use only the valid metrics that were found
            selected_metric=primary_metric,
        )

    def _make_cells_from_scorecard_data(
        self,
        scorecard_data: ScorecardData,
        max_policies: int,
        primary_metric: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        policy_names = list(scorecard_data.policyNames)
        if len(policy_names) > max_policies:
            # Sort by average score and take top N
            avg_scores = scorecard_data.policyAverageScores
            top_policies = sorted(
                avg_scores.keys(), key=lambda p: avg_scores[p], reverse=True
            )[:max_policies]

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

        return cells


def create_scorecard_widget(**kwargs) -> ScorecardWidget:
    """Create and return a new scorecard widget.

    Args:
        **kwargs: Additional keyword arguments passed to ScorecardWidget constructor

    Returns:
        ScorecardWidget instance
    """
    return ScorecardWidget(**kwargs)
