"""
Scorecard Widget for Jupyter Notebooks

An anywidget-based implementation of the observatory scorecard component.
Provides interactive policy evaluation scorecards with hover and click functionality.
"""

import pathlib
from typing import Any, Callable, Dict, List

import anywidget
import traitlets

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {
            "selected_cell": [],
            "replay_opened": [],
            "metric_changed": [],
        }

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
        print(
            f"📊 Multi-metric data set with {len(policy_names)} policies and {len(eval_names)} evaluations"
        )
        print(f"📈 Available metrics: {', '.join(metrics)}")
        print(f"📈 Selected metric: {selected_metric}")

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


def create_scorecard_widget(**kwargs) -> ScorecardWidget:
    """Create and return a new scorecard widget.

    Args:
        **kwargs: Additional keyword arguments passed to ScorecardWidget constructor

    Returns:
        ScorecardWidget instance
    """
    return ScorecardWidget(**kwargs)
