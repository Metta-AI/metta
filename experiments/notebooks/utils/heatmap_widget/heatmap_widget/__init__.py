"""
Heatmap Widget for Jupyter Notebooks

An anywidget-based implementation of the observatory heatmap component.
Provides interactive policy evaluation heatmaps with hover and click functionality.
"""

import pathlib
from typing import Any, Callable, Dict, List

import anywidget
import traitlets

bundler_output_dir = pathlib.Path(__file__).parent / "static"


# FIXME: we need something like `dotenv` and `.env.local` files up in here.
# _DEV = False
_DEV = True

bundled_assets_dir = pathlib.Path(__file__).parent / "static"
src_dir = pathlib.Path(__file__).parent / "../src"

if _DEV:
    # from `npx vite`
    ESM = "http://localhost:5174/src/index.js?anywidget"
    print("DEV MODE")
else:
    # from `npx vite build`
    ESM = (bundled_assets_dir / "index.js").read_text()
    print("PRODUCTION MODE")

CSS = (src_dir / "styles.css").read_text()


class HeatmapWidget(anywidget.AnyWidget):
    """
    Interactive heatmap widget for policy evaluation data.

    Displays a heatmap of policy performance across different evaluations,
    with interactive features like hover, click, and replay URL opening.
    """

    # AnyWidget requires _esm property for JavaScript code
    _esm = ESM
    _css = CSS
    name = traitlets.Unicode("HeatmapWidget").tag(sync=True)

    # Widget traits (data that syncs between Python and JavaScript)
    heatmap_data = traitlets.Dict({}).tag(sync=True)
    selected_metric = traitlets.Unicode("reward").tag(sync=True)
    num_policies_to_show = traitlets.Int(20).tag(sync=True)
    selected_cell = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    replay_opened = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {"selected_cell": [], "replay_opened": [], "metric_changed": []}

        # This print should work now!
        print("🚀 HeatmapWidget initialized successfully!")

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
        """Set the heatmap data.

        Args:
            cells: Nested dict of {policy_name: {eval_name: {value, replayUrl, evalName}}}
            eval_names: List of evaluation names
            policy_names: List of policy names
            policy_average_scores: Dict mapping policy names to average scores
            selected_metric: Name of the selected metric
        """
        self.heatmap_data = {
            "cells": cells,
            "evalNames": eval_names,
            "policyNames": policy_names,
            "policyAverageScores": policy_average_scores,
        }
        self.selected_metric = selected_metric
        print(f"📊 Data set with {len(policy_names)} policies and {len(eval_names)} evaluations")
        print(f"📈 Selected metric: {selected_metric}")

    def set_multi_metric_data(
        self,
        cells: Dict[str, Dict[str, Dict[str, Any]]],
        eval_names: List[str],
        policy_names: List[str],
        metrics: List[str],
        selected_metric: str | None = None,
    ):
        """Set heatmap data with multiple metrics per cell.

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
            raise ValueError(f"Selected metric '{selected_metric}' not found in available metrics: {metrics}")

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

        self.heatmap_data = {
            "cells": cells,
            "evalNames": eval_names,
            "policyNames": policy_names,
            "policyAverageScores": policy_average_scores,
            "availableMetrics": metrics,
        }
        self.selected_metric = selected_metric or ""
        print(f"📊 Multi-metric data set with {len(policy_names)} policies and {len(eval_names)} evaluations")
        print(f"📈 Available metrics: {', '.join(metrics)}")
        print(f"📈 Selected metric: {selected_metric}")

    def update_metric(self, metric: str):
        """Update the selected metric.

        This will trigger a re-render of the heatmap with the new metric
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


def create_heatmap_widget(**kwargs) -> HeatmapWidget:
    """Create and return a new heatmap widget.

    Args:
        **kwargs: Additional keyword arguments passed to HeatmapWidget constructor

    Returns:
        HeatmapWidget instance
    """
    return HeatmapWidget(**kwargs)


def create_demo_heatmap() -> HeatmapWidget:
    """Create a demo heatmap widget with sample data.

    Returns:
        HeatmapWidget instance with demo data
    """
    print("🎯 Creating demo heatmap widget...")
    widget = HeatmapWidget()

    # Sample data structure matching the observatory format
    demo_cells = {
        "policy_alpha_v1": {
            "navigation/maze1": {"value": 85.2, "replayUrl": "sample_replay_1.json", "evalName": "navigation/maze1"},
            "navigation/maze2": {"value": 78.9, "replayUrl": "sample_replay_2.json", "evalName": "navigation/maze2"},
            "combat/arena1": {"value": 92.1, "replayUrl": "sample_replay_3.json", "evalName": "combat/arena1"},
            "combat/arena2": {"value": 88.7, "replayUrl": "sample_replay_4.json", "evalName": "combat/arena2"},
        },
        "policy_beta_v2": {
            "navigation/maze1": {"value": 79.8, "replayUrl": "sample_replay_5.json", "evalName": "navigation/maze1"},
            "navigation/maze2": {"value": 83.4, "replayUrl": "sample_replay_6.json", "evalName": "navigation/maze2"},
            "combat/arena1": {"value": 87.3, "replayUrl": "sample_replay_7.json", "evalName": "combat/arena1"},
            "combat/arena2": {"value": 91.2, "replayUrl": "sample_replay_8.json", "evalName": "combat/arena2"},
        },
        "policy_gamma_v1": {
            "navigation/maze1": {"value": 82.1, "replayUrl": "sample_replay_9.json", "evalName": "navigation/maze1"},
            "navigation/maze2": {"value": 76.5, "replayUrl": "sample_replay_10.json", "evalName": "navigation/maze2"},
            "combat/arena1": {"value": 89.8, "replayUrl": "sample_replay_11.json", "evalName": "combat/arena1"},
            "combat/arena2": {"value": 85.9, "replayUrl": "sample_replay_12.json", "evalName": "combat/arena2"},
        },
    }

    demo_eval_names = ["navigation/maze1", "navigation/maze2", "combat/arena1", "combat/arena2"]
    demo_policy_names = list(demo_cells.keys())
    demo_policy_averages = {
        "policy_alpha_v1": 86.2,
        "policy_beta_v2": 85.4,
        "policy_gamma_v1": 83.6,
    }

    widget.set_data(
        cells=demo_cells,
        eval_names=demo_eval_names,
        policy_names=demo_policy_names,
        policy_average_scores=demo_policy_averages,
        selected_metric="reward",
    )

    # Add some demo callbacks
    def on_cell_selected(cell_info):
        print(f"📍 Cell selected: {cell_info['policyUri']} on {cell_info['evalName']}")

    def on_replay_opened(replay_info):
        print(
            f"🎬 Replay opened: {replay_info['replayUrl']} for {replay_info['policyUri']} on {replay_info['evalName']}"
        )

    def on_metric_changed(metric_name):
        print(f"📊 Metric changed to: {metric_name}")

    widget.on_cell_selected(on_cell_selected)
    widget.on_replay_opened(on_replay_opened)
    widget.on_metric_changed(on_metric_changed)

    print("✅ Demo heatmap widget created with sample data!")
    return widget


def create_multi_metric_demo() -> HeatmapWidget:
    """Create a demo heatmap widget with multiple metrics per cell.

    This demonstrates how selectedMetric actually changes the displayed values.

    Returns:
        HeatmapWidget instance with multi-metric demo data
    """
    print("🎯 Creating multi-metric demo heatmap widget...")
    widget = HeatmapWidget()

    # Sample data with multiple metrics per cell
    demo_cells = {
        "policy_alpha_v1": {
            "navigation/maze1": {
                "metrics": {"reward": 85.2, "episode_length": 120.5, "success_rate": 0.92, "completion_time": 45.3},
                "replayUrl": "sample_replay_1.json",
                "evalName": "navigation/maze1",
            },
            "navigation/maze2": {
                "metrics": {"reward": 78.9, "episode_length": 135.2, "success_rate": 0.85, "completion_time": 52.1},
                "replayUrl": "sample_replay_2.json",
                "evalName": "navigation/maze2",
            },
            "combat/arena1": {
                "metrics": {"reward": 92.1, "episode_length": 98.7, "success_rate": 0.98, "completion_time": 38.9},
                "replayUrl": "sample_replay_3.json",
                "evalName": "combat/arena1",
            },
            "combat/arena2": {
                "metrics": {"reward": 88.7, "episode_length": 110.3, "success_rate": 0.94, "completion_time": 42.6},
                "replayUrl": "sample_replay_4.json",
                "evalName": "combat/arena2",
            },
        },
        "policy_beta_v2": {
            "navigation/maze1": {
                "metrics": {"reward": 79.8, "episode_length": 142.1, "success_rate": 0.88, "completion_time": 49.7},
                "replayUrl": "sample_replay_5.json",
                "evalName": "navigation/maze1",
            },
            "navigation/maze2": {
                "metrics": {"reward": 83.4, "episode_length": 128.9, "success_rate": 0.91, "completion_time": 47.2},
                "replayUrl": "sample_replay_6.json",
                "evalName": "navigation/maze2",
            },
            "combat/arena1": {
                "metrics": {"reward": 87.3, "episode_length": 105.6, "success_rate": 0.96, "completion_time": 41.4},
                "replayUrl": "sample_replay_7.json",
                "evalName": "combat/arena1",
            },
            "combat/arena2": {
                "metrics": {"reward": 91.2, "episode_length": 102.4, "success_rate": 0.97, "completion_time": 40.1},
                "replayUrl": "sample_replay_8.json",
                "evalName": "combat/arena2",
            },
        },
        "policy_gamma_v1": {
            "navigation/maze1": {
                "metrics": {"reward": 82.1, "episode_length": 130.7, "success_rate": 0.89, "completion_time": 48.5},
                "replayUrl": "sample_replay_9.json",
                "evalName": "navigation/maze1",
            },
            "navigation/maze2": {
                "metrics": {"reward": 76.5, "episode_length": 145.3, "success_rate": 0.83, "completion_time": 54.8},
                "replayUrl": "sample_replay_10.json",
                "evalName": "navigation/maze2",
            },
            "combat/arena1": {
                "metrics": {"reward": 89.8, "episode_length": 108.2, "success_rate": 0.95, "completion_time": 43.1},
                "replayUrl": "sample_replay_11.json",
                "evalName": "combat/arena1",
            },
            "combat/arena2": {
                "metrics": {"reward": 85.9, "episode_length": 115.6, "success_rate": 0.93, "completion_time": 44.7},
                "replayUrl": "sample_replay_12.json",
                "evalName": "combat/arena2",
            },
        },
    }

    demo_eval_names = ["navigation/maze1", "navigation/maze2", "combat/arena1", "combat/arena2"]
    demo_policy_names = list(demo_cells.keys())
    available_metrics = ["reward", "episode_length", "success_rate", "completion_time"]

    widget.set_multi_metric_data(
        cells=demo_cells,
        eval_names=demo_eval_names,
        policy_names=demo_policy_names,
        metrics=available_metrics,
        selected_metric="reward",
    )

    # Add some demo callbacks
    def on_cell_selected(cell_info):
        print(f"📍 Cell selected: {cell_info['policyUri']} on {cell_info['evalName']}")

    def on_replay_opened(replay_info):
        print(
            f"🎬 Replay opened: {replay_info['replayUrl']} for {replay_info['policyUri']} on {replay_info['evalName']}"
        )

    def on_metric_changed(metric_name):
        print(f"📊 Metric changed to: {metric_name}")

    widget.on_cell_selected(on_cell_selected)
    widget.on_replay_opened(on_replay_opened)
    widget.on_metric_changed(on_metric_changed)

    print("✅ Multi-metric demo heatmap widget created!")
    print("📈 Try widget.update_metric('episode_length') to see values change!")
    return widget


if __name__ == "__main__":
    # Demo usage
    widget = create_demo_heatmap()
    print("Demo heatmap widget created!")
    print("Use widget.set_data() to update with your own data")
    print("Use widget.update_metric() to change the displayed metric")
    print("Use widget.on_cell_selected(), widget.on_replay_opened(), and widget.on_metric_changed() to add callbacks")
