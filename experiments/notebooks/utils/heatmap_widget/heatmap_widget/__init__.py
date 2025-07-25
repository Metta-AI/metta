from .HeatmapWidget import HeatmapWidget, create_demo_heatmap
from .metta_client import MettaAPIClient, fetch_real_heatmap_data

__all__ = [
    "HeatmapWidget",
    "create_demo_heatmap",
    "MettaAPIClient",
    "create_demo_heatmap",
    "fetch_real_heatmap_data",
]

if __name__ == "__main__":
    # Demo usage
    widget = create_demo_heatmap()
    print("Demo heatmap widget created!")
    print("Use widget.set_data() to update with your own data")
    print("Use widget.update_metric() to change the displayed metric")
    print("Use widget.on_cell_selected(), widget.on_replay_opened(), and widget.on_metric_changed() to add callbacks")
