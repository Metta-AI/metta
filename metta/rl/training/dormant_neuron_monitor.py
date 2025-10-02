"""Dormant neuron detection training component."""

import logging
from typing import Dict

import torch
from pydantic import Field

from metta.rl.training import TrainerComponent
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class DormantNeuronMonitorConfig(Config):
    """Configuration for dormant neuron monitoring."""

    epoch_interval: int = Field(default=1, description="How often to check for dormant neurons (in epochs)")
    weight_threshold: float = Field(
        default=1e-6, description="Weight magnitude threshold below which neurons are considered dormant"
    )
    min_layer_size: int = Field(default=10, description="Minimum layer size to analyze (skip very small layers)")
    track_by_layer: bool = Field(default=True, description="Track dormant neurons per layer")
    track_overall: bool = Field(default=True, description="Track overall dormant neuron statistics")


class DormantNeuronMonitor(TrainerComponent):
    """Monitors neural network weights to detect dormant neurons at the end of each epoch."""

    def __init__(self, config: DormantNeuronMonitorConfig):
        """Initialize dormant neuron monitor component."""
        enabled = config.epoch_interval > 0
        super().__init__(epoch_interval=config.epoch_interval if enabled else 0)
        self._master_only = True
        self._enabled = enabled
        self._config = config

        # Track dormant neuron history
        self._dormant_neuron_history: Dict[str, list] = {}

    def on_epoch_end(self, epoch: int) -> None:
        """Analyze weights to detect dormant neurons."""
        if not self._enabled:
            return

        context = self.context
        policy = context.policy

        dormant_neuron_stats = self._analyze_dormant_neurons(policy, epoch)

        # Update context with dormant neuron stats
        if hasattr(context, "dormant_neuron_stats"):
            context.dormant_neuron_stats.update(dormant_neuron_stats)
        else:
            context.dormant_neuron_stats = dormant_neuron_stats

        # Report to stats reporter if available
        stats_reporter = context.stats_reporter
        if stats_reporter and hasattr(stats_reporter, "update_dormant_neuron_stats"):
            stats_reporter.update_dormant_neuron_stats(dormant_neuron_stats)
        elif stats_reporter:
            # Fallback: add to general stats
            for key, value in dormant_neuron_stats.items():
                setattr(stats_reporter, f"dormant_neuron_{key}", value)

        logger.debug(f"Dormant neuron analysis completed for epoch {epoch}: {len(dormant_neuron_stats)} metrics")

    def _analyze_dormant_neurons(self, policy: torch.nn.Module, epoch: int) -> Dict[str, float]:
        """Analyze policy weights to detect dormant neurons."""
        dormant_neuron_stats = {}
        total_dormant_neurons = 0
        total_neurons = 0

        for name, param in policy.named_parameters():
            # Skip biases and 1D parameters
            if param.dim() < 2:
                continue

            # Skip very small layers
            if param.numel() < self._config.min_layer_size:
                continue

            # Analyze this layer
            layer_dormant_neurons = self._count_dormant_neurons_in_layer(param, name)
            layer_total_neurons = param.numel()

            total_dormant_neurons += layer_dormant_neurons
            total_neurons += layer_total_neurons

            if self._config.track_by_layer:
                # Per-layer statistics
                layer_name = name.replace(".", "_")
                # Calculate total output neurons (not total parameters)
                total_output_neurons = param.size(0)  # First dimension is output neurons
                dormant_neuron_stats[f"dormant_neurons/{layer_name}/count"] = float(layer_dormant_neurons)
                dormant_neuron_stats[f"dormant_neurons/{layer_name}/ratio"] = float(
                    layer_dormant_neurons / total_output_neurons
                )
                dormant_neuron_stats[f"dormant_neurons/{layer_name}/total_neurons"] = float(total_output_neurons)

                # Track history
                if layer_name not in self._dormant_neuron_history:
                    self._dormant_neuron_history[layer_name] = []
                self._dormant_neuron_history[layer_name].append(layer_dormant_neurons)

        if self._config.track_overall and total_neurons > 0:
            # Overall statistics
            dormant_neuron_stats["dormant_neurons/overall/count"] = float(total_dormant_neurons)
            dormant_neuron_stats["dormant_neurons/overall/ratio"] = float(total_dormant_neurons / total_neurons)
            dormant_neuron_stats["dormant_neurons/overall/total_neurons"] = float(total_neurons)

            # Track overall history
            if "overall" not in self._dormant_neuron_history:
                self._dormant_neuron_history["overall"] = []
            self._dormant_neuron_history["overall"].append(total_dormant_neurons)

        return dormant_neuron_stats

    def _count_dormant_neurons_in_layer(self, param: torch.Tensor, layer_name: str) -> int:
        """Count dormant neurons in a specific layer."""
        # Apply threshold to find dormant neurons
        dormant_mask = param.abs() < self._config.weight_threshold

        if param.dim() == 2:  # Linear layer: [out_features, in_features]
            # A neuron is dormant if ALL its incoming weights are below threshold
            dormant_neurons = dormant_mask.all(dim=1).sum().item()
        elif param.dim() == 4:  # Conv2d layer: [out_channels, in_channels, kernel_h, kernel_w]
            # A neuron (output channel) is dormant if ALL its weights are below threshold
            dormant_neurons = dormant_mask.view(param.size(0), -1).all(dim=1).sum().item()
        else:
            # For other dimensions, use a more general approach
            # Reshape to [output_neurons, input_weights_per_neuron]
            if param.dim() > 2:
                reshaped = param.view(param.size(0), -1)
                dormant_neurons = reshaped.abs().all(dim=1).sum().item()
            else:
                dormant_neurons = 0

        return dormant_neurons

    def get_dormant_neuron_history(self) -> Dict[str, list]:
        """Get the history of dormant neuron counts."""
        return self._dormant_neuron_history.copy()

    def get_layer_with_most_dormant_neurons(self) -> str:
        """Get the layer name with the highest dormant neuron ratio."""
        if not self._dormant_neuron_history:
            return ""

        max_ratio = 0.0
        worst_layer = ""

        for layer_name, history in self._dormant_neuron_history.items():
            if layer_name == "overall":
                continue
            if history:
                # Get the latest ratio
                latest_count = history[-1]
                # We'd need to track total neurons per layer to get accurate ratio
                # For now, just use the count
                if latest_count > max_ratio:
                    max_ratio = latest_count
                    worst_layer = layer_name

        return worst_layer
