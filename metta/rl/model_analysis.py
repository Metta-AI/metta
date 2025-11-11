"""Utilities for analyzing policy network structure.

This module provides several metrics for monitoring neural network health during training:

**Dead Neuron Statistics** (`compute_dead_neuron_stats`): Identifies neurons with very small
outgoing weights, indicating they may not be contributing to the network's output. Dead neurons
are detected by computing the mean absolute weight magnitude for each neuron's outgoing connections.
Neurons below a threshold (default 1e-6) are considered dead. High dead neuron counts can indicate
over-regularization, vanishing gradients, or insufficient capacity.

**ReLU Activation Statistics** (`ReLUActivationState`): Tracks how frequently each ReLU neuron
activates (outputs > 0) during forward passes. This is measured as the fraction of samples where
each neuron fires. Low activation frequencies suggest neurons that are rarely used, which can
indicate redundancy or dead neurons. Very high frequencies (>0.95) suggest neurons that fire
almost always, which may indicate they're not learning useful nonlinearities.

**Saturated Activation Statistics** (`SaturatedActivationState`): Monitors saturation in smooth
activation functions (tanh, sigmoid) where neurons output values near the activation boundaries.
For tanh, saturation occurs when abs(output) > threshold (default 0.95). For sigmoid, saturation
occurs when output is close to 0 or 1. Saturated neurons have near-zero gradients, making them
difficult to train and potentially indicating vanishing gradient problems or poor initialization.

**Fisher Information Metrics** (`FisherInformationState`): Uses an exponentially weighted moving
average (EMA) of squared gradients as a proxy for the Fisher Information Matrix. This provides
a measure of parameter importance - parameters with consistently large gradients contribute more
to learning. The mean importance metric helps identify unused or dead parameters (importance < 1e-8)
that may be candidates for pruning. The EMA smooths gradient noise within each epoch while
maintaining sensitivity to recent changes.

**Gradient Flow Metrics** (`GradientFlowState`): Tracks the flow of gradients through the network
by computing per-unit gradient norms (L2 norm of gradients w.r.t. activations). Uses EMA to smooth
noise and identify units with persistently near-zero gradients, which indicates no learning signal
reaches those units. This helps diagnose vanishing gradient problems and identify layers where
gradients are blocked or attenuated.

**ReLU Gradient Flow** (`ReLUGradientFlowState`): Tracks how often gradients pass through ReLU
activations (f'=1 when input > 0). Measures the fraction of backward passes where each ReLU unit
receives a non-zero gradient, indicating whether learning signals propagate through that unit.
Low gradient flow rates suggest dead ReLUs or vanishing gradients at that layer.

**Usage**: To enable these metrics in a training run, attach the appropriate hooks to your
`TrainTool` instance. For forward hooks (ReLU activations, saturated activations), use
`add_training_hook()`. For backward hooks (Fisher information), use `add_training_backward_hook()`.
Example::

    from metta.rl.model_analysis import (
        attach_fisher_information_hooks,
        attach_relu_activation_hooks,
        attach_saturated_activation_hooks,
    )
    from metta.tools.train import TrainTool

    cfg = TrainTool(...)
    cfg.add_training_hook("actor_mlp", attach_relu_activation_hooks())
    cfg.add_training_backward_hook("actor_mlp", attach_fisher_information_hooks())
    cfg.add_training_hook("critic_head", attach_saturated_activation_hooks(activation="tanh"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from metta.rl.training import ComponentContext
    from metta.tools.train import BackwardHookBuilder, HookBuilder

__all__ = [
    "FisherInformationState",
    "GradientFlowState",
    "ReLUActivationState",
    "ReLUGradientFlowState",
    "SaturatedActivationState",
    "attach_fisher_information_hooks",
    "attach_gradient_flow_hooks",
    "attach_relu_activation_hooks",
    "attach_relu_gradient_flow_hooks",
    "attach_saturated_activation_hooks",
    "compute_dead_neuron_stats",
    "compute_saturated_activation_stats",
    "get_fisher_information_metrics",
    "get_gradient_flow_metrics",
    "get_relu_activation_metrics",
    "get_relu_gradient_flow_metrics",
    "get_saturated_activation_metrics",
]


def compute_dead_neuron_stats(policy: nn.Module, *, threshold: float = 1e-6) -> Dict[str, float]:
    """Compute dead neuron statistics for a policy.

    Dead neurons are output units in linear layers whose mean absolute outgoing weight magnitude
    falls below a threshold (default 1e-6). This metric helps identify neurons that have become
    effectively inactive during training, which can indicate over-regularization, vanishing
    gradients, or insufficient model capacity. The statistics include per-layer dead neuron
    fractions and aggregate counts across the entire network. High dead neuron rates (>50%) may
    suggest the network is underutilizing its capacity, while very low rates (<1%) indicate
    healthy neuron utilization.

    Args:
        policy: The neural network policy to analyze.
        threshold: Minimum mean absolute weight magnitude to consider a neuron active (default 1e-6).

    Returns:
        Dictionary of metrics with keys like:
        - ``weights/dead_neurons/{layer_name}``: Fraction of dead neurons per layer
        - ``weights/dead_neurons/total_neurons``: Total number of neurons analyzed
        - ``weights/dead_neurons/dead_neurons``: Total number of dead neurons
        - ``weights/dead_neurons/fraction``: Overall fraction of dead neurons
        - ``weights/dead_neurons/active_fraction``: Overall fraction of active neurons
    """

    if not isinstance(policy, nn.Module):
        return {}

    total_neurons = 0
    dead_neurons = 0
    stats: Dict[str, float] = {}

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue

        weight_tensor = torch.nan_to_num(weight.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        # Use mean absolute weight per output unit to measure activity.
        activity = weight_tensor.abs().mean(dim=1)
        if activity.numel() == 0:
            continue

        layer_total = int(activity.numel())
        layer_dead = int((activity < threshold).sum().item())
        if layer_total == 0:
            continue

        total_neurons += layer_total
        dead_neurons += layer_dead

        sanitized = name.replace(".", "/") if name else "root"
        stats[f"weights/dead_neurons/{sanitized}"] = layer_dead / layer_total

    if total_neurons == 0:
        return {}

    dead_fraction = dead_neurons / total_neurons
    stats["weights/dead_neurons/total_neurons"] = float(total_neurons)
    stats["weights/dead_neurons/dead_neurons"] = float(dead_neurons)
    stats["weights/dead_neurons/fraction"] = dead_fraction
    stats["weights/dead_neurons/active_fraction"] = 1.0 - dead_fraction

    return stats


class ReLUActivationState:
    """State container tracking ReLU activation frequencies per component.

    This class monitors how frequently each ReLU neuron activates (outputs a value > 0) during
    training. The activation frequency is computed as the fraction of samples where each neuron
    fires across forward passes within an epoch. Neurons with very low activation frequencies
    (<5%) are rarely used and may indicate redundancy or dead neurons. Neurons with very high
    frequencies (>95%) fire almost always, suggesting they're not learning useful nonlinearities
    and may be effectively linear. Healthy networks typically show a broad distribution of
    activation frequencies, with most neurons firing on 20-80% of samples.
    """

    def __init__(self) -> None:
        self.active: dict[str, list[int]] = {}
        self.totals: dict[str, list[int]] = {}

    def has_component(self, component_name: str) -> bool:
        return component_name in self.active

    def register_component(self, component_name: str) -> None:
        self.active.setdefault(component_name, [])
        self.totals.setdefault(component_name, [])

    def update(self, component_name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            flat = tensor.unsqueeze(0)
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])

        counts = (flat > 0).sum(dim=0).tolist()
        if not counts:
            return
        total = int(flat.shape[0])

        active = self.active.setdefault(component_name, [0] * len(counts))
        totals = self.totals.setdefault(component_name, [0] * len(counts))

        if len(active) < len(counts):
            active.extend([0] * (len(counts) - len(active)))
            totals.extend([0] * (len(counts) - len(totals)))

        for idx, value in enumerate(counts):
            if idx >= len(active):
                active.append(0)
                totals.append(0)
            active[idx] += int(value)
            totals[idx] += total

    def to_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for component, active_values in self.active.items():
            component_totals = self.totals.get(component, [])
            for idx, (act, tot) in enumerate(zip(active_values, component_totals, strict=False)):
                if tot:
                    metrics[f"activations/relu/{component}/{idx}"] = act / tot
        return metrics

    def reset(self) -> None:
        for component, values in self.active.items():
            self.active[component] = [0] * len(values)
        for component, values in self.totals.items():
            self.totals[component] = [0] * len(values)


class SaturatedActivationState:
    """State container tracking saturated neuron statistics for smooth activations (tanh, sigmoid).

    Saturation occurs when activation outputs are near the function boundaries, resulting in
    near-zero gradients. For tanh activations, saturation happens when abs(output) > threshold
    (default 0.95, meaning outputs near ±1). For sigmoid, saturation occurs when outputs are
    close to 0 or 1. High saturation rates (>50%) indicate that many neurons are operating in
    regions with vanishing gradients, making learning difficult and potentially causing training
    instabilities. This metric helps diagnose gradient flow issues and can suggest the need for
    better initialization, gradient clipping, or alternative activation functions.
    """

    def __init__(self, activation: str, saturation_threshold: float = 0.95) -> None:
        """
        Args:
            activation: Activation type, either "tanh" or "sigmoid"
            saturation_threshold: Threshold for considering a neuron saturated.
                For tanh: abs(output) > threshold (default 0.95)
                For sigmoid: output < (1 - threshold) or output > threshold (default 0.95)
        """
        if activation not in {"tanh", "sigmoid"}:
            raise ValueError(f"Unsupported activation '{activation}'. Expected 'tanh' or 'sigmoid'.")
        self.activation = activation
        self.saturation_threshold = saturation_threshold
        self.saturated: dict[str, list[int]] = {}
        self.totals: dict[str, list[int]] = {}

    def has_component(self, component_name: str) -> bool:
        return component_name in self.saturated

    def register_component(self, component_name: str) -> None:
        self.saturated.setdefault(component_name, [])
        self.totals.setdefault(component_name, [])

    def update(self, component_name: str, tensor: torch.Tensor) -> None:
        """Update saturation statistics from activation output tensor."""
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            flat = tensor.unsqueeze(0)
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])

        if flat.numel() == 0:
            return

        # Compute saturation mask
        if self.activation == "tanh":
            # For tanh: saturated when abs(output) is close to 1.0
            saturated_mask = flat.abs() > self.saturation_threshold
        else:  # sigmoid
            # For sigmoid: saturated when output is close to 0 or 1
            saturated_mask = (flat < (1.0 - self.saturation_threshold)) | (flat > self.saturation_threshold)

        counts = saturated_mask.sum(dim=0).tolist()
        if not counts:
            return
        total = int(flat.shape[0])

        saturated = self.saturated.setdefault(component_name, [0] * len(counts))
        totals = self.totals.setdefault(component_name, [0] * len(counts))

        if len(saturated) < len(counts):
            saturated.extend([0] * (len(counts) - len(saturated)))
            totals.extend([0] * (len(counts) - len(totals)))

        for idx, value in enumerate(counts):
            if idx >= len(saturated):
                saturated.append(0)
                totals.append(0)
            saturated[idx] += int(value)
            totals[idx] += total

    def to_metrics(self) -> Dict[str, float]:
        """Convert accumulated saturation statistics to metrics."""
        metrics: Dict[str, float] = {}
        total_saturated = 0
        total_neurons = 0

        for component, saturated_values in self.saturated.items():
            component_totals = self.totals.get(component, [])
            component_saturated = 0
            component_total = 0

            for idx, (sat, tot) in enumerate(zip(saturated_values, component_totals, strict=False)):
                if tot:
                    fraction = sat / tot
                    metrics[f"activations/{self.activation}/saturated_fraction/{component}/{idx}"] = fraction
                    component_saturated += sat
                    component_total += tot

            if component_total > 0:
                metrics[f"activations/{self.activation}/saturation/{component}/total_neurons"] = float(component_total)
                metrics[f"activations/{self.activation}/saturation/{component}/saturated_neurons"] = float(
                    component_saturated
                )
                metrics[f"activations/{self.activation}/saturation/{component}/fraction"] = float(
                    component_saturated / component_total
                )
                metrics[f"activations/{self.activation}/saturation/{component}/active_fraction"] = float(
                    1.0 - (component_saturated / component_total)
                )
                total_saturated += component_saturated
                total_neurons += component_total

        if total_neurons > 0:
            prefix = f"activations/{self.activation}/saturation"
            metrics[f"{prefix}/total_neurons"] = float(total_neurons)
            metrics[f"{prefix}/saturated_neurons"] = float(total_saturated)
            metrics[f"{prefix}/fraction"] = float(total_saturated / total_neurons)
            metrics[f"{prefix}/active_fraction"] = float(1.0 - (total_saturated / total_neurons))

        return metrics

    def reset(self) -> None:
        """Reset accumulated saturation statistics."""
        for component, values in self.saturated.items():
            self.saturated[component] = [0] * len(values)
        for component, values in self.totals.items():
            self.totals[component] = [0] * len(values)


class FisherInformationState:
    """State container tracking Fisher Information Matrix (EMA of squared gradients) per component.

    Uses an exponentially weighted moving average (EMA) of squared gradients as a proxy for the
    Fisher Information Matrix diagonal. The Fisher Information captures parameter importance by
    measuring how much each parameter's gradients contribute to the loss. Parameters with
    consistently large squared gradients (high Fisher information) are more important for learning,
    while parameters with consistently small gradients (low Fisher information, < 1e-8) may be
    unused or dead. This metric helps identify parameters that can potentially be pruned without
    affecting model performance. The EMA smooths gradient noise within each epoch (default beta=0.95),
    providing stable importance estimates while maintaining sensitivity to recent changes. State
    resets at epoch end, providing sufficient within-epoch smoothing for dead parameter detection.
    """

    def __init__(self, beta: float = 0.95) -> None:
        """Initialize Fisher Information state with EMA decay factor.

        Args:
            beta: EMA decay factor (default 0.95). Higher values give more weight to historical gradients
                within the epoch. Note: State resets at epoch end, so this only affects within-epoch smoothing.
        """
        self.beta = beta
        self.fisher_ema: dict[str, dict[str, torch.Tensor]] = {}
        self.unused_threshold: float = 1e-8

    def has_component(self, component_name: str) -> bool:
        return component_name in self.fisher_ema

    def register_component(self, component_name: str) -> None:
        self.fisher_ema.setdefault(component_name, {})

    def update(self, component_name: str, param_name: str, grad: torch.Tensor) -> None:
        """Update EMA of squared gradients for Fisher Information Matrix proxy."""
        if grad is None:
            return
        grad = grad.detach()
        squared_grad = grad.pow(2)

        component_fim = self.fisher_ema.setdefault(component_name, {})
        if param_name not in component_fim:
            component_fim[param_name] = torch.zeros_like(squared_grad)

        # EMA update: F = beta * F + (1 - beta) * grad^2
        with torch.no_grad():
            component_fim[param_name].mul_(self.beta).add_((1.0 - self.beta) * squared_grad)

    def to_metrics(self) -> Dict[str, float]:
        """Convert EMA Fisher Information to metrics, including dead parameter detection."""
        metrics: Dict[str, float] = {}
        dead_params: list[str] = []

        for component, param_fim in self.fisher_ema.items():
            for param_name, fim_ema in param_fim.items():
                # Mean importance (proxy for Fisher information)
                importance = fim_ema.mean().item()
                metrics[f"fisher_info/{component}/{param_name}/mean"] = importance
                metrics[f"fisher_info/{component}/{param_name}/trace"] = float(fim_ema.sum().item())
                metrics[f"fisher_info/{component}/{param_name}/max"] = float(fim_ema.max().item())

                # Detect possibly unused/dead parameters
                if importance < self.unused_threshold:
                    dead_params.append(f"{component}/{param_name}")
                    metrics[f"fisher_info/{component}/{param_name}/dead"] = 1.0
                else:
                    metrics[f"fisher_info/{component}/{param_name}/dead"] = 0.0

        # Aggregate dead parameter counts per component
        if dead_params:
            component_dead_counts: dict[str, int] = {}
            for param_path in dead_params:
                component = param_path.split("/", 1)[0]
                component_dead_counts[component] = component_dead_counts.get(component, 0) + 1

            for component, count in component_dead_counts.items():
                total_params = len(self.fisher_ema.get(component, {}))
                if total_params > 0:
                    metrics[f"fisher_info/{component}/dead_fraction"] = count / total_params
                    metrics[f"fisher_info/{component}/dead_count"] = float(count)

        return metrics

    def reset(self) -> None:
        """Reset Fisher Information EMA (clears all tracked parameters).

        Called at epoch end to start fresh for the next epoch. This is intentional:
        within-epoch EMA provides sufficient smoothing for dead parameter detection.
        """
        self.fisher_ema.clear()


class GradientFlowState:
    """State container tracking gradient flow (per-unit gradient norms) per component.

    Tracks the L2 norm of gradients w.r.t. activations for each unit, using EMA to smooth noise.
    Units with persistently near-zero gradient norms indicate that no learning signal reaches them,
    which can indicate vanishing gradients, dead units, or gradient blocking. This metric helps
    diagnose gradient flow issues throughout the network.
    """

    def __init__(self, beta: float = 0.95) -> None:
        """Initialize gradient flow state with EMA decay factor.

        Args:
            beta: EMA decay factor (default 0.95). Higher values give more weight to historical gradients
                within the epoch.
        """
        self.beta = beta
        self.grad_norm_ema: dict[str, torch.Tensor] = {}
        self.update_count: dict[str, int] = {}

    def has_component(self, component_name: str) -> bool:
        return component_name in self.grad_norm_ema

    def register_component(self, component_name: str, num_units: int) -> None:
        """Register a component with known number of units."""
        if component_name not in self.grad_norm_ema:
            # Initialize with zeros on CPU (will be moved to device when first gradient arrives)
            self.grad_norm_ema[component_name] = torch.zeros(num_units, dtype=torch.float32)
            self.update_count[component_name] = 0

    def update(self, component_name: str, grad_norm: torch.Tensor) -> None:
        """Update EMA of per-unit gradient norms."""
        if grad_norm.numel() == 0:
            return

        grad_norm = grad_norm.detach().cpu().float()

        if component_name not in self.grad_norm_ema:
            self.grad_norm_ema[component_name] = torch.zeros_like(grad_norm)
            self.update_count[component_name] = 0

        # EMA update: G = beta * G + (1 - beta) * grad_norm
        with torch.no_grad():
            self.grad_norm_ema[component_name].mul_(self.beta).add_((1.0 - self.beta) * grad_norm)
            self.update_count[component_name] += 1

    def to_metrics(self) -> Dict[str, float]:
        """Convert gradient flow EMA to metrics."""
        metrics: Dict[str, float] = {}
        for component, grad_ema in self.grad_norm_ema.items():
            count = self.update_count.get(component, 1)
            if count == 0:
                continue

            # Mean gradient norm per unit
            mean_grad = grad_ema.mean().item()
            metrics[f"gradient_flow/{component}/mean"] = mean_grad

            # Per-unit gradient norms (for units with significant gradients)
            for idx, grad_val in enumerate(grad_ema):
                if grad_val.item() > 1e-8:  # Only log non-trivial gradients
                    metrics[f"gradient_flow/{component}/unit_{idx}"] = grad_val.item()

            # Fraction of units with near-zero gradients
            near_zero = (grad_ema < 1e-8).sum().item()
            total_units = grad_ema.numel()
            if total_units > 0:
                metrics[f"gradient_flow/{component}/blocked_fraction"] = near_zero / total_units
                metrics[f"gradient_flow/{component}/active_fraction"] = 1.0 - (near_zero / total_units)

        return metrics

    def reset(self) -> None:
        """Reset gradient flow EMA (clears all tracked components)."""
        self.grad_norm_ema.clear()
        self.update_count.clear()


class ReLUGradientFlowState:
    """State container tracking how often gradients pass through ReLU activations.

    For ReLU, gradients pass through when the input was > 0 (f'=1). This tracks the fraction
    of backward passes where each ReLU unit receives a non-zero gradient, indicating whether
    learning signals propagate through that unit. Low gradient flow rates suggest dead ReLUs
    or vanishing gradients at that layer.
    """

    def __init__(self) -> None:
        self.gradient_passed: dict[str, list[int]] = {}
        self.totals: dict[str, list[int]] = {}

    def has_component(self, component_name: str) -> bool:
        return component_name in self.gradient_passed

    def register_component(self, component_name: str) -> None:
        self.gradient_passed.setdefault(component_name, [])
        self.totals.setdefault(component_name, [])

    def update(self, component_name: str, grad_input: torch.Tensor) -> None:
        """Update gradient flow statistics from gradient w.r.t. input.

        For ReLU, grad_input is non-zero only when input > 0 (gradient passes through).
        """
        if grad_input.ndim == 0:
            return
        if grad_input.ndim == 1:
            flat = grad_input.unsqueeze(0)
        else:
            flat = grad_input.reshape(-1, grad_input.shape[-1])

        # Count units with non-zero gradients (gradient passed through)
        passed_mask = flat.abs() > 1e-8
        counts = passed_mask.sum(dim=0).tolist()
        if not counts:
            return
        total = int(flat.shape[0])

        gradient_passed = self.gradient_passed.setdefault(component_name, [0] * len(counts))
        totals = self.totals.setdefault(component_name, [0] * len(counts))

        if len(gradient_passed) < len(counts):
            gradient_passed.extend([0] * (len(counts) - len(gradient_passed)))
            totals.extend([0] * (len(counts) - len(totals)))

        for idx, value in enumerate(counts):
            if idx >= len(gradient_passed):
                gradient_passed.append(0)
                totals.append(0)
            gradient_passed[idx] += int(value)
            totals[idx] += total

    def to_metrics(self) -> Dict[str, float]:
        """Convert gradient flow statistics to metrics."""
        metrics: Dict[str, float] = {}
        for component, passed_values in self.gradient_passed.items():
            component_totals = self.totals.get(component, [])
            for idx, (passed, tot) in enumerate(zip(passed_values, component_totals, strict=False)):
                if tot:
                    metrics[f"gradient_flow/relu/{component}/{idx}"] = passed / tot
        return metrics

    def reset(self) -> None:
        """Reset gradient flow statistics."""
        for component, values in self.gradient_passed.items():
            self.gradient_passed[component] = [0] * len(values)
        for component, values in self.totals.items():
            self.totals[component] = [0] * len(values)


def attach_saturated_activation_hooks(
    *,
    activation: str = "tanh",
    saturation_threshold: float = 0.95,
    extractor: Any = "critic_1",
) -> "HookBuilder":
    """Create a builder that prepares saturated activation tracking for smooth activations (tanh/sigmoid).

    Args:
        activation: Activation type, either "tanh" or "sigmoid"
        saturation_threshold: Threshold for considering a neuron saturated (default 0.95)
        extractor: Function or key to extract activation tensor from module output

    Returns:
        HookBuilder that can be registered with add_training_hook()
    """
    if activation not in {"tanh", "sigmoid"}:
        raise ValueError(f"Unsupported activation '{activation}'. Expected 'tanh' or 'sigmoid'.")

    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        state_key = f"saturated_activation_state_{activation}"
        state = context.model_metrics.get(state_key)
        if not isinstance(state, SaturatedActivationState):
            state = SaturatedActivationState(activation=activation, saturation_threshold=saturation_threshold)
            context.model_metrics[state_key] = state

        if state.has_component(component_name):
            return None

        extractor_fn = _build_extractor(extractor)
        if extractor_fn is None:
            return None

        tracker = state
        tracker.register_component(component_name)

        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            tensor = extractor_fn(output)
            if tensor is None:
                return
            tracker.update(component_name, tensor.detach())

        return hook

    return builder


def attach_relu_activation_hooks(
    *,
    extractor: Any = "actor_hidden",
) -> "HookBuilder":
    """Create a builder that prepares ReLU activation tracking for a given component."""

    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        state = context.model_metrics.get("relu_activation_state")
        if not isinstance(state, ReLUActivationState):
            state = ReLUActivationState()
            context.model_metrics["relu_activation_state"] = state

        if state.has_component(component_name):
            return None

        extractor_fn = _build_extractor(extractor)
        if extractor_fn is None:
            return None

        tracker = state
        tracker.register_component(component_name)

        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            tensor = extractor_fn(output)
            if tensor is None:
                return
            tracker.update(component_name, tensor.detach())

        return hook

    return builder


def get_relu_activation_metrics(state: Any, *, reset: bool = False) -> Dict[str, float]:
    if not isinstance(state, ReLUActivationState):
        return {}
    metrics = state.to_metrics()
    if reset:
        state.reset()
    return metrics


def get_saturated_activation_metrics(state: Any, *, activation: str = "tanh", reset: bool = False) -> Dict[str, float]:
    """Get saturated activation metrics from trainer state."""
    if not isinstance(state, SaturatedActivationState):
        return {}
    if state.activation != activation:
        return {}
    metrics = state.to_metrics()
    if reset:
        state.reset()
    return metrics


def attach_fisher_information_hooks(*, beta: float = 0.95) -> "BackwardHookBuilder":
    """Create a builder that prepares Fisher Information tracking for a given component.

    Uses EMA (Exponentially Weighted Moving Average) of squared gradients as a proxy for Fisher Information.
    The EMA accumulates within each epoch and resets at epoch end, providing sufficient smoothing
    for dead parameter detection without needing cross-epoch persistence.

    Args:
        beta: EMA decay factor (default 0.95). Higher values give more weight to historical gradients
            within the epoch.

    Returns:
        BackwardHookBuilder that can be registered with add_training_backward_hook()
    """

    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        state = context.model_metrics.get("fisher_information_state")
        if not isinstance(state, FisherInformationState):
            state = FisherInformationState(beta=beta)
            context.model_metrics["fisher_information_state"] = state

        if state.has_component(component_name):
            return None

        tracker = state
        tracker.register_component(component_name)

        def hook(module: nn.Module, _grad_input: tuple[Any, ...], _grad_output: tuple[Any, ...]) -> None:
            # Update EMA of squared gradients for all parameters in the module
            with torch.no_grad():
                for param_name, param in module.named_parameters(recurse=False):
                    if param.grad is not None:
                        full_param_name = f"{component_name}.{param_name}"
                        tracker.update(component_name, full_param_name, param.grad)

        return hook

    return builder


def attach_gradient_flow_hooks(*, beta: float = 0.95) -> "BackwardHookBuilder":
    """Create a builder that prepares gradient flow tracking for a given component.

    Tracks per-unit gradient norms (L2 norm of gradients w.r.t. activations) using EMA.
    Units with persistently near-zero gradient norms indicate no learning signal reaches them.

    Args:
        beta: EMA decay factor (default 0.95). Higher values give more weight to historical gradients
            within the epoch.

    Returns:
        BackwardHookBuilder that can be registered with add_training_backward_hook()
    """

    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        state = context.model_metrics.get("gradient_flow_state")
        if not isinstance(state, GradientFlowState):
            state = GradientFlowState(beta=beta)
            context.model_metrics["gradient_flow_state"] = state

        if state.has_component(component_name):
            return None

        tracker = state

        def hook(module: nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
            # grad_output[0] is gradient w.r.t. module output
            if not grad_output or grad_output[0] is None:
                return

            g = grad_output[0].detach()
            with torch.no_grad():
                # Reduce to 2D: (batch, features) by averaging over spatial dimensions
                gn = g
                while gn.dim() > 2:
                    gn = gn.flatten(2).mean(dim=2)

                # Compute mean L2 norm per unit: sqrt(mean(g^2)) per unit
                # Shape: (batch, units) -> (units,)
                per_unit = gn.pow(2).mean(dim=0).sqrt().cpu()

                # Register component on first update if needed
                if not tracker.has_component(component_name):
                    tracker.register_component(component_name, num_units=per_unit.numel())

                tracker.update(component_name, per_unit)

        return hook

    return builder


def attach_relu_gradient_flow_hooks() -> "BackwardHookBuilder":
    """Create a builder that tracks how often gradients pass through ReLU activations.

    For ReLU, gradients pass through when input > 0 (f'=1). This tracks the fraction
    of backward passes where each ReLU unit receives a non-zero gradient.

    Returns:
        BackwardHookBuilder that can be registered with add_training_backward_hook()
    """

    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        state = context.model_metrics.get("relu_gradient_flow_state")
        if not isinstance(state, ReLUGradientFlowState):
            state = ReLUGradientFlowState()
            context.model_metrics["relu_gradient_flow_state"] = state

        if state.has_component(component_name):
            return None

        tracker = state
        tracker.register_component(component_name)

        def hook(module: nn.Module, grad_input: tuple[Any, ...], _grad_output: tuple[Any, ...]) -> None:
            # grad_input[0] is gradient w.r.t. module input
            # For ReLU, this is non-zero only when input > 0 (gradient passes through)
            if not grad_input or grad_input[0] is None:
                return

            tracker.update(component_name, grad_input[0].detach())

        return hook

    return builder


def get_fisher_information_metrics(state: Any, *, reset: bool = False) -> Dict[str, float]:
    """Get Fisher Information metrics from trainer state."""
    if not isinstance(state, FisherInformationState):
        return {}
    metrics = state.to_metrics()
    if reset:
        state.reset()
    return metrics


def get_gradient_flow_metrics(state: Any, *, reset: bool = False) -> Dict[str, float]:
    """Get gradient flow metrics from trainer state."""
    if not isinstance(state, GradientFlowState):
        return {}
    metrics = state.to_metrics()
    if reset:
        state.reset()
    return metrics


def get_relu_gradient_flow_metrics(state: Any, *, reset: bool = False) -> Dict[str, float]:
    """Get ReLU gradient flow metrics from trainer state."""
    if not isinstance(state, ReLUGradientFlowState):
        return {}
    metrics = state.to_metrics()
    if reset:
        state.reset()
    return metrics


def _build_extractor(spec: Any) -> Optional[Callable[[Any], Optional[torch.Tensor]]]:
    if callable(spec):
        return spec

    if isinstance(spec, str):
        key = spec

        def extractor(output: Any) -> Optional[torch.Tensor]:
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, TensorDictBase):
                try:
                    tensor = output.get(key)
                except KeyError:
                    return None
                return tensor if isinstance(tensor, torch.Tensor) else None
            if isinstance(output, dict):
                tensor = output.get(key)
                return tensor if isinstance(tensor, torch.Tensor) else None
            return None

        return extractor

    return None


def compute_saturated_activation_stats(
    policy: nn.Module,
    *,
    activation: str,
    derivative_threshold: float = 1e-3,
) -> Dict[str, float]:
    """Compute saturated neuron statistics for activations with bounded derivatives.

    This static analysis method evaluates saturation by computing the activation derivative
    at each neuron's bias value. For tanh, the derivative is 1 - tanh²(bias). For sigmoid,
    it's σ(bias) * (1 - σ(bias)). Neurons with derivative magnitudes below the threshold
    (default 1e-3) are considered saturated. Unlike the dynamic `SaturatedActivationState` which
    tracks saturation during training, this method provides a static snapshot based on the
    current weight values. It's useful for analyzing model initialization and understanding
    saturation patterns before training begins. High saturation at initialization suggests poor
    weight initialization that may lead to vanishing gradients.

    Args:
        policy: The neural network policy to analyze.
        activation: Activation type, either "tanh" or "sigmoid".
        derivative_threshold: Minimum derivative magnitude to consider a neuron active (default 1e-3).

    Returns:
        Dictionary of metrics with keys like:
        - ``activations/{activation}/saturated_fraction/{layer_name}``: Per-layer saturation fraction
        - ``activations/{activation}/saturation/total_neurons``: Total neurons analyzed
        - ``activations/{activation}/saturation/saturated_neurons``: Total saturated neurons
        - ``activations/{activation}/saturation/fraction``: Overall saturation fraction
        - ``activations/{activation}/saturation/active_fraction``: Overall active fraction
    """

    if not isinstance(policy, nn.Module):
        return {}

    supported = {"tanh", "sigmoid"}
    if activation not in supported:
        raise ValueError(f"Unsupported activation '{activation}'. Expected one of {sorted(supported)}.")

    total_neurons = 0
    saturated_neurons = 0
    stats: Dict[str, float] = {}

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        out_features = getattr(module, "out_features", None)
        if out_features is None:
            continue

        bias = getattr(module, "bias", None)
        if bias is None:
            device = module.weight.device if hasattr(module, "weight") else torch.device("cpu")
            bias_tensor = torch.zeros(out_features, device=device)
        else:
            if bias.ndim != 1 or bias.numel() != out_features:
                continue
            bias_tensor = bias.detach()

        bias_tensor = torch.nan_to_num(bias_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if activation == "tanh":
            derivative = 1.0 - torch.tanh(bias_tensor).pow(2)
        else:  # activation == "sigmoid"
            sigma = torch.sigmoid(bias_tensor)
            derivative = sigma * (1.0 - sigma)

        if derivative.numel() == 0:
            continue

        layer_total = int(derivative.numel())
        layer_saturated = int((derivative.abs() < derivative_threshold).sum().item())
        if layer_total == 0:
            continue

        total_neurons += layer_total
        saturated_neurons += layer_saturated

        sanitized = name.replace(".", "/") if name else "root"
        stats[f"activations/{activation}/saturated_fraction/{sanitized}"] = layer_saturated / layer_total

    if total_neurons == 0:
        return {}

    saturated_fraction = saturated_neurons / total_neurons
    prefix = f"activations/{activation}/saturation"
    stats[f"{prefix}/total_neurons"] = float(total_neurons)
    stats[f"{prefix}/saturated_neurons"] = float(saturated_neurons)
    stats[f"{prefix}/fraction"] = saturated_fraction
    stats[f"{prefix}/active_fraction"] = 1.0 - saturated_fraction

    return stats
