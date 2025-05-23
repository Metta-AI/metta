import torch
from tensordict import TensorDict

from .metta_moduly import MettaModule


class SafeModule(MettaModule):
    """Wraps computation with comprehensive safety checks."""

    def __init__(self, module: MettaModule, action_bounds=None, nan_check=True):
        """Initialize a safety wrapper around any MettaModule.

        Args:
            module (MettaModule): The module to wrap with safety checks
            action_bounds (tuple[float, float] | None): Min/max bounds for action outputs
            nan_check (bool): Whether to check for NaN/Inf values
        """
        # Inherit keys from wrapped module
        super().__init__(in_keys=module.in_keys, out_keys=module.out_keys)
        self.module = module
        self.action_bounds = action_bounds
        self.nan_check = nan_check

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Execute wrapped module with safety checks.

        Args:
            tensordict (TensorDict): Input data

        Returns:
            TensorDict: Output data with safety validations applied

        Raises:
            ValueError: If NaN/Inf detected or other safety violations
        """
        # Pre-execution validation
        if self.nan_check:
            self._validate_inputs(tensordict)

        # Execute wrapped module
        result = self.module(tensordict)

        # Post-execution validation and correction
        if self.nan_check:
            self._validate_outputs(result)

        if self.action_bounds:
            self._apply_action_bounds(result)

        return result

    def _validate_inputs(self, tensordict: TensorDict):
        """Input validation catching common failures."""
        for key in self.module.in_keys:
            if key in tensordict:
                tensor = tensordict[key]
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN detected in input '{key}'")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf detected in input '{key}'")

    def _validate_outputs(self, tensordict: TensorDict):
        """Output validation with RL-specific checks."""
        for key in self.module.out_keys:
            if key in tensordict:
                tensor = tensordict[key]
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN detected in output '{key}'")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf detected in output '{key}'")

    def _apply_action_bounds(self, tensordict: TensorDict):
        """Apply action bounds for RL policy networks."""
        if self.action_bounds is not None:
            min_action, max_action = self.action_bounds
            for key in self.module.out_keys:
                if key.endswith("_action") and key in tensordict:
                    # Clip action outputs to bounds
                    tensordict[key] = torch.clamp(tensordict[key], min_action, max_action)


class RegularizedModule(MettaModule):
    """Wraps computation with regularization, building on existing ParamLayer patterns."""

    def __init__(self, module: MettaModule, l2_scale=0.01, l1_scale=0.0):
        """Initialize regularization wrapper around any MettaModule.

        Args:
            module (MettaModule): The module to wrap with regularization
            l2_scale (float): L2 regularization weight
            l1_scale (float): L1 regularization weight
        """
        # Inherit keys from wrapped module
        super().__init__(in_keys=module.in_keys, out_keys=module.out_keys)
        self.module = module
        self.l2_scale = l2_scale
        self.l1_scale = l1_scale

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Execute wrapped module and add regularization losses.

        Args:
            tensordict (TensorDict): Input data

        Returns:
            TensorDict: Output data with regularization losses added
        """
        result = self.module(tensordict)

        # Add regularization losses to TensorDict during training
        if self.training and (self.l2_scale > 0 or self.l1_scale > 0):
            reg_loss = self._compute_regularization()
            if reg_loss is not None:
                # Expand scalar loss to match batch dimension
                batch_size = result.batch_size[0]
                reg_loss_batched = reg_loss.unsqueeze(0).expand(batch_size)

                if "regularization_loss" in result:
                    result["regularization_loss"] = result["regularization_loss"] + reg_loss_batched
                else:
                    result["regularization_loss"] = reg_loss_batched

        return result

    def _compute_regularization(self) -> torch.Tensor | None:
        """Compute L1/L2 regularization extending existing patterns."""
        # Check if module has any parameters
        try:
            first_param = next(self.module.parameters())
            device = first_param.device
        except StopIteration:
            # Module has no parameters, no regularization needed
            return None

        reg_loss = torch.tensor(0.0, device=device)

        for param in self.module.parameters():
            if param.requires_grad:
                if self.l2_scale > 0:
                    reg_loss += self.l2_scale * torch.sum(param**2)
                if self.l1_scale > 0:
                    reg_loss += self.l1_scale * torch.sum(torch.abs(param))

        return reg_loss


class WeightMonitoringModule(MettaModule):
    """Adds weight monitoring to any MettaModule."""

    def __init__(self, module: MettaModule, clip_value=None, monitor_health=True):
        """Initialize weight monitoring wrapper around any MettaModule.

        Args:
            module (MettaModule): The module to wrap with monitoring
            clip_value (float | None): Value to clip weights to (if provided)
            monitor_health (bool): Whether to monitor weight health
        """
        # Inherit keys from wrapped module
        super().__init__(in_keys=module.in_keys, out_keys=module.out_keys)
        self.module = module
        self.clip_value = clip_value
        self.monitor_health = monitor_health

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Execute wrapped module with weight monitoring.

        Args:
            tensordict (TensorDict): Input data

        Returns:
            TensorDict: Output data (monitoring happens as side effects)
        """
        result = self.module(tensordict)

        if self.training:
            if self.monitor_health:
                self._monitor_weight_health()

            if self.clip_value:
                self._clip_weights()

        return result

    def _monitor_weight_health(self):
        """Monitor weight health across all parameters."""
        for name, param in self.module.named_parameters():
            if param.data.dim() >= 2:  # Only for weight matrices
                # Dead neuron detection
                dead_neurons = (param.data.abs() < 1e-6).all(dim=1).sum()
                if dead_neurons > 0:
                    print(f"Warning: {dead_neurons} potentially dead neurons in {name}")

                # Weight norm monitoring
                weight_norm = param.data.norm()
                if weight_norm > 100.0:
                    print(f"Warning: Large weight norm in {name}: {weight_norm:.4f}")

    def _clip_weights(self):
        """Clip all weights to prevent gradient explosion."""
        if self.clip_value is not None:
            with torch.no_grad():
                for param in self.module.parameters():
                    if param.data.dim() >= 2:  # Only clip weight matrices
                        param.data.clamp_(-self.clip_value, self.clip_value)
