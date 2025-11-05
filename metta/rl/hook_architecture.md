# Hook Architecture for Forward and Backward Pass Monitoring

## Overview

The hook system enables monitoring of neural network activations and gradients during training by attaching callbacks to
specific components in the policy network. The architecture uses a builder pattern that allows hooks to be registered
before training starts, with state managed through `ComponentContext`.

## Architecture Layers

### 1. PolicyAutoBuilder (Hook Registration)

`PolicyAutoBuilder` provides low-level hook registration methods that directly attach PyTorch hooks to module
components:

```python
# Forward hook registration
def register_component_hook_rule(
    self,
    *,
    component_name: str,
    hook: Callable[..., None],
) -> RemovableHandle:
    module = self.components.get(component_name)
    if module is None:
        raise KeyError(f"Component '{component_name}' not found in policy.")
    return module.register_forward_hook(hook)

# Backward hook registration
def register_component_backward_hook_rule(
    self,
    *,
    component_name: str,
    hook: Callable[..., None],
) -> RemovableHandle:
    module = self.components.get(component_name)
    if module is None:
        raise KeyError(f"Component '{component_name}' not found in policy.")
    return module.register_full_backward_hook(hook)
```

**Key Points:**

- `components` is an `OrderedDict[str, nn.Module]` mapping component names to their modules
- Forward hooks use `register_forward_hook()` - called after forward pass with `(module, input, output)`
- Backward hooks use `register_full_backward_hook()` - called during backward pass with
  `(module, grad_input, grad_output)`
- Returns `RemovableHandle` for cleanup

### 2. Hook Builder Pattern

Hook builders are factory functions that create hooks when called with a component name and context. They follow this
signature:

```python
# Forward hook builder
HookBuilder = Callable[
    [str, ComponentContext],
    Optional[Callable[[Any, tuple[Any, ...], Any], None]]
]

# Backward hook builder
BackwardHookBuilder = Callable[
    [str, ComponentContext],
    Optional[Callable[[Any, tuple[Any, ...], tuple[Any, ...]], None]]
]
```

**Hook Builder Structure:**

1. **State Management**: Retrieves or creates state objects from `context.model_metrics` dictionary
2. **Component Registration**: Registers the component with the state tracker
3. **Hook Closure**: Returns a closure that captures the tracker and component name
4. **Data Extraction**: Hook extracts relevant tensors from forward/backward pass
5. **State Update**: Hook calls tracker's `update()` method with extracted data

**Example (Forward Hook Builder):**

```python
def attach_relu_activation_hooks() -> "HookBuilder":
    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        # 1. Get or create state from context.model_metrics
        state = context.model_metrics.get("relu_activation_state")
        if not isinstance(state, ReLUActivationState):
            state = ReLUActivationState()
            context.model_metrics["relu_activation_state"] = state

        # 2. Register component
        if state.has_component(component_name):
            return None  # Already registered
        state.register_component(component_name)

        # 3. Return closure that captures state
        tracker = state
        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            tensor = extractor_fn(output)
            if tensor is None:
                return
            tracker.update(component_name, tensor.detach())
        return hook
    return builder
```

**Example (Backward Hook Builder):**

```python
def attach_gradient_flow_hooks(*, beta: float = 0.95) -> "BackwardHookBuilder":
    def builder(component_name: str, context: "ComponentContext") -> Optional[Callable[..., None]]:
        # 1. Get or create state
        state = context.model_metrics.get("gradient_flow_state")
        if not isinstance(state, GradientFlowState):
            state = GradientFlowState(beta=beta)
            context.model_metrics["gradient_flow_state"] = state

        # 2. Return closure for backward hook
        tracker = state
        def hook(module: nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
            # grad_output[0] is gradient w.r.t. module output
            if not grad_output or grad_output[0] is None:
                return
            g = grad_output[0].detach()
            # Compute per-unit gradient norms
            gn = g
            while gn.dim() > 2:
                gn = gn.flatten(2).mean(dim=2)
            per_unit = gn.pow(2).mean(dim=0).sqrt().cpu()
            tracker.update(component_name, per_unit)
        return hook
    return builder
```

### 3. TrainTool (Hook Management)

`TrainTool` manages hook registration lifecycle:

**Storage:**

```python
_training_hooks: list[HookSpec] = PrivateAttr(default_factory=list)
_training_backward_hooks: list[BackwardHookSpec] = PrivateAttr(default_factory=list)
_active_policy_hooks: list[RemovableHandle] = PrivateAttr(default_factory=list)
_active_policy_backward_hooks: list[RemovableHandle] = PrivateAttr(default_factory=list)
```

**Registration API:**

```python
def add_training_hook(
    self,
    component_name: str,
    hook_builder: HookBuilder,
) -> None:
    """Register a hook-producing builder for a specific component after initialization."""
    self._training_hooks.append((component_name, hook_builder))

def add_training_backward_hook(
    self,
    component_name: str,
    hook_builder: BackwardHookBuilder,
) -> None:
    """Register a backward hook-producing builder for a specific component after initialization."""
    self._training_backward_hooks.append((component_name, hook_builder))
```

**Hook Attachment (called during training setup):**

```python
def _register_policy_hooks(self, *, policy: Policy, trainer: Trainer) -> None:
    self._clear_policy_hooks()
    if not isinstance(policy, PolicyAutoBuilder):
        return

    # Register forward hooks
    for component_name, hook_builder in self._training_hooks:
        module = policy.components.get(component_name)
        if module is None:
            continue
        # Call builder with component name and context
        hook = hook_builder(component_name, trainer.context)
        if hook is None:
            continue
        # Attach hook to module via PolicyAutoBuilder
        handle = policy.register_component_hook_rule(
            component_name=component_name,
            hook=hook,
        )
        self._active_policy_hooks.append(handle)

    # Register backward hooks (same pattern)
    for component_name, hook_builder in self._training_backward_hooks:
        module = policy.components.get(component_name)
        if module is None:
            continue
        hook = hook_builder(component_name, trainer.context)
        if hook is None:
            continue
        handle = policy.register_component_backward_hook_rule(
            component_name=component_name,
            hook=hook,
        )
        self._active_policy_backward_hooks.append(handle)
```

**Cleanup:**

```python
def _clear_policy_hooks(self) -> None:
    if self._active_policy_hooks:
        for handle in self._active_policy_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._active_policy_hooks.clear()
    if self._active_policy_backward_hooks:
        for handle in self._active_policy_backward_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._active_policy_backward_hooks.clear()
```

### 4. ComponentContext (State Storage)

`ComponentContext` provides a `model_metrics` dictionary for storing runtime state objects:

```python
class ComponentContext:
    def __init__(self, ...):
        # ...
        self.model_metrics: Dict[str, Any] = {}
        """Dictionary storing model analysis state objects."""
```

**State objects stored in `model_metrics`:**

- `"relu_activation_state"` → `ReLUActivationState`
- `"fisher_information_state"` → `FisherInformationState`
- `"gradient_flow_state"` → `GradientFlowState`
- `"relu_gradient_flow_state"` → `ReLUGradientFlowState`
- `"saturated_activation_state_tanh"` → `SaturatedActivationState`
- `"saturated_activation_state_sigmoid"` → `SaturatedActivationState`

## Hook Execution Flow

### Forward Hooks

1. **Registration**: `TrainTool.add_training_hook()` stores `(component_name, hook_builder)`
2. **Attachment**: `_register_policy_hooks()` calls builder with `(component_name, trainer.context)`
3. **Hook Creation**: Builder returns closure that captures state tracker
4. **PyTorch Registration**: `PolicyAutoBuilder.register_component_hook_rule()` attaches to module
5. **Execution**: During forward pass, PyTorch calls hook with `(module, input, output)`
6. **Data Extraction**: Hook extracts tensor from output (via extractor function)
7. **State Update**: Hook calls `tracker.update(component_name, tensor)`
8. **Metrics Collection**: `StatsReporter` reads state and converts to metrics

### Backward Hooks

1. **Registration**: `TrainTool.add_training_backward_hook()` stores `(component_name, hook_builder)`
2. **Attachment**: Same pattern as forward hooks
3. **PyTorch Registration**: Uses `register_full_backward_hook()` instead
4. **Execution**: During backward pass, PyTorch calls hook with `(module, grad_input, grad_output)`
5. **Gradient Access**: Hook accesses `grad_output[0]` (gradient w.r.t. output) or `grad_input[0]` (gradient w.r.t.
   input)
6. **Computation**: Hook computes metrics (e.g., gradient norms, Fisher information)
7. **State Update**: Hook calls `tracker.update()` with computed values
8. **Metrics Collection**: Same as forward hooks

## Setup in TrainTool

### Example Usage

```python
from metta.rl.model_analysis import (
    attach_fisher_information_hooks,
    attach_gradient_flow_hooks,
    attach_relu_activation_hooks,
    attach_relu_gradient_flow_hooks,
    attach_saturated_activation_hooks,
)
from metta.tools.train import TrainTool

def train() -> TrainTool:
    cfg = TrainTool(...)

    # Forward hooks (called after forward pass)
    cfg.add_training_hook("actor_mlp", attach_relu_activation_hooks())
    cfg.add_training_hook("critic_head", attach_saturated_activation_hooks(activation="tanh"))

    # Backward hooks (called during backward pass)
    cfg.add_training_backward_hook("actor_mlp", attach_fisher_information_hooks())
    cfg.add_training_backward_hook("actor_mlp", attach_gradient_flow_hooks())
    cfg.add_training_backward_hook("actor_mlp", attach_relu_gradient_flow_hooks())

    return cfg
```

### Execution Timeline

1. **Config Creation**: User creates `TrainTool` and calls `add_training_hook()` / `add_training_backward_hook()`
   - Hooks are stored as `(component_name, hook_builder)` tuples
   - No hooks are attached yet

2. **Training Setup**: `TrainTool.invoke()` is called
   - Creates policy via `_load_or_create_policy()`
   - Creates trainer via `_initialize_trainer()`
   - Calls `_register_policy_hooks(policy=policy, trainer=trainer)`

3. **Hook Attachment**: `_register_policy_hooks()` executes
   - Iterates through stored hook builders
   - Calls each builder with `(component_name, trainer.context)`
   - Builder returns actual hook function (or None)
   - Hook is attached to module via `PolicyAutoBuilder` methods
   - `RemovableHandle` is stored for cleanup

4. **Training Loop**: Hooks execute during forward/backward passes
   - Forward hooks: called after each module's forward pass
   - Backward hooks: called during backward pass when gradients flow
   - State objects accumulate data in `context.model_metrics`

5. **Metrics Collection**: `StatsReporter.on_epoch_end()` collects metrics
   - Reads state objects from `context.model_metrics`
   - Calls `state.to_metrics()` to convert to dictionary
   - Calls `state.reset()` to clear for next epoch
   - Logs metrics to wandb

6. **Cleanup**: `_clear_policy_hooks()` removes all hooks
   - Called in `finally` block of `invoke()`
   - Removes all `RemovableHandle` objects

## Key Design Decisions

1. **Builder Pattern**: Hooks are created lazily when trainer is initialized, allowing access to `ComponentContext`
2. **State in Context**: All state objects stored in `context.model_metrics` dictionary, not as arbitrary attributes
3. **Component-Based**: Hooks target specific named components, not the entire policy
4. **Epoch Reset**: State objects reset every epoch (intentional - provides sufficient within-epoch smoothing)
5. **Type Safety**: Hook builders have explicit type signatures for forward vs backward hooks
6. **No Checkpointing**: State objects are not persisted (they reset every epoch anyway)
