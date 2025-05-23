# Metta Architecture Refactoring Proposal

## Executive Summary

This proposal recommends replacing the current `LayerBase` architecture with a cleaner design inspired by proven TorchRL patterns. Rather than incrementally enhancing the existing monolithic approach, we propose a clean migration to separated computation and infrastructure concerns.

**Core Insight:** `LayerBase`'s monolithic design—combining computation, lifecycle management, DAG traversal, shape management, and dependency coordination—creates fundamental testing difficulties and architectural coupling that incremental fixes cannot address.

**Proposed Solution:** Replace `LayerBase` with pure computation modules (`MettaModule`) managed by a lightweight container, while preserving the valuable existing infrastructure: component registry, TensorDict communication, and Hydra configuration.

## Current Architecture Analysis

### Infrastructure Worth Preserving

The existing system has valuable infrastructure that should be retained:

- **Component Registry**: `MettaAgent.components` using `nn.ModuleDict` enables runtime component lookup and hotswapping
- **Indirect Execution**: Components called by name (`self.components["_value_"]`) supports dynamic replacement
- **TensorDict Communication**: Efficient data sharing between components
- **Hydra Configuration**: Config-driven component instantiation enables experimentation
- **PolicyStore/PolicyState**: Proven state management and persistence

### The LayerBase Problem

`LayerBase` combines multiple responsibilities:

```python
class LayerBase(nn.Module):
    # Computation logic
    def _forward(self, td: TensorDict) -> TensorDict
    
    # Infrastructure management
    def setup(self, source_components=None)
    def forward(self, td: TensorDict)  # DAG traversal + caching
    
    # Lifecycle management
    def _initialize(self)
    @property ready(self)
    
    # Relationship management
    self._sources, self._source_components
    
    # Shape coordination
    self._in_tensor_shapes, self._out_tensor_shape
```

This creates several fundamental issues:

1. **Testing Overhead**: Testing requires manual setup (shape configuration, lifecycle management, source mocking) as shown in `test_linear.py`
2. **Unclear Responsibilities**: Mixed concerns make the codebase harder to navigate and modify
3. **Coupling**: Changes to infrastructure can inadvertently affect computation logic and vice versa
4. **Obscured Hierarchy**: Component relationships are managed by components themselves rather than being explicitly visible

These issues require architectural change rather than incremental fixes.

## Proposed Architecture

### Design Principles

1. **Separation of Concerns**: Pure computation modules separate from infrastructure management
2. **Preserve Valuable Infrastructure**: Maintain component registry, TensorDict communication, and Hydra configuration
3. **TorchRL-Inspired Patterns**: Apply proven patterns for explicit data dependencies and composition
4. **Clean Migration**: Replace rather than bridge architectures for conceptual clarity

### Core Components

#### MettaModule: Pure Computation Core

Inspired by TorchRL's computational modules, `MettaModule` handles only computation:

```python
class MettaModule(nn.Module):
    """Pure computation module inspired by TorchRL patterns."""
    
    def __init__(self, in_keys=None, out_keys=None):
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Direct computation without DAG traversal."""
        pass
```

#### Enhanced Component Registry

The existing `MettaAgent.components` becomes a smarter container that manages both storage and execution:

```python
class ComponentContainer(nn.ModuleDict):
    """Enhanced version of MettaAgent.components with execution management."""
    
    def __init__(self):
        super().__init__()
        self.execution_graph = {}
    
    def register_component(self, name, module, dependencies=None):
        """Register component with explicit dependencies."""
        self[name] = module  # Standard nn.ModuleDict storage
        self.execution_graph[name] = dependencies or []
    
    def execute(self, tensordict: TensorDict) -> TensorDict:
        """Execute components in dependency order."""
        execution_order = self._topological_sort()
        for component_name in execution_order:
            self[component_name](tensordict)
        return tensordict
    
    def replace_component(self, name, new_module):
        """Preserve hotswapping capability."""
        self[name] = new_module
```

#### Wrapper Patterns

Following TorchRL's wrapper approach for cross-cutting concerns. Wrappers are fully composable - any wrapper can wrap any other wrapper or base module:

```python
class SafeModule(MettaModule):
    """Wraps computation with comprehensive safety checks."""
    
    def __init__(self, module, input_specs=None, output_specs=None, 
                 action_bounds=None, nan_check=True):
        # Inherit keys from wrapped module
        super().__init__(in_keys=module.in_keys, out_keys=module.out_keys)
        self.module = module
        self.input_specs = input_specs or {}
        self.output_specs = output_specs or {}
        self.action_bounds = action_bounds
        self.nan_check = nan_check
        
    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Pre-execution validation
        self._validate_inputs(tensordict)
        
        # Execute wrapped module
        result = self.module(tensordict)
        
        # Post-execution validation
        self._validate_outputs(result)
        
        return result
    
    def _validate_inputs(self, tensordict: TensorDict):
        """Input validation catching common failures."""
        for key in getattr(self.module, 'in_keys', []):
            if key not in tensordict:
                raise ValueError(f"Missing required input key: {key}")
            
            tensor = tensordict[key]
            
            # NaN/Inf checks
            if self.nan_check:
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN detected in input '{key}'")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf detected in input '{key}'")
            
            # Shape validation
            if key in self.input_specs:
                expected_shape = self.input_specs[key]
                if tensor.shape[1:] != expected_shape[1:]:  # Skip batch dimension
                    raise ValueError(f"Shape mismatch for '{key}': expected {expected_shape}, got {tensor.shape}")
    
    def _validate_outputs(self, tensordict: TensorDict):
        """Output validation with RL-specific checks."""
        for key in getattr(self.module, 'out_keys', []):
            if key not in tensordict:
                raise ValueError(f"Module failed to produce expected output: {key}")
            
            tensor = tensordict[key]
            
            # NaN/Inf checks for outputs
            if self.nan_check:
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN detected in output '{key}'")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf detected in output '{key}'")
            
            # Action bounds for RL policy networks
            if key.endswith("_action") and self.action_bounds:
                min_action, max_action = self.action_bounds
                if tensor.min() < min_action or tensor.max() > max_action:
                    # Clip instead of erroring for policy outputs
                    tensordict[key] = torch.clamp(tensor, min_action, max_action)
            
            # Value bounds for value functions
            if key.endswith("_value"):
                if tensor.abs().max() > 1000.0:
                    print(f"Warning: Extreme value estimates in '{key}': [{tensor.min():.2f}, {tensor.max():.2f}]")

class RegularizedModule(MettaModule):
    """Wraps computation with regularization, building on existing ParamLayer patterns."""
    
    def __init__(self, module, l2_scale=0.01, l1_scale=0.0):
        # Inherit keys from wrapped module
        super().__init__(in_keys=module.in_keys, out_keys=module.out_keys)
        self.module = module
        self.l2_scale = l2_scale
        self.l1_scale = l1_scale
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        result = self.module(tensordict)
        
        # Add regularization losses to TensorDict
        if self.training:
            reg_loss = self._compute_regularization()
            if "regularization_loss" in result:
                result["regularization_loss"] = result["regularization_loss"] + reg_loss
            else:
                result["regularization_loss"] = reg_loss
        
        return result
    
    def _compute_regularization(self) -> torch.Tensor:
        """Compute L1/L2 regularization extending existing patterns."""
        reg_loss = torch.tensor(0.0, device=next(self.module.parameters()).device)
        
        for param in self.module.parameters():
            if param.requires_grad:
                if self.l2_scale > 0:
                    reg_loss += self.l2_scale * torch.sum(param**2)
                if self.l1_scale > 0:
                    reg_loss += self.l1_scale * torch.sum(torch.abs(param))
        
        return reg_loss

class WeightMonitoringModule(MettaModule):
    """Adds weight monitoring to any MettaModule."""
    
    def __init__(self, module, clip_value=None, monitor_health=True):
        # Inherit keys from wrapped module
        super().__init__(in_keys=module.in_keys, out_keys=module.out_keys)
        self.module = module
        self.clip_value = clip_value
        self.monitor_health = monitor_health
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        result = self.module(tensordict)
        
        if self.training and self.monitor_health:
            self._monitor_weight_health()
        
        if self.clip_value and self.training:
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
        with torch.no_grad():
            for param in self.module.parameters():
                if param.data.dim() >= 2:  # Only clip weight matrices
                    param.data.clamp_(-self.clip_value, self.clip_value)
```

#### Wrapper Composability

Wrappers can be arbitrarily nested since each wrapper follows the same interface:

```python
# Build up layers of functionality
base_policy = PolicyModule(64, 32)

# Method 1: Step by step wrapping
safe_policy = SafeModule(base_policy, action_bounds=(-1.0, 1.0))
regularized_safe_policy = RegularizedModule(safe_policy, l2_scale=0.01)
final_policy = WeightMonitoringModule(regularized_safe_policy, clip_value=1.0)

# Method 2: Nested composition
final_policy = WeightMonitoringModule(
    RegularizedModule(
        SafeModule(base_policy, action_bounds=(-1.0, 1.0)),
        l2_scale=0.01
    ),
    clip_value=1.0
)

# All wrappers work the same: tensordict -> tensordict
result = final_policy(input_tensordict)
```

#### Testing Wrapper Patterns

Each wrapper can be tested independently and in composition:

```python
# Test individual wrappers
def test_safe_module():
    mock_module = MockModule()
    safe_module = SafeModule(mock_module, action_bounds=(-1.0, 1.0))
    result = safe_module(test_tensordict)
    # Test safety validation logic in isolation

def test_regularized_module():
    mock_module = MockModule()
    reg_module = RegularizedModule(mock_module, l2_scale=0.01)
    result = reg_module(test_tensordict)
    # Test regularization logic in isolation

# Test wrapper composition
def test_composed_wrappers():
    base_module = SimpleLinear(10, 5)
    composed = RegularizedModule(
        SafeModule(base_module, action_bounds=(-1.0, 1.0)),
        l2_scale=0.01
    )
    result = composed(test_tensordict)
    # Test that both safety and regularization work together
```

## Testing in Isolation

The new architecture enables true isolated testing of components:

### Current Testing Approach

```python
# Current: Manual setup required (from test_linear.py)
def test_linear_layer():
    linear_layer = Linear(name="_test_", sources=[...], nn_params=...)
    linear_layer._in_tensor_shapes = [[input_size]]  # Manual configuration
    linear_layer._out_tensor_shape = [output_size]   # Manual configuration
    linear_layer._initialize()                       # Manual lifecycle
    linear_layer._source_components = None           # Manual mocking
    
    # Finally can test
    result = linear_layer._forward(test_tensordict)
```

### New Testing Approach

```python
# New: Direct testing with zero setup
def test_linear_module():
    linear = LinearModule(in_features=10, out_features=16)
    result = linear({"input": torch.randn(4, 10)})
    assert result["output"].shape == (4, 16)

def test_actor_module():
    actor = ActorModule(state_dim=64, action_dim=8)
    result = actor({"state": torch.randn(2, 64)})
    assert result["action"].shape == (2, 8)

def test_merge_module():
    merge = MergeModule(["input1", "input2"], merge_type="concat")
    result = merge({
        "input1": torch.randn(3, 5),
        "input2": torch.randn(3, 7)
    })
    assert result["merged"].shape == (3, 12)
```

### Benefits of Isolated Testing

1. **No Infrastructure Dependencies**: Test pure computation without DAG setup, lifecycle management, or shape propagation
2. **Fast Execution**: No recursive dependency resolution or complex initialization
3. **Clear Intent**: Tests focus on the specific functionality being tested
4. **Easy Debugging**: When a test fails, the issue is isolated to that specific component
5. **Parallel Development**: Components can be developed and tested independently

### Component vs Integration Testing

```python
# Component testing - test logic in isolation
def test_regularization_computation():
    base_module = LinearModule(5, 3)
    reg_module = RegularizedModule(base_module, l2_scale=0.01)
    
    # Test that regularization loss is computed correctly
    result = reg_module({"input": torch.randn(2, 5)})
    assert "regularization_loss" in result
    assert result["regularization_loss"] > 0

# Integration testing - test components work together
def test_full_agent_pipeline():
    agent = MettaAgent(config)
    result = agent(test_observation)
    assert "action" in result
    assert "value" in result
```

This separation allows for comprehensive testing at both the component and system levels while maintaining fast, focused unit tests.

## Implementation Strategy

### Week 1: Foundation + Test Infrastructure

**Day 1-2: Core Foundation (8 hours)**
```python
# MettaModule base class (~4 hours)
class MettaModule(nn.Module):
    def __init__(self, in_keys=None, out_keys=None): pass

# ComponentContainer implementation (~4 hours)  
class ComponentContainer(nn.ModuleDict):
    def execute(self, tensordict): pass
```

**Day 3-5: Parallel Component Migration + Testing (12 hours)**
```python
# Convert all components simultaneously (independent work)
# LinearModule: 2 hours
# ActorModule: 3 hours  
# MergeModule: 3 hours
# ObsModule, LSTMModule, etc: 4 hours total
```

**Testing Strategy (Parallel Development):**
- **Unit Tests**: Write alongside each component conversion
- **Performance Benchmarks**: Establish baselines for current system
- **Integration Tests**: Compare old vs new agent behavior
- **Migration Validation**: Ensure identical outputs for same inputs

### Week 2: Integration + Validation

**Day 1-3: MettaAgent Integration (12 hours)**
```python
# Integrate ComponentContainer into MettaAgent (~8 hours)
class MettaAgent(nn.Module):
    def __init__(self, config):
        self.components = ComponentContainer()

# Config compatibility layer (~4 hours)
```

**Day 4-5: Wrapper Patterns + Final Testing (8 hours)**
```python
# Safety wrappers (~4 hours)
class SafeModule(MettaModule): pass
class RegularizedModule(MettaModule): pass

# Comprehensive testing suite (~4 hours)
```

### Testing Requirements

**Unit Tests (Developed in Parallel):**
- Each MettaModule tested in isolation
- Wrapper composability testing
- Performance regression testing

**Integration Tests:**
- End-to-end agent behavior validation
- Config loading compatibility
- Hotswapping functionality preservation

**Performance Tests:**
- Memory usage comparison
- Forward pass timing benchmarks
- Training speed validation

**Migration Tests:**
- Identical outputs for identical inputs
- Model loading/saving compatibility
- Gradual rollout validation

### Realistic Timeline

**Total Effort: ~40 hours (2 weeks at 50% allocation)**

**Risk Mitigation:**
- All components are independent conversions
- Testing runs parallel to development
- Existing infrastructure preserved
- Clear rollback path at each stage

**Acceleration Factors:**
- Components are simplified, not complicated
- No dependency between component conversions  
- Comprehensive test suite enables confident migration
- Clean architecture eliminates dual system maintenance

## Benefits

### Immediate Benefits

1. **Streamlined Testing**: Pure computation modules eliminate manual shape configuration and lifecycle management
2. **Clear Data Dependencies**: Explicit `in_keys`/`out_keys` make component relationships transparent
3. **Isolated Development**: Components can be developed and tested independently without infrastructure

### Medium-term Benefits

1. **Cleaner Architecture**: Single responsibility modules with clear interfaces
2. **Composition Over Inheritance**: Wrapper patterns enable feature addition without complex inheritance
3. **Preserved Infrastructure**: Keep valuable patterns (registry, TensorDict, Hydra) while fixing component design

### Long-term Benefits

1. **Future PyTorch Compatibility**: Explicit interfaces ready for torch.compile and other emerging features
2. **Advanced Research Patterns**: Container-managed execution enables sophisticated experimentation
3. **Maintainable Codebase**: Separated concerns reduce coupling and improve code clarity

### Extensibility Considerations

The proposed architecture aims to support future extensions through clear interfaces:

```python
# State management could be added as a module
class StateManager(MettaModule):
    in_keys = ["current_state", "action"] 
    out_keys = ["next_state", "state_delta"]

# Or as a wrapper around existing components
class StatefulModule(MettaModule):
    def __init__(self, module, state_manager):
        # Adds state tracking to any module
        pass

# Alternative execution strategies could be swapped
class MettaAgent:
    def __init__(self, config):
        self.components = ComponentContainer()
        
        # Execution strategy based on config
        if config.execution == "parallel":
            self.executor = ParallelDAGExecutor()
        else:
            self.executor = SequentialDAGExecutor()
```

**Extension Points:**
- New MettaModules integrate through standard interfaces
- Wrapper patterns allow enhancement without modification
- ComponentContainer could support different execution strategies
- TensorDict provides a common data exchange format

The explicit `in_keys`/`out_keys` contracts and wrapper composability should make it straightforward to add functionality like memory management, attention mechanisms, or distributed execution without requiring changes to existing components.

## Migration Strategy

### Component Migration

- Convert existing `LayerBase` components to `MettaModule` pattern
- Preserve all functionality while simplifying interfaces
- Maintain Hydra configuration compatibility where possible

### Infrastructure Preservation

- Enhance `MettaAgent.components` to become the ComponentContainer (preserves hotswapping interface)
- Preserve TensorDict communication patterns
- Maintain PolicyStore/PolicyState for model persistence

### Validation Approach

- Comprehensive testing of new components
- Performance benchmarks against current implementation
- Gradual rollout with fallback options

## Frequently Asked Questions

**Q: Will existing Hydra configurations work with the new architecture?**

A: Largely yes, but requires validation. The component instantiation patterns should remain similar, with the main change being target class names:

```yaml
# Config structure should remain similar
components:
  linear_1:
    _target_: metta.agent.lib.LinearModule  # Changed from Linear
    in_features: 64
    out_features: 32
    sources: ["obs_processor"]  # Dependency handling needs verification
```

However, the `sources` field and dependency management may require changes depending on how ComponentContainer handles relationships compared to current LayerBase setup. This needs careful testing during implementation.

**Q: Will existing saved models load correctly after migration?**

A: This requires careful verification. While ComponentContainer inherits from `nn.ModuleDict` like current `MettaAgent.components`, compatibility depends on:
- Preserving exact component names in state dict
- Maintaining parameter structure within each component
- PolicyStore loading mechanisms working with new component types

Model compatibility should be thoroughly tested and may require migration utilities for existing saved policies. This is a critical validation point during implementation.

**Q: What's the main risk and how is it mitigated?**

A: The primary risks are performance regression and compatibility issues. Mitigation strategy:
- Comprehensive benchmarking before/after migration
- Model loading/saving compatibility verification
- Performance tests running in parallel with development  
- Component-by-component validation
- Clear rollback path if critical issues arise

**Q: What documentation updates will be needed?**

A: Documentation updates will include:
- Component development guide (MettaModule patterns)
- Testing workflow updates (simplified unit testing)
- Wrapper pattern examples for common use cases
- Migration guide for configs and any compatibility issues discovered

**Q: Can we migrate gradually or is it all-or-nothing?**

A: This depends on ComponentContainer implementation. If it expects all components to be MettaModules, gradual migration may require:
- Bridge patterns to mix old and new components temporarily
- Or a coordinated migration of all components simultaneously

The migration strategy needs validation during Phase 1 implementation to determine the most practical approach.

**Q: How does this affect training and inference pipelines?**

A: Should be minimal impact if the `MettaAgent.forward()` interface remains unchanged. However, any pipeline code that directly accesses component internals or relies on LayerBase-specific behavior will need updates. This requires testing with actual training workflows.

## Conclusion

This proposal recommends replacing the monolithic `LayerBase` architecture with a clean, modular design that separates computation from infrastructure concerns. By adopting TorchRL-inspired patterns while preserving Metta's valuable infrastructure, we can achieve:

**Immediate Improvements:**
- Zero-setup component testing
- Clearer architectural responsibilities  
- Simplified development workflow

**Long-term Benefits:**
- Maintainable, extensible codebase
- Future PyTorch compatibility
- Research-friendly experimentation platform

The migration requires careful validation of compatibility concerns (model loading, config handling) but offers a path to cleaner architecture without sacrificing existing capabilities.

**Recommendation:** Proceed with the 2-week implementation timeline, starting with foundation development and component migration, while running comprehensive testing in parallel to ensure system reliability and compatibility.