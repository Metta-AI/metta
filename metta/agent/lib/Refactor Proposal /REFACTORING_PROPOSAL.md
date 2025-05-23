# Metta Architecture Refactoring Proposal

## Executive Summary

This proposal recommends replacing the current `LayerBase` architecture with a cleaner design inspired by TorchRL's proven patterns. Rather than incrementally enhancing the existing monolithic approach, we propose a clean migration to separated computation and infrastructure concerns.

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
    def _initialize()
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

Inspired by TorchRL's `TensorDictModule`, `MettaModule` handles only computation:

```python
class MettaModule(nn.Module):
    """Pure computation module inspired by TorchRL's TensorDictModule."""
    
    def __init__(self, in_keys=None, out_keys=None):
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Direct computation without DAG traversal."""
        pass
```

#### Enhanced Component Registry

The existing `MettaAgent.components` becomes a smarter container that follows TorchRL's `TensorDictModule` pattern:

```python
from tensordict.nn import TensorDictModuleBase
from torch import nn
from tensordict import TensorDict
import torch

class ComponentContainer(TensorDictModuleBase):
    """Enhanced container following TorchRL's TensorDictModule pattern."""
    
    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict()  # Store actual components
        self.component_configs = {}  # Store component configurations for deferred init
        self.dependencies = {}  # Track component dependencies
        self.initialized = False
        
        # TensorDictModule interface - will be dynamically determined
        self.in_keys = []
        self.out_keys = []
    
    def register_component(self, name, component_class, config, dependencies=None, 
                          in_keys=None, out_keys=None):
        """Register component configuration for deferred initialization."""
        self.component_configs[name] = {
            'class': component_class,
            'config': config,
            'in_keys': in_keys or [],
            'out_keys': out_keys or [],
        }
        self.dependencies[name] = dependencies or []
        
        # Update container's keys based on registered component specs
        self._update_container_keys()
    
    def initialize_with_input_shapes(self, input_shapes: dict):
        """Initialize all components with specified input shapes.
        
        Args:
            input_shapes: Dict mapping container's in_keys to their shapes (without batch dim)
                         e.g., {"observation": (64,)} for obs_dim=64
        """
        if self.initialized:
            return
        
        # Check for circular dependencies
        self._check_dependencies()
        
        # Validate that all container inputs have shapes specified
        for in_key in self.in_keys:
            if in_key not in input_shapes:
                raise ValueError(f"Shape not specified for container input '{in_key}'")
        
        # Initialize components in dependency order
        shape_registry = input_shapes.copy()
        
        # Initialize each component recursively
        for component_name in self._topological_sort():
            self._initialize_component(component_name, shape_registry)
        
        self.initialized = True
    
    def _initialize_component(self, component_name: str, shape_registry: dict):
        """Initialize a single component with inferred input shapes."""
        if component_name in self.components:
            return  # Already initialized
        
        config_info = self.component_configs[component_name]
        component_class = config_info['class']
        base_config = config_info['config'].copy()
        in_keys = config_info['in_keys']
        out_keys = config_info['out_keys']
        
        # Ensure all dependencies are initialized first
        for dep_name in self.dependencies[component_name]:
            self._initialize_component(dep_name, shape_registry)
        
        # Infer input shapes from dependencies or sample input
        inferred_shapes = {}
        for in_key in in_keys:
            if in_key not in shape_registry:
                raise ValueError(f"Cannot infer shape for key '{in_key}' needed by '{component_name}'")
            inferred_shapes[in_key] = shape_registry[in_key]
        
        # Update config with inferred shapes
        updated_config = self._update_config_with_shapes(
            component_class, base_config, inferred_shapes, in_keys, out_keys
        )
        
        # Create the component instance
        component = component_class(**updated_config, in_keys=in_keys, out_keys=out_keys)
        self.components[component_name] = component
        
        # Infer output shapes through dry run
        self._infer_output_shapes(component, component_name, shape_registry, inferred_shapes)
    
    def _update_config_with_shapes(self, component_class, config, input_shapes, in_keys, out_keys):
        """Update component config with inferred input dimensions."""
        updated_config = config.copy()
        
        # Common patterns for different component types
        if hasattr(component_class, '_update_config_from_shapes'):
            # Allow components to define their own shape inference
            return component_class._update_config_from_shapes(config, input_shapes, in_keys, out_keys)
        
        # Default patterns
        if 'LinearModule' in component_class.__name__:
            # Linear layers typically need in_features
            if len(in_keys) == 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config['in_features'] = input_shape[-1]  # Last dimension
        
        elif 'Conv2dModule' in component_class.__name__:
            # Conv layers need in_channels
            if len(in_keys) == 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config['in_channels'] = input_shape[0]  # Assuming CHW format
        
        elif 'LSTMModule' in component_class.__name__:
            # LSTM needs input_size
            if len(in_keys) >= 1:
                input_shape = input_shapes[in_keys[0]]  # First input key
                updated_config['input_size'] = input_shape[-1]
        
        return updated_config
    
    def _infer_output_shapes(self, component, component_name, shape_registry, input_shapes):
        """Infer output shapes through a dry forward pass."""
        # Create mock input tensordict
        mock_inputs = {}
        for in_key, shape in input_shapes.items():
            mock_inputs[in_key] = torch.zeros(1, *shape)
        
        mock_tensordict = TensorDict(mock_inputs, batch_size=1)
        
        # Dry run to get output shapes
        with torch.no_grad():
            component.eval()
            result = component(mock_tensordict)
        
        # Record output shapes
        for out_key in component.out_keys:
            if out_key in result:
                shape_registry[out_key] = result[out_key].shape[1:]  # Remove batch dimension
    
    def _update_container_keys(self):
        """Update container's in_keys and out_keys based on registered components."""
        # Collect all keys from component configurations
        all_in_keys = set()
        all_out_keys = set()
        
        for config_info in self.component_configs.values():
            all_in_keys.update(config_info['in_keys'])
            all_out_keys.update(config_info['out_keys'])
        
        # Container's inputs are keys not produced by any component
        self.in_keys = [key for key in all_in_keys if key not in all_out_keys]
        # Container's outputs are all component outputs
        self.out_keys = list(all_out_keys)
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """TensorDictModule interface - execute all components."""
        if not self.initialized:
            raise RuntimeError("ComponentContainer must be initialized with input shapes first")
        
        # For full execution, just run all components in dependency order
        for component_name in self._topological_sort():
            self._execute_component(component_name, tensordict)
        return tensordict
    
    def execute_component(self, component_name: str, tensordict: TensorDict) -> TensorDict:
        """Recursive execution preserving current elegant pattern."""
        if not self.initialized:
            raise RuntimeError("ComponentContainer must be initialized with input shapes first")
        return self._execute_component(component_name, tensordict)
    
    def _execute_component(self, component_name: str, tensordict: TensorDict) -> TensorDict:
        """Internal recursive execution method."""
        component = self.components[component_name]
        
        # Check if already computed (caching) - check if all outputs exist
        if all(out_key in tensordict for out_key in component.out_keys):
            return tensordict
        
        # Recursively compute dependencies first
        if component_name in self.dependencies:
            for dep_name in self.dependencies[component_name]:
                self._execute_component(dep_name, tensordict)
        
        # Execute this component
        component(tensordict)
        return tensordict
    
    def _check_dependencies(self):
        """Check for circular dependencies."""
        visited = set()
        temp_visited = set()
        
        def visit(component_name):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            for dep_name in self.dependencies.get(component_name, []):
                if dep_name not in self.component_configs:
                    raise ValueError(f"Dependency '{dep_name}' not found for component '{component_name}'")
                visit(dep_name)
            temp_visited.remove(component_name)
            visited.add(component_name)
        
        for component_name in self.component_configs.keys():
            if component_name not in visited:
                visit(component_name)
    
    def _topological_sort(self):
        """Get components in dependency order."""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(component_name):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            for dep_name in self.dependencies.get(component_name, []):
                visit(dep_name)
            temp_visited.remove(component_name)
            visited.add(component_name)
            result.append(component_name)
        
        for component_name in self.component_configs.keys():
            if component_name not in visited:
                visit(component_name)
        
        return result
    
    def replace_component(self, name, new_module):
        """Preserve hotswapping capability."""
        self.components[name] = new_module
        self._update_container_keys()
```

#### Recursive Execution with MettaModule

Each `MettaModule` becomes purely computational, while the container handles the recursive dependency resolution:

```python
# Usage preserves the elegant recursive pattern:
class MettaAgent(MettaModule):
    def __init__(self, config):
        super().__init__(in_keys=["observation"], out_keys=["action", "value"])
        self.components = ComponentContainer()
        
        # Register components with dependencies
        # Components are registered during initialization based on config
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Option 1: Execute specific component recursively (preserves current pattern)
        self.components.execute_component("policy", tensordict)
        
        # Option 2: Execute all components (TensorDictModule interface)
        # tensordict = self.components(tensordict)
        
        return tensordict
```

#### Complete Initialization Example

Here's how the deferred initialization works in practice:

```python
# Step 1: Register component configurations (no actual initialization yet)
agent = MettaAgent()

# Register observation processor
agent.components.register_component(
    name="obs_processor",
    component_class=LinearModule,
    config={"out_features": 128},  # in_features will be inferred
    dependencies=[],
    in_keys=["observation"],
    out_keys=["features"]
)

# Register policy network
agent.components.register_component(
    name="policy",
    component_class=LinearModule,
    config={"out_features": 8},  # in_features will be inferred from obs_processor output
    dependencies=["obs_processor"],
    in_keys=["features"],
    out_keys=["action"]
)

# Register value network
agent.components.register_component(
    name="value",
    component_class=LinearModule,  
    config={"out_features": 1},  # in_features will be inferred from obs_processor output
    dependencies=["obs_processor"],
    in_keys=["features"], 
    out_keys=["value"]
)

# Step 2: Initialize with input shapes (this triggers shape inference)
# The container automatically determined its in_keys = ["observation"] 
agent.components.initialize_with_input_shapes({
    "observation": (64,)  # obs_dim = 64 (without batch dimension)
})

# Now the components are fully initialized with correct dimensions:
# - obs_processor: Linear(64 -> 128) 
# - policy: Linear(128 -> 8)
# - value: Linear(128 -> 1)

# Step 3: Use normally
observation = torch.randn(4, 64)
tensordict = TensorDict({"observation": observation}, batch_size=4)
result = agent.forward(tensordict)  # Returns updated tensordict with action, value
```

#### Advanced Shape Inference

Components can define custom shape inference logic:

```python
class LSTMModule(MettaModule):
    @classmethod
    def _update_config_from_shapes(cls, config, input_shapes, in_keys, out_keys):
        """Custom shape inference for LSTM."""
        updated_config = config.copy()
        
        # LSTM needs input_size from first input
        if len(in_keys) >= 1:
            input_shape = input_shapes[in_keys[0]]
            updated_config['input_size'] = input_shape[-1]
        
        # If hidden state is provided, infer hidden_size from it
        if 'hidden' in input_shapes and 'hidden_size' not in config:
            hidden_shape = input_shapes['hidden']
            updated_config['hidden_size'] = hidden_shape[-1]
        
        return updated_config
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
    linear = LinearModule(in_features=10, out_features=16, 
                         in_keys=["input"], out_keys=["output"])
    tensordict = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
    result = linear(tensordict)
    assert result["output"].shape == (4, 16)

def test_actor_module():
    actor = PolicyNetwork(feature_dim=64, action_dim=8,
                         in_keys=["features"], out_keys=["action"])
    tensordict = TensorDict({"features": torch.randn(2, 64)}, batch_size=2)
    result = actor(tensordict)
    assert result["action"].shape == (2, 8)
```

## Implementation Strategy

### Week 1: Foundation + Test Infrastructure

**Day 1-2: Core Foundation (8 hours)**
```python
# MettaModule base class (~4 hours)
class MettaModule(nn.Module):
    def __init__(self, in_keys=None, out_keys=None): pass

# ComponentContainer implementation (~4 hours)  
class ComponentContainer(TensorDictModuleBase):
    def register_component(self, name, component_class, config, dependencies=None): pass
    def initialize_with_input_shapes(self, input_shapes): pass
```

**Day 3-5: Parallel Component Migration + Testing (12 hours)**
```python
# Convert all components simultaneously (independent work)
# LinearModule: 2 hours
# ActorModule: 3 hours  
# MergeModule: 3 hours
# ObsModule, LSTMModule, etc: 4 hours total
```

### Week 2: Integration + Validation

**Day 1-3: MettaAgent Integration (12 hours)**
```python
# Integrate ComponentContainer into MettaAgent (~8 hours)
class MettaAgent(MettaModule):
    def __init__(self, config):
        super().__init__(in_keys=["observation"], out_keys=["action", "value"])
        self.components = ComponentContainer()

# Config compatibility layer (~4 hours)
```

**Day 4-5: Optional Wrappers + Final Testing (8 hours)**
```python
# Safety wrappers (~4 hours)
class SafeModule(MettaModule): pass

# Comprehensive testing suite (~4 hours)
```

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
# Config structure will need updates for new registration pattern
components:
  obs_processor:
    _target_: metta.agent.lib.LinearModule
    config:
      out_features: 128  # in_features will be inferred
    dependencies: []
    in_keys: ["observation"]
    out_keys: ["features"]
  
  policy:
    _target_: metta.agent.lib.LinearModule
    config:
      out_features: 8  # in_features will be inferred
    dependencies: ["obs_processor"]
    in_keys: ["features"]
    out_keys: ["action"]
```

The `sources` field is replaced with explicit `dependencies`, and component parameters are nested under `config` to separate them from registration metadata. This requires config loader updates during implementation.

**Q: Will existing saved models load correctly after migration?**

A: This requires careful verification. ComponentContainer inherits from `TensorDictModuleBase` and stores components in an internal `nn.ModuleDict`. Compatibility depends on:
- Preserving exact component names in state dict
- Maintaining parameter structure within each component  
- PolicyStore loading mechanisms working with new component types
- TensorDictModuleBase state dict compatibility

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