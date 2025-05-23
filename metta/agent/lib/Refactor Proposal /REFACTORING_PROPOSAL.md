# Metta Architecture Refactoring Proposal

## Executive Summary

This proposal recommends replacing the current `LayerBase` architecture with a cleaner design inspired by TorchRL's proven patterns. Rather than incrementally enhancing the existing monolithic approach, we propose a clean migration to separated computation and infrastructure concerns.

**Core Insight:** `LayerBase`'s monolithic design—combining computation, lifecycle management, DAG traversal, shape management, and dependency coordination—creates fundamental testing difficulties and architectural coupling that incremental fixes cannot address.

**Proposed Solution:** Replace `LayerBase` with pure computation modules (`MettaModule`) managed by a composable wrapper architecture, while preserving the valuable existing infrastructure: component registry, TensorDict communication, and Hydra configuration.

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
4. **Composable Wrapper Design**: Build features through composition rather than monolithic inheritance
5. **Clean Migration**: Replace rather than bridge architectures for conceptual clarity

### Core Components

#### MettaModule: Pure Computation Core

Inspired by TorchRL's `TensorDictModule`, `MettaModule` handles only computation:

```python
class MettaModule(nn.Module):
    """Pure computation module inspired by TorchRL's TensorDictModule."""
    
    def __init__(self, in_keys=None, out_keys=None, input_shapes=None, output_shapes=None):
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []
        self.input_shapes = input_shapes or {}
        self.output_shapes = output_shapes or {}
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Direct computation without DAG traversal."""
        pass
```

#### Composable Container Architecture

Instead of a monolithic container, we build functionality through composable wrappers:

##### 1. Base ComponentContainer: Simple Storage + Elegant Execution

```python
class ComponentContainer(TensorDictModuleBase):
    """Base container with immediate registration and elegant recursive execution.
    
    This is the foundation layer that provides:
    - Simple component storage in nn.ModuleDict
    - Elegant recursive execution pattern
    - Basic dependency tracking
    - TensorDictModule compliance
    """
    
    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict()  # Store actual component instances
        self.dependencies = {}  # Track component dependencies
        self.in_keys = []  # Determined from registered components
        self.out_keys = []  # Determined from registered components
    
    def register_component(self, name: str, component: MettaModule, dependencies: List[str] = None):
        """Register an actual component instance immediately.
        
        Args:
            name: Component name for registry and dependency resolution
            component: Actual MettaModule instance (already initialized)
            dependencies: List of component names this module depends on
        """
        self.components[name] = component
        self.dependencies[name] = dependencies or []
        self._update_container_keys()
    
    def execute_component(self, component_name: str, tensordict: TensorDict) -> TensorDict:
        """Recursive execution preserving current elegant pattern.
        
        This is the heart of the architecture - the beautiful recursive execution
        pattern that automatically handles dependency resolution and caching.
        """
        component = self.components[component_name]
        
        # Check if already computed (output presence caching)
        if all(out_key in tensordict for out_key in component.out_keys):
            return tensordict
        
        # Recursively compute dependencies first
        for dep_name in self.dependencies[component_name]:
            self.execute_component(dep_name, tensordict)
        
        # Execute this component
        return component(tensordict)
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """TensorDictModule interface - execute all components."""
        for component_name in self._topological_sort():
            self.execute_component(component_name, tensordict)
        return tensordict
    
    def replace_component(self, name: str, new_component: MettaModule):
        """Preserve hotswapping capability."""
        self.components[name] = new_component
        self._update_container_keys()
    
    def _update_container_keys(self):
        """Update container's in_keys and out_keys based on registered components."""
        all_in_keys = set()
        all_out_keys = set()
        
        for component in self.components.values():
            all_in_keys.update(component.in_keys)
            all_out_keys.update(component.out_keys)
        
        # Container's inputs are keys not produced by any component
        self.in_keys = [key for key in all_in_keys if key not in all_out_keys]
        self.out_keys = list(all_out_keys)
    
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
        
        for component_name in self.components.keys():
            if component_name not in visited:
                visit(component_name)
        
        return result
```

##### 2. SafeComponentContainer: Validation Layer

```python
class SafeComponentContainer(ComponentContainer):
    """Adds comprehensive validation to component registration and execution.
    
    This wrapper adds:
    - Component interface validation
    - Dependency existence checking
    - Circular dependency detection
    - Runtime shape validation
    """
    
    def register_component(self, name: str, component: MettaModule, dependencies: List[str] = None):
        """Register component with comprehensive validation."""
        self._validate_component(component)
        self._validate_dependencies(name, dependencies or [])
        super().register_component(name, component, dependencies)
    
    def _validate_component(self, component: MettaModule):
        """Validate component has proper MettaModule interface."""
        if not isinstance(component, MettaModule):
            raise TypeError(f"Component must be a MettaModule, got {type(component)}")
        
        if not hasattr(component, 'in_keys') or not hasattr(component, 'out_keys'):
            raise ValueError(f"Component must have in_keys and out_keys attributes")
        
        if not isinstance(component.in_keys, list) or not isinstance(component.out_keys, list):
            raise ValueError(f"Component in_keys and out_keys must be lists")
        
        # Check for output key conflicts
        for existing_name, existing_component in self.components.items():
            overlap = set(component.out_keys) & set(existing_component.out_keys)
            if overlap:
                raise ValueError(f"Output key conflict: {overlap} between '{name}' and '{existing_name}'")
    
    def _validate_dependencies(self, name: str, dependencies: List[str]):
        """Validate dependencies exist and check for cycles."""
        for dep_name in dependencies:
            if dep_name not in self.components and dep_name != name:
                raise ValueError(f"Dependency '{dep_name}' not found for component '{name}'")
        
        # Create temporary dependency structure and check for cycles
        temp_deps = self.dependencies.copy()
        temp_deps[name] = dependencies
        self._check_circular_dependencies(temp_deps)
    
    def _check_circular_dependencies(self, deps_dict: Dict[str, List[str]]):
        """Check for circular dependencies in dependency graph."""
        visited = set()
        temp_visited = set()
        
        def visit(component_name):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            for dep_name in deps_dict.get(component_name, []):
                if dep_name in deps_dict:  # Only visit if it's a registered component
                    visit(dep_name)
            temp_visited.remove(component_name)
            visited.add(component_name)
        
        for component_name in deps_dict.keys():
            if component_name not in visited:
                visit(component_name)
```

##### 3. LazyComponentContainer: Deferred Initialization Layer

```python
class LazyComponentContainer(ComponentContainer):
    """Adds deferred initialization with automatic shape inference.
    
    This wrapper adds:
    - Configuration-based registration
    - Automatic shape inference during initialization
    - Deferred component instantiation
    - Custom shape inference patterns
    """
    
    def __init__(self):
        super().__init__()
        self.component_configs = {}  # Store component configurations
        self.initialized = False
    
    def register_component_config(self, name: str, component_class, config: Dict[str, Any],
                                 dependencies: List[str] = None, in_keys: List[str] = None,
                                 out_keys: List[str] = None):
        """Register component configuration for deferred initialization.
        
        Args:
            name: Component name for registry and dependency resolution
            component_class: Class to instantiate (e.g., LinearModule)
            config: Configuration dict (e.g., {"out_features": 128})
            dependencies: List of component names this module depends on
            in_keys: Input keys for this component
            out_keys: Output keys for this component
        """
        self.component_configs[name] = {
            'class': component_class,
            'config': config,
            'in_keys': in_keys or [],
            'out_keys': out_keys or [],
        }
        self.dependencies[name] = dependencies or []
        
        # Update container's keys based on registered component specs
        self._update_container_keys_from_configs()
    
    def initialize_with_input_shapes(self, input_shapes: Dict[str, tuple]):
        """Initialize all components with specified input shapes.
        
        Args:
            input_shapes: Dict mapping container's in_keys to their shapes (without batch dim)
                         e.g., {"observation": (64,)} for obs_dim=64
        """
        if self.initialized:
            return
        
        # Validate that all container inputs have shapes specified
        for in_key in self.in_keys:
            if in_key not in input_shapes:
                raise ValueError(f"Shape not specified for container input '{in_key}'")
        
        # Initialize components in dependency order
        shape_registry = input_shapes.copy()
        
        for component_name in self._topological_sort_configs():
            component = self._create_component_with_shapes(component_name, shape_registry)
            # Register the actual component using the base class method
            super().register_component(component_name, component, self.dependencies[component_name])
        
        self.initialized = True
    
    def _create_component_with_shapes(self, component_name: str, shape_registry: Dict[str, tuple]) -> MettaModule:
        """Create a component instance with inferred shapes."""
        config_info = self.component_configs[component_name]
        component_class = config_info['class']
        base_config = config_info['config'].copy()
        in_keys = config_info['in_keys']
        out_keys = config_info['out_keys']
        
        # Ensure all dependencies are initialized first
        for dep_name in self.dependencies[component_name]:
            if dep_name not in self.components:
                dep_component = self._create_component_with_shapes(dep_name, shape_registry)
                super().register_component(dep_name, dep_component, self.dependencies[dep_name])
        
        # Infer input shapes from shape registry
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
        component = component_class(**updated_config)
        
        # Infer output shapes through dry run and update shape registry
        self._infer_output_shapes(component, shape_registry, inferred_shapes)
        
        return component
    
    def _update_config_with_shapes(self, component_class, config: Dict[str, Any], 
                                  input_shapes: Dict[str, tuple], in_keys: List[str], 
                                  out_keys: List[str]) -> Dict[str, Any]:
        """Update component config with inferred input dimensions."""
        updated_config = config.copy()
        
        # Allow components to define custom shape inference
        if hasattr(component_class, '_update_config_from_shapes'):
            return component_class._update_config_from_shapes(config, input_shapes, in_keys, out_keys)
        
        # Default patterns for common component types
        if 'LinearModule' in component_class.__name__:
            if len(in_keys) == 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config['in_features'] = input_shape[-1]  # Last dimension
                updated_config['in_key'] = in_keys[0]
                updated_config['out_key'] = out_keys[0]
        
        elif 'Conv2dModule' in component_class.__name__:
            if len(in_keys) == 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config['in_channels'] = input_shape[0]  # Assuming CHW format
                updated_config['in_key'] = in_keys[0]
                updated_config['out_key'] = out_keys[0]
        
        elif 'LSTMModule' in component_class.__name__:
            if len(in_keys) >= 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config['input_size'] = input_shape[-1]
                # LSTMModule uses different key names
                updated_config['in_key'] = in_keys[0]
                updated_config['out_key'] = out_keys[0]
        
        return updated_config
    
    def _infer_output_shapes(self, component: MettaModule, shape_registry: Dict[str, tuple], 
                           input_shapes: Dict[str, tuple]):
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
        
        # Record output shapes (remove batch dimension)
        for out_key in component.out_keys:
            if out_key in result:
                shape_registry[out_key] = result[out_key].shape[1:]
    
    def _topological_sort_configs(self):
        """Get component configs in dependency order."""
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
    
    def _update_container_keys_from_configs(self):
        """Update container's in_keys and out_keys based on component configurations."""
        all_in_keys = set()
        all_out_keys = set()
        
        for config_info in self.component_configs.values():
            all_in_keys.update(config_info['in_keys'])
            all_out_keys.update(config_info['out_keys'])
        
        # Container's inputs are keys not produced by any component
        self.in_keys = [key for key in all_in_keys if key not in all_out_keys]
        self.out_keys = list(all_out_keys)
```

##### 4. SafeLazyComponentContainer: Combined Features

```python
class SafeLazyComponentContainer(LazyComponentContainer, SafeComponentContainer):
    """Combines safety validation with lazy initialization.
    
    This is the full-featured container that provides:
    - Deferred initialization with shape inference (from LazyComponentContainer)
    - Comprehensive validation (from SafeComponentContainer) 
    - Elegant recursive execution (from base ComponentContainer)
    """
    
    def register_component_config(self, name: str, component_class, config: Dict[str, Any],
                                 dependencies: List[str] = None, in_keys: List[str] = None,
                                 out_keys: List[str] = None):
        """Register component configuration with validation."""
        # Validate the component class has proper interface
        if not hasattr(component_class, '__bases__') or not any(
            issubclass(base, MettaModule) for base in component_class.__mro__
        ):
            raise TypeError(f"Component class must inherit from MettaModule")
        
        # Use LazyComponentContainer's registration
        LazyComponentContainer.register_component_config(
            self, name, component_class, config, dependencies, in_keys, out_keys
        )
    
    def initialize_with_input_shapes(self, input_shapes: Dict[str, tuple]):
        """Initialize with both lazy initialization and safety validation."""
        # The safety validation happens automatically when LazyComponentContainer
        # calls super().register_component() during initialization, which triggers
        # SafeComponentContainer's validation logic
        LazyComponentContainer.initialize_with_input_shapes(self, input_shapes)
```

### Usage Examples

#### 1. Simple Immediate Usage
```python
# For immediate use with pre-built components
container = ComponentContainer()
policy = LinearModule(in_features=64, out_features=8, in_key="features", out_key="action")
container.register_component("policy", policy)

td = TensorDict({"features": torch.randn(4, 64)}, batch_size=4)
result = container.execute_component("policy", td)
```

#### 2. Safety-Critical Applications
```python
# Adds comprehensive validation
container = SafeComponentContainer()
container.register_component("policy", policy)  # Validates interface, checks conflicts
```

#### 3. Dynamic Architecture with Shape Inference
```python
# For configurations that need shape inference
container = LazyComponentContainer()

container.register_component_config(
    name="obs_processor",
    component_class=LinearModule,
    config={"out_features": 128},  # in_features will be inferred
    in_keys=["observation"],
    out_keys=["features"]
)

container.register_component_config(
    name="policy", 
    component_class=LinearModule,
    config={"out_features": 8},  # in_features inferred from obs_processor output
    dependencies=["obs_processor"],
    in_keys=["features"],
    out_keys=["action"]
)

# Initialize with input shapes - triggers automatic shape inference
container.initialize_with_input_shapes({"observation": (64,)})

# Now use normally
td = TensorDict({"observation": torch.randn(4, 64)}, batch_size=4)
result = container.execute_component("policy", td)  # Recursively executes obs_processor too
```

#### 4. Production Usage (Full Features)
```python
# Combines safety validation with lazy initialization
container = SafeLazyComponentContainer()

# Register configurations with full validation
container.register_component_config(...)

# Initialize with comprehensive validation
container.initialize_with_input_shapes({"observation": (64,)})
```

## Testing in Isolation

The wrapper-based architecture enables true isolated testing of components at every level:

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

#### Level 1: Pure Component Testing (Zero Setup)
```python
def test_linear_module():
    # Direct testing - no infrastructure needed
    linear = LinearModule(in_features=10, out_features=16, in_key="input", out_key="output")
    tensordict = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
    result = linear(tensordict)
    assert result["output"].shape == (4, 16)

def test_actor_module():
    # Test any MettaModule in complete isolation
    actor = PolicyNetwork(feature_dim=64, action_dim=8, in_key="features", out_key="action")
    tensordict = TensorDict({"features": torch.randn(2, 64)}, batch_size=2)
    result = actor(tensordict)
    assert result["action"].shape == (2, 8)
```

#### Level 2: Container Layer Testing
```python
def test_base_container():
    # Test storage and execution logic
    container = ComponentContainer()
    policy = LinearModule(64, 8, "features", "action")
    container.register_component("policy", policy)
    
    td = TensorDict({"features": torch.randn(4, 64)}, batch_size=4)
    result = container.execute_component("policy", td)
    assert result["action"].shape == (4, 8)

def test_safety_wrapper():
    # Test validation logic independently
    container = SafeComponentContainer()
    
    # Test interface validation
    with pytest.raises(TypeError, match="must be a MettaModule"):
        container.register_component("bad", nn.Linear(10, 5))
    
    # Test dependency validation
    policy = LinearModule(64, 8, "features", "action")
    with pytest.raises(ValueError, match="Dependency 'missing' not found"):
        container.register_component("policy", policy, dependencies=["missing"])

def test_lazy_wrapper():
    # Test shape inference independently
    container = LazyComponentContainer()
    container.register_component_config(
        "policy", LinearModule, {"out_features": 8}, 
        in_keys=["features"], out_keys=["action"]
    )
    container.initialize_with_input_shapes({"features": (64,)})
    
    # Verify component was created with correct dimensions
    assert container.components["policy"].linear.in_features == 64
    assert container.components["policy"].linear.out_features == 8
```

#### Level 3: Integration Testing
```python
def test_full_pipeline():
    # Test complete wrapper composition
    container = SafeLazyComponentContainer()
    
    # Register configurations
    container.register_component_config(...)
    
    # Test initialization with validation
    container.initialize_with_input_shapes({"observation": (64,)})
    
    # Test execution
    result = container.execute_component("policy", td)
```

## Benefits of Wrapper-Based Architecture

### Immediate Benefits

1. **Clear Separation of Concerns**: Each wrapper has a single, well-defined responsibility
   - **Base Container**: Storage + elegant recursive execution
   - **Safety Wrapper**: Validation and error checking
   - **Lazy Wrapper**: Shape inference and deferred initialization

2. **Composability**: Mix and match features as needed
   ```python
   basic = ComponentContainer()              # Simple use cases
   safe = SafeComponentContainer()           # Add validation
   lazy = LazyComponentContainer()           # Add shape inference  
   production = SafeLazyComponentContainer() # Full features
   ```

3. **Independent Testing**: Each layer can be tested in complete isolation
   - Test base container with pre-built components (no shape complexity)
   - Test safety wrapper with malformed inputs
   - Test lazy wrapper with shape inference edge cases
   - Test combinations through inheritance

4. **Easier Debugging**: Problems are isolated to specific layers
   - Shape inference issues → LazyComponentContainer
   - Validation failures → SafeComponentContainer  
   - Execution problems → Base ComponentContainer

### Medium-term Benefits

1. **Incremental Adoption**: Teams can adopt features gradually
   ```python
   # Start simple
   container = ComponentContainer()
   
   # Add safety when needed
   container = SafeComponentContainer()
   
   # Add lazy features when architectures get complex
   container = SafeLazyComponentContainer()
   ```

2. **Feature Extension**: New capabilities through additional wrappers
   ```python
   class ProfiledComponentContainer(ComponentContainer):
       """Adds execution timing and memory profiling."""
   
   class CachedComponentContainer(ComponentContainer):
       """Adds persistent disk caching of component outputs."""
   
   class DistributedComponentContainer(ComponentContainer):
       """Adds multi-GPU component execution."""
   ```

3. **Cleaner Error Messages**: Each wrapper can provide context-specific errors
   - Base container: "Component 'policy' not found"
   - Safety wrapper: "Output key conflict: ['action'] between 'policy' and 'backup_policy'"
   - Lazy wrapper: "Cannot infer shape for key 'features' needed by 'policy'"

### Long-term Benefits

1. **Architectural Flexibility**: Easy to experiment with new container behaviors
2. **Performance Optimization**: Profile and optimize each layer independently
3. **Future PyTorch Compatibility**: Clean interfaces ready for torch.compile
4. **Research Enablement**: Easy to add experimental features as new wrappers

## Implementation Strategy

### Week 1: Wrapper Foundation (20 hours)

**Day 1-2: Base ComponentContainer (8 hours)**
```python
# Base container with elegant recursive execution
class ComponentContainer(TensorDictModuleBase):
    def register_component(self, name, component, dependencies): pass
    def execute_component(self, name, tensordict): pass  # The elegant recursive pattern
    def forward(self, tensordict): pass  # TensorDictModule interface
```

**Day 3: SafeComponentContainer (4 hours)**
```python 
# Validation wrapper
class SafeComponentContainer(ComponentContainer):
    def _validate_component(self, component): pass
    def _validate_dependencies(self, name, dependencies): pass
    def _check_circular_dependencies(self, deps_dict): pass
```

**Day 4-5: LazyComponentContainer (8 hours)**
```python
# Shape inference wrapper  
class LazyComponentContainer(ComponentContainer):
    def register_component_config(self, name, component_class, config, ...): pass
    def initialize_with_input_shapes(self, input_shapes): pass
    def _create_component_with_shapes(self, name, shape_registry): pass
```

### Week 2: Integration + Validation (20 hours)

**Day 1-2: SafeLazyComponentContainer + Testing (8 hours)**
```python
# Combined wrapper + comprehensive test suite
class SafeLazyComponentContainer(LazyComponentContainer, SafeComponentContainer): pass
```

**Day 3-4: MettaAgent Integration (8 hours)**
```python
# Integrate wrapper containers into MettaAgent
class MettaAgent(MettaModule):
    def __init__(self, config):
        self.components = SafeLazyComponentContainer()  # or other variants
```

**Day 5: Documentation + Examples (4 hours)**
```python
# Usage examples for each wrapper type
# Migration guide from current ComponentContainer
# Performance benchmarks
```

### Implementation Benefits

1. **Parallel Development**: Each wrapper can be developed independently
2. **Incremental Testing**: Test each layer as it's built
3. **Risk Mitigation**: If one wrapper has issues, others still work
4. **Clear Milestones**: Each wrapper completion is a deliverable milestone

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