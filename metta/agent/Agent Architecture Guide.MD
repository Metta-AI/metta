# Metta Architecture Enhancement: Streamlining Component Development

## Table of Contents

### Part I: Understanding the Challenge
- [1. Understanding Our Current Architecture](#1-understanding-our-current-architecture)
- [2. The Vision: Our Proposed Enhancement](#2-the-vision-our-proposed-enhancement)
- [3. Core Design Philosophy](#3-core-design-philosophy)

### Part II: The New Architecture
- [4. The New Architecture](#4-the-new-architecture)
- [5. Quick Start: See It in Action](#5-quick-start-see-it-in-action)
- [6. Implementation Guide](#6-implementation-guide)
- [7. Advanced Patterns](#7-advanced-patterns)
- [8. Technical Reference](#8-technical-reference)

### Part III: Future Directions
- [Conclusion: Building on Our Strong Foundation](#conclusion-building-on-our-strong-foundation)
- [Metta Architecture: Future Improvements & Extensions](#metta-architecture-future-improvements--extensions)
- [The Power of Wrapper Patterns: Trainer ↔ Network Communication](#the-power-of-wrapper-patterns-trainer--network-communication)
- [Training Control: Traditional vs Metta Architecture](#training-control-traditional-vs-metta-architecture)
- [Key Selling Points: Why Teams Should Adopt This Architecture](#key-selling-points-why-teams-should-adopt-this-architecture)

---

## 1. Understanding Our Current Architecture

### The LayerBase System Today

Our current LayerBase system has served us well, but as our needs have evolved, we've identified opportunities to streamline the development experience. Let's look at the current workflow:

```python
# What it takes to create a simple linear layer today:
linear_layer = Linear(name="encoder", sources=["observation"], nn_params={"out_features": 128})

# But wait - it's not ready to use yet! We need a multi-step setup process:
linear_layer.setup(source_components={"observation": obs_source})  # Wire dependencies
linear_layer._initialize()  # Initialize internal state
if not linear_layer.ready:  # Check if everything worked
    raise RuntimeError("Component failed to initialize")

# Finally, after all that setup, we can use it:
result = linear_layer.forward(td)
```

### Areas for Enhancement

**Multi-Step Setup Process**: Currently, our LayerBase components require a structured initialization:
- ✓ Comprehensive configuration support
- ✓ Robust dependency management
- ○ Opportunity: Streamline the setup process
- ○ Opportunity: Reduce ceremony for simple use cases

**Configuration Integration**: Our YAML configuration system works well with Hydra:
```yaml
# Current YAML - comprehensive configuration support
encoder:
  _target_: metta.agent.lib.nn_layer_library.Linear
  name: encoder
  sources: [observation]  # Flexible string references
  _nn_params:
    out_features: 128

# This approach provides:
# ✓ Full Hydra integration
# ✓ Flexible component configuration
# ○ Opportunity: Simplify the config-to-component flow
# ○ Opportunity: Make dependencies more explicit
```

**Testing Workflow**: Our current testing approach is thorough but involves setup:
```python
def test_policy_layer():
    # Set up test environment
    mock_encoder = Mock()
    mock_encoder._out_tensor_shape = (None, 128)
    mock_encoder.out_keys = ["features"]
    
    # Create and configure the component
    policy = PolicyLayer(sources=["features"], nn_params={"out_features": 4})
    
    # Complete the setup process
    policy.setup(source_components={"features": mock_encoder})
    policy._in_tensor_shapes = {"features": (128,)}
    policy._initialize()
    
    # Verify readiness
    assert policy.ready
    
    # Test the computation
    td = TensorDict({"features": torch.randn(4, 128)})
    result = policy._forward(td)
```

**Development Considerations**:
- **Research Iteration**: Setup steps can slow down rapid experimentation
- **Team Onboarding**: New team members need time to learn the component lifecycle  
- **Code Maintenance**: Changes may need coordination across multiple concerns
- **Debugging**: Issue isolation can be complex due to the integrated nature

### Opportunities for Evolution

The current LayerBase system has been foundational to our success, but as our team grows and our research becomes more ambitious, we see opportunities to:

- **Accelerate Research Iteration**: Reduce setup time for rapid experimentation
- **Simplify Component Testing**: Enable more direct testing workflows
- **Enhance Code Clarity**: Make dependencies and data flow more explicit
- **Improve Developer Experience**: Streamline the path from idea to implementation

**Our goal is to build on the strengths of the current system while addressing these opportunities for enhancement.**

---

## 2. The Vision: Our Proposed Enhancement

### A Streamlined Development Experience

Building on the solid foundation of our current system, we envision a development experience that feels like this:

```python
# Create components - they work immediately
encoder = LinearModule(64, 128, "observation", "features", activation="relu")
policy = LinearModule(128, 4, "features", "action_logits", activation="softmax")

# Build a network - dependencies are automatic
network = ModularNetwork()
network.add_component("encoder", encoder)
network.add_component("policy", policy)  # automatically knows it needs "features"

# Use it - more complex initialization procedures (e.g. from config) are handled separately
# enables quick programmatic network experiments.
td = TensorDict({"observation": torch.randn(4, 64)})
result = network(td)
```

### The Key Points

**Separation of Concerns**: Instead of one class doing everything, we have specialists:
- **MettaModule**: Pure computation - just neural network math
- **ModularNetwork**: Component orchestration and dependency resolution  
- **NetworkBuilder**: Configuration processing and network construction

**Data-Centric Dependencies**: Components depend on **data keys**, not other components:
```python
# Instead of: "this component depends on that component"
policy = PolicyLayer(sources=["encoder"])  # Brittle string reference

# We have: "this component needs this data"
policy = PolicyLayer(128, 4, in_keys=["features"], out_keys=["action_logits"])  # Semantic dependency
```

**Configuration Handling**: Clean separation between config processing and component logic:
```yaml
# YAML - explicit and clear
input_shapes:
  observation: [64]

components:
  - name: encoder
    class: LinearModule
    config: {out_features: 128, activation: relu}
    in_keys: [sensor_data]
    out_keys: [robot_state]
    
  - name: policy
    class: LinearModule
    config: {out_features: 4, activation: softmax}
    in_keys: [features]  # Clear dependency on data, not component
    out_keys: [action_logits]
```

### What This Enables

**Base Class is Straightforward**: MettaModules are computational objects
that work without additional setup. 
```python
# No setup phase - ready to use immediately
sensor_encoder = LinearModule(256, 128, "sensor_data", "robot_state")
```

**Effortless Testing**: Test what matters without ceremony
```python
def test_policy():
    policy = LinearModule(128, 4, "features", "action_logits")
    td = TensorDict({"features": torch.randn(4, 128)})
    result = policy(td)
    assert "action_logits" in result
```

**Hot-Swappable Architectures**: A/B test different designs seamlessly
```python
# Swap architectures without changing downstream components
for encoder_type in ["linear", "transformer", "cnn"]:
    network.swap_component("encoder", create_encoder(encoder_type))
    performance = evaluate(network)
    print(f"{encoder_type}: {performance}")
```

---

## 3. Core Design Philosophy

### The Three Pillars

Our architecture rests on three fundamental principles that guide every design decision:

#### 1. **PyTorch Inheritance**

After extensive analysis, we've chosen **pure inheritance** (from `nn.Module`) over composition for 
our base class, `MettaModule`. Here's why:

```python
# The RIGHT way - Pure Inheritance
class LinearModule(MettaModule):
    def __init__(self, in_features: int, out_features: int, in_key="input", out_key="output"):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.linear = nn.Linear(in_features, out_features)  # Direct ownership
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        x = td[self.in_keys[0]]
        return {self.out_keys[0]: self.linear(x)}

# Perfect PyTorch ecosystem compatibility:
component = LinearModule(64, 128, "input", "output")
isinstance(component, nn.Module)     # ✓ True
component.parameters()              # ✓ Works perfectly
component.to(device)               # ✓ No wrapper issues
torch.save(component.state_dict()) # ✓ Clean state dict
```

**Why Composition Could Be Limiting**:
```python
# Alternative approach - Composition (considered but not recommended)
class MettaModule(nn.Module):
    def __init__(self, base_module: nn.Module, ...):
        self.base_module = base_module  # Wrapper around another module

# Considerations with this approach:
isinstance(component, nn.Linear)        # Would be False (affects type checking)
component.parameters()                  # More complex parameter discovery
torch.jit.script(component)            # Potential JIT compilation complications
# Additional considerations: function call overhead, state dict structure, device movement
```

#### 2. **Key-Based Data Flow**

Components declare semantic dependencies through data keys, not component references:

```python
# Components say what data they need and produce
encoder = linear_module(64, 128, "observation", "features")  # produces "features"
policy = linear_module(128, 4, "features", "action_logits")  # needs "features"

# The magic: automatic dependency resolution
network.add_component("encoder", encoder)
network.add_component("policy", policy)  # automatically depends on encoder via "features"
```

**Why This is Powerful**:
- **Hot-swappable**: Any component producing "features" can replace the encoder
- **Clear contracts**: Input/output keys define the interface
- **No hidden dependencies**: Data flow is explicit and traceable

#### 3. **Dual Naming for Dual Purposes**

We use both **component names** and **output keys** because they serve different roles. Component names 
refers to the the component as *placed in the architecture*, and are maintained by the network. A better
name may in fact be `node_identifier`. We may want components to have `uid`s so we have a truly universal way 
of referencing them (this is only a concern in the context of experimentation and dynamic networks). 

**Component Names**: Stable operational interfaces
```python
trainer.freeze_component("encoder")      # Always works
network.get_component("encoder").parameters()  # Debugging interface
```

**Output Keys**: Semantic data dependencies
```python
# Any component producing "features" can satisfy policy's dependency
cnn_encoder = ConvModule(3, 64, 3, "observation", "features")
transformer_encoder = AttentionModule(64, 8, "observation", "features")
network.swap_component("encoder", transformer_encoder)  # Just works!
```

### Streamlining the Configuration

We've completely separated configuration concerns from component logic:

**Before**: Configuration was scattered and entangled
```
YAML → Hydra → Component.__init__() → Component.setup() → Ready Check
```

**After**: Clean separation of concerns
```
YAML → NetworkBuilder → Explicit Components (ready to use)
```

This separation means:
- ✓ **Early validation**: NetworkBuilder can validate the configuration before component creation. 
- ✓ **Pure components**: Components only handle computation  
- ✓ **Automatic inference**: Shapes resolved at config time
- ✓ **Simple lifecycle**: Components ready upon instantiation

---

## 4. The New Architecture

### The Big Picture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MettaModule     │    │ ModularNetwork  │    │ NetworkBuilder  │
│                 │    │                 │    │                 │
│ Pure            │    │ Component       │    │ Config          │
│ Computation     │◄───┤ Container       │◄───┤ Processing      │
│                 │    │                 │    │                 │
│ Key-based       │    │ Dependency      │    │ Shape           │
│ Interface       │    │ Resolution      │    │ Inference       │
│                 │    │                 │    │                 │
│ Shape           │    │ Execution       │    │ Network         │
│ Validation      │    │ Orchestration   │    │ Creation        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### How Data Flows

We use an example outside of the MettaAgent's internal logic to illustratate data flow. 
```
Initial TensorDict: {"grid_state": [batch, agents, 8, 8], "agent_position": [batch, agents, 2]}
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Environment Encoder                   │
│  in_keys: ["grid_state"]         out_keys: ["env_features"]    │
│                                                                 │
│  1. Read td["grid_state"] → [batch, agents, 8, 8]             │
│  2. Apply nn.Conv2d(1, 32, 3) + pooling                       │
│  3. td["env_features"] = result → [batch, agents, 128]        │
└─────────────────────────────────────────────────────────────────┘
                                    │
                TensorDict: {"grid_state": [batch, agents, 8, 8],
                            "agent_positions": [batch, agents, 2],
                            "env_features": [batch, agents, 128]}
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Explorer Agent  │    │ Collector Agent │    │ Guardian Agent  │
│ in:["env_       │    │ in:["env_       │    │ in:["env_       │
│    features",   │    │    features",   │    │    features",   │
│    "agent_pos"] │    │    "agent_pos"] │    │    "agent_pos"] │
│out:["explore_   │    │out:["collect_   │    │out:["guard_     │
│     actions"]   │    │     actions"]   │    │     actions"]   │
│                 │    │                 │    │                 │
│ 1. Read shared  │    │ 1. Read shared  │    │ 1. Read shared  │
│    env features │    │    env features │    │    env features │
│ 2. Linear(130,4)│    │ 2. Linear(130,6)│    │ 2. Linear(130,8)│
│    [up,down,    │    │    [gather,drop,│    │    [patrol,     │
│     left,right] │    │     craft,wait, │    │     defend,     │
│ 3. td["explore_ │    │     build,scan] │    │     alert,etc.] │
│    actions"] =  │    │ 3. td["collect_ │    │ 3. td["guard_   │
│    result       │    │    actions"] =  │    │    actions"] =  │
│                 │    │    result       │    │    result       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
Final TensorDict: {
    "grid_state": [batch, agents, 8, 8],        # Original environment
    "agent_positions": [batch, agents, 2],      # Agent positions
    "env_features": [batch, agents, 128],       # Shared environment encoding
    "explore_actions": [batch, n_explorers, 4], # Exploration actions
    "collect_actions": [batch, n_collectors, 6],# Collection actions  
    "guard_actions": [batch, n_guardians, 8]    # Guardian actions
}

Key Insights:
• Shared environment encoding across all agent types
• Each agent type has specialized action spaces
• Scalable to different numbers of each agent type
• Adding new agent types doesn't break existing ones
• Environment changes propagate to all agents automatically
```

### Dependency Resolution Magic

```
Multi-Agent Dependencies (automatic):
┌──────────────┐    produces    ┌──────────────┐
│ Environment  │ ─────────────► │"env_features"│
│ Encoder      │                │              │
└──────────────┘                └──────────────┘
                                        │
                                        │ consumed by all agents
                                        ▼
                                ┌──────────────┐
                                │ Explorer     │
                                │ Collector    │ 
                                │ Guardian     │
                                │ Custom Agent │
                                └──────────────┘

Agent Specialization Example:
┌──────────────┐                ┌──────────────┐
│ Basic Agent  │ ─────────────► │"env_features"│ ◄─ Same encoding!
│ (4 actions)  │                │              │
└──────────────┘                └──────────────┘
                                        │
┌──────────────┐                        │ All agent variants
│ Advanced     │ ─────────────► ┌──────────────┐ work with same
│ Agent (12    │                │"env_features"│ ◄─ environment
│ actions)     │                │              │    understanding
└──────────────┘                └──────────────┘
```

Key Insight: Components depend on DATA KEYS, not each other


### The Three Core Classes

#### MettaModule: Pure Computation


```python
class MettaModule(nn.Module, ABC):
    """Abstract base for all components - handles computation only."""
    
    def __init__(self, in_keys=None, out_keys=None, input_shapes=None, output_shapes=None):
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []
        self.input_shapes = input_shapes or {}
        self.output_shapes = output_shapes or {}

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Framework method - handles validation and TensorDict management."""
        self._validate_shapes(tensordict)
        outputs = self._compute(tensordict)  # Delegate to component logic
        
        for key, value in outputs.items():
            tensordict[key] = value
        return tensordict
    
    @abstractmethod
    def _compute(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        """Component implements its specific computation here."""
        pass
```

#### ModularNetwork: Smart Container

```python
class ModularNetwork(nn.Module):
    """Manages components and resolves dependencies automatically."""

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict()  # Proper PyTorch registration
        self.out_key_to_name = {}  # Maps data keys to component names
        
    def add_component(self, name: str, component: MettaModule):
        """Add component and register its output keys."""
        self.components[name] = component
        for out_key in component.out_keys:
            self.out_key_to_name[out_key] = name  # Track who produces what
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Execute all components in dependency order."""
        for component in self.components.values():
            td = component(td)
        return td
    
    def swap_component(self, name: str, new_component: MettaModule):
        """Hot-swap components while maintaining dependencies."""
        # Update key mappings automatically
```

#### NetworkBuilder: Configuration Engine

```python
class NetworkBuilder:
    """Processes configurations and creates explicit networks."""
    
    def build_from_config(self, config: Dict[str, Any]) -> ModularNetwork:
        """Three-phase process: validate → infer shapes → create network"""
        self._validate_config(config)
        resolved_configs = self._resolve_shapes(config)
        return self._create_network(resolved_configs)
```

### From Problem to Solution

**Before**: Tangled responsibilities
```python
class LayerBase(nn.Module):
    # Everything mixed together
    def _forward(self, td) -> TensorDict      # Computation
    def setup(self, sources) -> None          # Infrastructure  
    def _initialize() -> None                 # Lifecycle
    @property ready(self) -> bool            # State management
```

**After**: Clean separation
```python
# Computation only
class LinearModule(MettaModule):
    def _compute(self, td) -> Dict[str, torch.Tensor]:
        return {self.out_keys[0]: self.linear(td[self.in_keys[0]])}

# Infrastructure only  
class ModularNetwork(nn.Module):
    def add_component(self, name, component): ...
    def forward(self, td): ...

# Configuration only
class NetworkBuilder:
    def build_from_config(self, config): ...
```

---

## 5. Quick Start: See It in Action

Now that you understand the philosophy, let's see how simple and powerful the new system is in practice:

### Your First Network

```python
# Step 1: Create components (they work immediately)
state_encoder = LinearModule(64, 128, "observation", "state_features", activation="relu")
policy_head = LinearModule(128, 4, "state_features", "action_logits", activation="softmax")
value_head = LinearModule(128, 1, "state_features", "state_value")

# Step 2: Compose into a network (dependencies are automatic)
network = ModularNetwork()
network.add_component("encoder", state_encoder)        # produces "state_features"
network.add_component("policy", policy_head)          # consumes "state_features" → actor
network.add_component("critic", value_head)           # consumes "state_features" → critic

# Step 3: Use it (no setup required)
td = TensorDict({"observation": torch.randn(4, 64)})
result = network(td)

print(result.keys())
# Output: dict_keys(['observation', 'state_features', 'action_logits', 'state_value'])
# Perfect actor-critic setup: shared features, separate policy/value outputs
```

### Configuration-Driven Networks

```yaml
# config.yaml - Actor-Critic Configuration
input_shapes:
  observation: [64]

components:
  - name: state_encoder
    class: LinearModule
    config: {out_features: 128, activation: relu}
    in_keys: [observation]
    out_keys: [state_features]
    
  - name: policy_head
    class: LinearModule
    config: {out_features: 4, activation: softmax}
    in_keys: [state_features]
    out_keys: [action_logits]
    
  - name: value_head
    class: LinearModule
    config: {out_features: 1}
    in_keys: [state_features]  # Shared features for critic
    out_keys: [state_value]
```

```python
# Build from config (with automatic shape inference)
builder = NetworkBuilder()
network = builder.build_from_config(config)
result = network(td)  # Works immediately
```

### The Power of Hot-Swapping

```python
# A/B test different RL architectures effortlessly
encoder_architectures = {
    "linear": lambda: LinearModule(64, 128, "observation", "state_features"),
    "deeper": lambda: SequentialModule([
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU()
    ], "observation", "state_features", (64,), (128,)),
    "attention": lambda: AttentionModule(64, 8, "observation", "state_features")
}

for name, create_encoder in encoder_architectures.items():
    print(f"Testing {name} state encoder...")
    network.swap_component("state_encoder", create_encoder())
    
    # Policy and value heads are unchanged - they just need "state_features"
    performance = evaluate_actor_critic(network)
    print(f"{name}: reward={performance['reward']:.2f}, value_error={performance['value_error']:.4f}")
```

### Testing is Now Trivial

```python
# Before: Complex setup required
def test_old_way():
    mock_encoder = Mock()
    policy = PolicyLayer(sources=["encoder"])
    policy.setup(source_components={"encoder": mock_encoder})
    # ... lots more setup

# After: Direct testing
def test_new_way():
    policy = LinearModule(128, 4, "features", "action_logits")
    td = TensorDict({"features": torch.randn(4, 128)})
    result = policy(td)
    assert "action_logits" in result  # Done!
```

---

## 6. Implementation Guide

### Creating Custom Components

The heart of the system is creating MettaModule subclasses. Here's an example for an attention module:

#### Components Inherit from MettaModule

When you need custom logic, inherit directly:

```python

class AttentionModule(nn.MultiheadAttention):
    """Pure attention module - inherits from MultiheadAttention."""
    
    def __init__(self, embed_dim: int, num_heads: int, 
                 in_key: str = "input", out_key: str = "attended"):
        super().__init__(embed_dim, num_heads)
        
        # MettaModule interface
        self.in_keys = [in_key]
        self.out_keys = [out_key]
        self.input_shapes = {in_key: (embed_dim,)}
        self.output_shapes = {out_key: (embed_dim,)}
    
    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.in_keys[0]]
        attn_output, _ = super().forward(x, x, x)
        td[self.out_keys[0]] = attn_output
        return td

# Perfect PyTorch compatibility
attention = AttentionModule(128, 8, "features", "attended_features")
attention.parameters()  # All parameters visible
attention.to(device)    # Device movement works perfectly
```

### Advanced Component Patterns

#### Multi-Output Components

```python
class LSTMModule(MettaModule):
    """LSTM with state management."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(
            in_keys=["sequence", "hidden_state", "cell_state"],
            out_keys=["output", "new_hidden", "new_cell"]
        )
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
    
    def forward(self, td: TensorDict) -> TensorDict:
        # Initialize states if not present
        if "hidden_state" not in td:
            batch_size = td["sequence"].shape[0]
            td["hidden_state"] = torch.zeros(1, batch_size, self.hidden_size)
            td["cell_state"] = torch.zeros(1, batch_size, self.hidden_size)
        
        return super().forward(td)  # Call parent's validation + _compute
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        x = td["sequence"]
        h = td["hidden_state"]
        c = td["cell_state"]
        
        output, (new_h, new_c) = self.lstm(x, (h, c))
        
        return {
            "output": output,
            "new_hidden": new_h,
            "new_cell": new_c
        }
```

#### External Library Integration

```python
# Create MettaModules using inheritance
class PufferLibModule(MettaModule):
    def __init__(self, puffer_env):
        super().__init__(
            in_keys=["raw_obs"],
            out_keys=["processed_obs"],
            input_shapes={"raw_obs": puffer_env.observation_space.shape},
            output_shapes={"processed_obs": puffer_env.observation_space.shape}
        )
        self.processor = PufferLibProcessor(puffer_env)
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        raw_obs = td["raw_obs"]
        processed = self.processor(raw_obs)
        return {"processed_obs": processed}

class HuggingFaceModule(MettaModule):
    """Integrate HuggingFace models seamlessly."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(
            in_keys=["text_input"],
            out_keys=["text_embeddings"],
            input_shapes={"text_input": (-1,)},  # Variable length
            output_shapes={"text_embeddings": (768,)}  # BERT embedding size
        )
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        text_input = td["text_input"]
        tokens = self.tokenizer(text_input, return_tensors="pt", padding=True)
        embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
        return {"text_embeddings": embeddings}

# Use exactly like any other component
hf_encoder = HuggingFaceModule("bert-base-uncased")
network.add_component("text_encoder", hf_encoder)
```

### Multi-Head Networks

One of the most powerful patterns is shared computation with multiple heads:

```python
# Hierarchical RL: Shared state understanding feeding multiple control levels
state_encoder = LinearModule(64, 256, "observation", "state_representation", activation="relu")

# High-level controller: decides abstract goals/subgoals
high_level_controller = LinearModule(256, 8, "state_representation", "subgoal_logits")
# Low-level controller: executes primitive actions to achieve subgoals  
low_level_controller = LinearModule(264, 4, "augmented_state", "action_logits", activation="softmax")
# Value estimation for both levels
value_estimator = LinearModule(256, 1, "state_representation", "state_value")

# All controllers depend on shared state understanding
network = ModularNetwork()
network.add_component("encoder", state_encoder)
network.add_component("high_level", high_level_controller)
network.add_component("low_level", low_level_controller)    # Uses state + subgoal
network.add_component("critic", value_estimator)

# Single forward pass computes hierarchical control
td = TensorDict({
    "observation": torch.randn(4, 64),
    "augmented_state": torch.randn(4, 264)  # state_representation + current_subgoal
})
result = network(td)
# Contains: state_representation, subgoal_logits, action_logits, state_value
```

---

## 7. Advanced Patterns

### Wrapper Patterns (TorchRL-Inspired)

One of our most powerful features is the wrapper system, inspired by TorchRL's transforms. Wrappers let you modify component behavior without changing the core logic:

#### Output Processing Wrappers

```python
class ClippingWrapper(MettaModule):
    """Clips component outputs to specified ranges."""
    
    def __init__(self, wrapped_component: MettaModule, clip_ranges: Dict[str, tuple]):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.clip_ranges = clip_ranges  # {out_key: (min_val, max_val)}
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        # Get outputs from wrapped component
        outputs = self.wrapped_component._compute(td)
        
        # Apply clipping to specified outputs
        for out_key, value in outputs.items():
            if out_key in self.clip_ranges:
                min_val, max_val = self.clip_ranges[out_key]
                outputs[out_key] = torch.clamp(value, min_val, max_val)
        
        return outputs

# Usage: Stack wrappers for complex behavior
policy = LinearModule(128, 4, "features", "action_logits", activation="softmax")
clipped_policy = ClippingWrapper(policy, {"action_logits": (-1.0, 1.0)})
```

#### Decorator Patterns

For elegant composition, use decorators:

```python
def clipping(**clip_ranges):
    """Decorator that applies clipping to component outputs."""
    def decorator(component_or_class):
        if callable(component_or_class) and not isinstance(component_or_class, MettaModule):
            # Applied to class or constructor
            def wrapped_constructor(*args, **kwargs):
                component = component_or_class(*args, **kwargs)
                return ClippingWrapper(component, clip_ranges)
            return wrapped_constructor
        else:
            # Applied to component instance
            return ClippingWrapper(component_or_class, clip_ranges)
    return decorator

def normalization(norm_type: str = "l2"):
    """Decorator that applies normalization to component outputs."""
    def decorator(component_or_class):
        if callable(component_or_class) and not isinstance(component_or_class, MettaModule):
            def wrapped_constructor(*args, **kwargs):
                component = component_or_class(*args, **kwargs)
                return NormalizationWrapper(component, norm_type)
            return wrapped_constructor
        else:
            return NormalizationWrapper(component_or_class, norm_type)
    return decorator

# Usage: Stack decorators for enhanced components
@clipping(action_logits=(-1.0, 1.0))
@normalization("l2")
def create_enhanced_policy():
    return LinearModule(128, 4, "features", "action_logits", activation="softmax")

policy = create_enhanced_policy()
```

### Memory and State Management

```python
class MemoryAugmentedEncoder(MettaModule):
    """Component with internal memory for sequential processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, memory_size: int):
        super().__init__(
            in_keys=["observation", "prev_memory"],
            out_keys=["encoding", "updated_memory"],
            input_shapes={
                "observation": (input_dim,), 
                "prev_memory": (memory_size, hidden_dim)
            },
            output_shapes={
                "encoding": (hidden_dim,), 
                "updated_memory": (memory_size, hidden_dim)
            }
        )
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.memory_attention = nn.MultiheadAttention(hidden_dim, 8)
        self.memory_update = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, td: TensorDict) -> TensorDict:
        # Initialize memory if not present
        if "prev_memory" not in td:
            batch_size = td["observation"].shape[0]
            td["prev_memory"] = torch.zeros(batch_size, *self.input_shapes["prev_memory"])
        
        return super().forward(td)  # Calls our _compute
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        obs = td["observation"]
        memory = td["prev_memory"]
        
        # Encode observation
        encoded = self.encoder(obs)
        
        # Attend to memory
        attended, _ = self.memory_attention(encoded.unsqueeze(0), memory, memory)
        attended = attended.squeeze(0)
        
        # Update memory (simplified for clarity)
        updated_memory = self.memory_update(attended, memory.mean(dim=1, keepdim=True))
        
        return {
            "encoding": attended,
            "updated_memory": updated_memory.unsqueeze(1).expand(-1, memory.shape[1], -1)
        }
```

### Configuration-Driven Wrapper Integration

```python
# Wrappers can be specified in YAML configurations
config = {
    "components": [
        {
            "name": "policy",
            "class": "LinearModule",
            "config": {"out_features": 4, "activation": "softmax"},
            "wrappers": [
                {"type": "clipping", "config": {"action_logits": [-1.0, 1.0]}},
                {"type": "normalization", "config": {"norm_type": "l2"}}
            ],
            "in_keys": ["features"],
            "out_keys": ["action_logits"]
        }
    ]
}

# NetworkBuilder automatically applies wrappers during construction
network = NetworkBuilder().build_from_config(config)
```

---

## 8. Technical Reference

### Core Class Implementations

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from tensordict import TensorDict

class MettaModule(nn.Module, ABC):
    """Abstract base class for all Metta components."""
    
    def __init__(self, in_keys=None, out_keys=None, input_shapes=None, output_shapes=None):
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []
        self.input_shapes = input_shapes or {}
        self.output_shapes = output_shapes or {}

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Framework method - handles validation and TensorDict management."""
        self._validate_shapes(tensordict)
        outputs = self._compute(tensordict)
        
        for key, value in outputs.items():
            tensordict[key] = value
        return tensordict
    
    @abstractmethod
    def _compute(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        """Subclasses implement their specific computation logic here."""
        pass
    
    def _validate_shapes(self, tensordict: TensorDict):
        """Validate input tensor shapes against specifications."""
        for key in self.in_keys:
            if key in self.input_shapes:
                expected_shape = self.input_shapes[key]
                actual_shape = tensordict[key].shape[1:]  # Skip batch dimension
                if actual_shape != expected_shape:
                    raise ValueError(f"Shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}")

class LinearModule(MettaModule):
    """Linear transformation component with optional activation."""
    
    def __init__(self, in_features: int, out_features: int, 
                 in_key: str = "input", out_key: str = "output", 
                 activation: str = None, bias: bool = True):
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
            input_shapes={in_key: (in_features,)},
            output_shapes={out_key: (out_features,)}
        )
        
        # Create layers based on activation
        if activation is None:
            self.network = nn.Linear(in_features, out_features, bias=bias)
        else:
            layers = [nn.Linear(in_features, out_features, bias=bias)]
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "softmax":
                layers.append(nn.Softmax(dim=-1))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            self.network = nn.Sequential(*layers)
    
    def _compute(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        x = tensordict[self.in_keys[0]]
        output = self.network(x)
        return {self.out_keys[0]: output}

class ConvModule(MettaModule):
    """2D Convolution component."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 in_key: str = "image", out_key: str = "features", **conv_kwargs):
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
            input_shapes={in_key: (in_channels, None, None)},  # Height/width variable
            output_shapes={out_key: (out_channels, None, None)}  # Would need inference
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
    
    def _compute(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        x = tensordict[self.in_keys[0]]
        output = self.conv(x)
        return {self.out_keys[0]: output}

class SequentialModule(MettaModule):
    """Sequential composition of layers."""
    
    def __init__(self, layers: List[nn.Module], in_key: str, out_key: str,
                 input_shape: tuple, output_shape: tuple):
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
            input_shapes={in_key: input_shape},
            output_shapes={out_key: output_shape}
        )
        self.sequential = nn.Sequential(*layers)
    
    def _compute(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        x = tensordict[self.in_keys[0]]
        output = self.sequential(x)
        return {self.out_keys[0]: output}

class ModularNetwork(nn.Module):
    """Container for modular components with dependency resolution."""

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict()
        self.out_key_to_name = {}
        
    def forward(self, td: TensorDict) -> TensorDict:
        """Execute all components in dependency order."""
        for component in self.components.values():
            td = component(td)
        return td
    
    def add_component(self, name: str, component: MettaModule):
        """Add a component to the network."""
        self.components[name] = component
        for out_key in component.out_keys:
            self.out_key_to_name[out_key] = name

    def get_component(self, name: str) -> MettaModule:
        """Get a component by name."""
        return self.components[name]

    def swap_component(self, name: str, new_component: MettaModule):
        """Replace an existing component."""
        if name in self.components:
            old_component = self.components[name]
            
            self.components[name] = new_component
            for out_key in old_component.out_keys:
                del self.out_key_to_name[out_key]
            for out_key in new_component.out_keys:
                self.out_key_to_name[out_key] = name

class NetworkBuilder:
    """Processes configurations and creates explicit networks."""
    
    def __init__(self):
        self.shape_inference_registry = {
            "LinearModule": self._infer_linear_shapes,
            "ConvModule": self._infer_conv_shapes,
        }
        
        self.class_registry = {
            "LinearModule": LinearModule,
            "ConvModule": ConvModule,
            "SequentialModule": SequentialModule,
        }
    
    def build_from_config(self, config: Dict[str, Any]) -> ModularNetwork:
        """Build a ModularNetwork from configuration."""
        network = ModularNetwork()
        
        for component_config in config["components"]:
            class_name = component_config["class"]
            component_params = component_config["config"].copy()
            
            # Infer missing parameters
            if class_name in self.shape_inference_registry:
                component_params = self.shape_inference_registry[class_name](
                    component_params, 
                    config.get("input_shapes", {}),
                    component_config["in_keys"],
                    component_config["out_keys"]
                )
            
            # Add key parameters
            component_params["in_key"] = component_config["in_keys"][0]
            component_params["out_key"] = component_config["out_keys"][0]
            
            # Create component
            if class_name in self.class_registry:
                component_class = self.class_registry[class_name]
                component = component_class(**component_params)
            else:
                raise ValueError(f"Unknown class: {class_name}")
            
            network.add_component(component_config["name"], component)
        
        return network
    
    def _infer_linear_shapes(self, config, input_shapes, in_keys, out_keys):
        """Automatic shape inference for linear modules."""
        if "in_features" not in config:
            input_shape = input_shapes[in_keys[0]]
            config["in_features"] = input_shape[-1]
        return config
    
    def _infer_conv_shapes(self, config, input_shapes, in_keys, out_keys):
        """Automatic shape inference for conv modules."""
        if "in_channels" not in config:
            input_shape = input_shapes[in_keys[0]]
            config["in_channels"] = input_shape[0]
        return config
```

---

## Conclusion: Building on Our Strong Foundation

This proposed architecture enhancement builds thoughtfully on the solid foundation our team has established with LayerBase. By introducing specialized concerns and key-based data flow, we can preserve the power and flexibility of our current system while streamlining the development experience.

The goal is evolutionary improvement: taking the proven concepts from LayerBase and refining them to support our growing team and increasingly ambitious research goals.

**Key Principles of This Enhancement**:

**Preserve What Works**: Maintain the configurability, dependency management, and robustness that make LayerBase valuable.

**Simplify Common Cases**: Make the simple things simpler while keeping complex things possible.

**Enable Growth**: Support our expanding team with more intuitive patterns and faster iteration cycles.

**Future-Ready Architecture**: Create a foundation that can evolve with our research needs and scale with our ambitions.

This enhancement represents a natural evolution of our architecture - building on proven foundations while opening new possibilities for innovation and growth.

---

# Metta Architecture: Future Improvements & Extensions

## How the Architecture is "Improvement-Ready"

The current Metta architecture is designed for extensibility. All improvements work as **extensions** of the existing system - no fundamental changes needed. The modular design and key-based data flow naturally support enhancements through wrappers, alternative execution engines, and signal passing via TensorDict.

---

## **Advanced Execution Engine**

Current networks execute components sequentially, even when some could run in parallel. An optimized execution engine would analyze the dependency graph and execute independent components simultaneously (like policy and value heads that both consume encoder features). This could provide 2-5x speedups for multi-head architectures. The engine would also apply operator fusion (combining linear + activation layers) and memory optimizations. Since components already declare their dependencies via keys, building this execution graph is straightforward.

```python
# Just swap the execution engine
network = ModularNetwork()
network.add_component("encoder", encoder)

# Default execution
result = network(td)

# Optimized execution (future)
optimized_network = OptimizedExecutionEngine(network)
result = optimized_network(td)  # 2-5x faster with parallel execution
```

## **Symbolic Shape System**

Manual shape management becomes error-prone with complex, dynamic architectures. A symbolic shape system would let you define shapes with variables (like batch_size, sequence_length) and automatically propagate them through the network. This enables building networks that work with any input size and catches shape mismatches at configuration time rather than runtime. It's especially valuable for transformer architectures where sequence lengths vary. The NetworkBuilder already has shape inference hooks, making this a natural extension.

```python
# Enhanced NetworkBuilder with symbolic shapes
batch_size = ShapeVariable("batch", constraints=[(">=", 1)])
seq_len = ShapeVariable("seq_len", constraints=[("<=", 2048)])

config = {
    "input_shapes": {"observation": (batch_size, seq_len, 512)},
    "components": [...] # Shapes propagate symbolically
}
network = NetworkBuilder().build_from_config(config)
```

## **Performance Execution Modes**

Different deployment scenarios need different optimizations - inference servers want maximum throughput while edge devices need minimal memory usage. Performance execution modes would pre-compile networks for specific optimization targets. Throughput mode would pre-allocate all tensors and use optimized kernels, while memory mode would use in-place operations and gradient checkpointing. This is especially important for production deployments where 10-50% performance improvements directly impact costs and user experience.

```python
# High-performance executor (future enhancement)
fast_executor = PerformantExecutor(network, mode="throughput")
outputs = fast_executor.execute_fast({"observation": obs})  # 10-50% faster

# Memory-optimized mode
memory_executor = PerformantExecutor(network, mode="memory")
outputs = memory_executor.execute_fast(inputs)  # Lower memory usage
```

## **Programmable Configurations**

Static YAML configs become limiting for systematic experimentation and hyperparameter search. Programmable configurations would let you write Python functions that generate configs based on parameters, enabling automatic hyperparameter sweeps and conditional architectures. This is crucial for research where you need to test hundreds of configuration combinations. The approach maintains the benefits of declarative configs while adding the power of programmatic generation. (I think we already have this)

```python
# Config functions instead of static YAML
@sweep("encoder_size", [64, 128, 256])
@sweep("learning_rate", [1e-4, 3e-4, 1e-3])
def create_config(encoder_size, learning_rate):
    return {
        "components": [
            {"name": "encoder", "class": "LinearModule", 
             "config": {"out_features": encoder_size}},
            # ... dynamic config based on parameters
        ]
    }

configs = create_config.generate_all()  # 9 configurations for hyperparameter search
```

## **Rich Developer Experience**

Debugging neural networks is notoriously difficult - you need to understand data flow, tensor shapes, memory usage, and performance bottlenecks. Rich developer tooling would provide interactive visualization of network architecture, real-time execution tracing, and performance profiling. This dramatically speeds up development and debugging, especially for complex multi-component networks. The modular architecture makes this possible since each component has explicit inputs/outputs and dependency relationships.

```python
# Network visualization and debugging
from metta.viz import NetworkVisualizer
from metta.debug import NetworkDebugger

viz = NetworkVisualizer(network)
viz.show_architecture()  # Interactive graph
viz.show_data_flow(sample_input)  # Trace execution

debugger = NetworkDebugger(network)
with debugger.trace_execution():
    result = network(td)  # Shows shapes, memory, timing per component
```

## **Production Features**

Research code rarely has the monitoring, A/B testing, and deployment capabilities needed for production systems. Production features would add automatic performance monitoring, traffic splitting for model comparison, and one-click deployment with optimization. This is essential for moving from research prototypes to production AI systems. The wrapper pattern naturally supports adding these concerns without modifying core components.

```python
# A/B testing framework
class ABTestingFramework:
    def add_variant(self, name: str, network: ModularNetwork, traffic_pct: float):
        self.variants[name] = (network, traffic_pct)
    
    def predict(self, inputs):
        variant = self.traffic_splitter.select_variant()
        return self.variants[variant][0](inputs)

# Deploy with optimization
production_network = ProductionWrapper(network)  # Adds monitoring
tensorrt_network = TensorRTWrapper(production_network)  # Adds optimization
production_network.deploy("https://api.mycompany.com/predict")
```

## **Ecosystem Integration**

Modern ML development relies on external tools like HuggingFace models, experiment tracking (Weights & Biases), and distributed training frameworks. Deep ecosystem integration would provide seamless interoperability with these tools through standardized interfaces. This reduces integration overhead and lets teams leverage best-in-class tools for each aspect of the ML pipeline. The MettaModule interface already supports wrapping external models, making deeper integration straightforward.

```python
# HuggingFace integration
@hf_wrapper("bert-base-uncased")
class BertEncoder(MettaModule):
    pass

# Experiment tracking
from metta.integrations.wandb import WandbTracker
tracker = WandbTracker(project="metta-experiments")
tracker.log_architecture(network)  # Automatic logging

# Distributed training
trainer = DistributedTrainer(network, strategy="ddp", precision="fp16")
trainer.fit(train_loader, val_loader)
```

## **Neural Architecture Search**

Manually designing architectures is time-consuming and often suboptimal compared to automated search. Neural Architecture Search would automatically explore different combinations of components, layer sizes, and connectivity patterns to find optimal architectures for specific tasks. This is particularly valuable for new domains where optimal architectures aren't known. The modular component system makes NAS natural - the search space is just different combinations of MettaModule components.

```python
# NAS integration
from metta.nas import ArchitectureSearchSpace

search_space = ArchitectureSearchSpace()
search_space.add_choice("encoder_type", ["LinearModule", "AttentionModule", "ConvModule"])
search_space.add_range("encoder_size", 64, 512, step=64)

nas_controller = NASController(search_space)
best_architecture = nas_controller.search(train_data, val_data, budget=100)

# Best architecture becomes a regular MettaModule network
best_network = nas_controller.build_network(best_architecture)
```

## **Model Ensembling**

Single models often underperform compared to ensembles of diverse architectures, but ensembling is typically complex to implement and deploy. Automatic model ensembling would train multiple diverse architectures and combine their predictions intelligently. This is especially valuable for high-stakes applications where prediction accuracy is critical. The component-swapping capability makes it easy to create diverse architectures from the same base components.

```python
# Ensemble of different architectures
ensemble = EnsembleNetwork([
    create_linear_network(),
    create_transformer_network(), 
    create_convnet_network()
])
ensemble.set_aggregation_strategy("weighted_average")
ensemble.fit_ensemble_weights(val_data)

# Still just a ModularNetwork under the hood
result = ensemble(td)  # Automatic ensemble prediction
```

---

# The Power of Wrapper Patterns: Trainer ↔ Network Communication

## Challenge: Complex Training Adaptation Pipeline

One challenge in deep learning systems is creating adaptive training pipelines where the trainer needs to communicate state, the network needs to adapt its behavior, and components need to modify their computation based on training signals.

Traditional approaches require tight coupling between trainer and network components. The wrapper pattern provides an elegant solution.

## **Major Use Case: Component-Level Training Control**

One of the most compelling applications of the wrapper pattern is **dynamic component-level training control**. Modern deep learning often requires fine-grained control over training behavior - different learning rates per component, selective freezing during curriculum learning, gradient accumulation for memory management, or putting components in eval mode while others train. Traditional architectures make this clunky with global settings or scattered per-component logic.

The Metta architecture provides an elegant solution: **trainer sets intent → agent wraps dynamically → components report back actual behavior**.

### **The Pattern in Action**

```python
# Trainer sets component-level training flags
training_td = TensorDict({
    "observation": batch.observation,
    "_component_lr": {"encoder": 1e-4, "policy": 3e-4, "value": 1e-5},
    "_frozen_components": ["encoder"],  # Freeze encoder this epoch
    "_gradient_accumulation": {"policy": 4},  # Accumulate grads for policy
    "_eval_mode_components": ["auxiliary"]  # Aux head in eval mode
})

# Agent reads flags and wraps components accordingly
class TrainingControlAgent:
    def forward(self, td: TensorDict) -> TensorDict:
        self._apply_training_controls(td)  # Dynamic wrapping based on signals
        return self.network(td)
    
    def _apply_training_controls(self, td: TensorDict):
        # Freeze components dynamically
        frozen = td.get("_frozen_components", [])
        for component_name in frozen:
            if not self._is_wrapped(component_name, "frozen"):
                original = self.network.get_component(component_name)
                frozen_component = NoGradWrapper(original, component_name)
                self.network.swap_component(component_name, frozen_component)
        
        # Set component learning rates
        lr_config = td.get("_component_lr", {})
        for component_name, lr in lr_config.items():
            if not self._is_wrapped(component_name, "lr_control"):
                component = self.network.get_component(component_name)
                lr_component = LearningRateWrapper(component, lr, component_name)
                self.network.swap_component(component_name, lr_component)

# Trainer receives rich feedback about actual training configuration
result = self.agent(training_td)
actual_lrs = result.get("_component_lr_actual", {})
component_status = result.get("_component_status", {})
```

### **Training Control Wrappers**

#### **1. No-Grad Wrapper for Selective Freezing**
```python
class NoGradWrapper(MettaModule):
    """Prevents gradient computation for wrapped component."""
    
    def __init__(self, wrapped_component: MettaModule, component_name: str):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.component_name = component_name
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.wrapped_component._compute(td)
        return outputs
    
    def forward(self, td: TensorDict) -> TensorDict:
        td = super().forward(td)
        
        # Report frozen status back to trainer
        td["_component_status"] = td.get("_component_status", {})
        td["_component_status"][self.component_name] = "frozen"
        return td
```

#### **2. Learning Rate Control Wrapper**
```python
class LearningRateWrapper(MettaModule):
    """Communicates component-specific learning rate to trainer."""
    
    def __init__(self, wrapped_component: MettaModule, learning_rate: float, component_name: str):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.learning_rate = learning_rate
        self.component_name = component_name
    
    def forward(self, td: TensorDict) -> TensorDict:
        td = self.wrapped_component(td)
        
        # Report learning rate back to trainer
        td["_component_lr_actual"] = td.get("_component_lr_actual", {})
        td["_component_lr_actual"][self.component_name] = self.learning_rate
        
        return td
```

#### **3. Gradient Accumulation Wrapper**
```python
class GradientAccumulationWrapper(MettaModule):
    """Handles component-level gradient accumulation."""
    
    def __init__(self, wrapped_component: MettaModule, accumulation_steps: int, component_name: str):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.accumulation_steps = accumulation_steps
        self.component_name = component_name
        self.current_step = 0
    
    def forward(self, td: TensorDict) -> TensorDict:
        self.current_step += 1
        should_update = self.current_step % self.accumulation_steps == 0
        
        td = self.wrapped_component(td)
        
        # Report accumulation status back to trainer
        td["_grad_accumulation_status"] = td.get("_grad_accumulation_status", {})
        td["_grad_accumulation_status"][self.component_name] = {
            "step": self.current_step,
            "should_update": should_update
        }
        
        return td
```

### **Smart Trainer with Component-Level Control**

```python
class ComponentAwareTrainer:
    def __init__(self, agent, training_phases):
        self.agent = agent
        self.training_phases = training_phases
        self.component_optimizers = {
            "encoder": torch.optim.Adam(agent.network.get_component("encoder").parameters()),
            "policy": torch.optim.Adam(agent.network.get_component("policy").parameters()),
            "critic": torch.optim.Adam(agent.network.get_component("critic").parameters())
        }
    
    def training_step(self, batch, phase="phase_1"):
        # Get training configuration for current phase
        phase_config = self.training_phases[phase]
        
        # Set component training controls
        td = TensorDict({
            "observation": batch.observation,
            "_component_lr": phase_config.get("component_lr", {}),
            "_frozen_components": phase_config.get("frozen_components", []),
            "_gradient_accumulation": phase_config.get("gradient_accumulation", {}),
            "_eval_mode_components": phase_config.get("eval_mode_components", [])
        })
        
        result = self.agent(td)
        
        # Trainer receives actual training configuration
        actual_lrs = result.get("_component_lr_actual", {})
        component_status = result.get("_component_status", {})
        grad_status = result.get("_grad_accumulation_status", {})
        
        # Update optimizers based on component feedback
        for component_name, lr in actual_lrs.items():
            if component_name in self.component_optimizers:
                for param_group in self.component_optimizers[component_name].param_groups:
                    param_group['lr'] = lr
        
        # Compute RL losses and update only components that should be updated
        policy_loss = self.compute_policy_loss(result, batch)
        value_loss = self.compute_value_loss(result, batch)
        total_loss = policy_loss + value_loss
        total_loss.backward()
        
        for component_name, optimizer in self.component_optimizers.items():
            # Check if component is frozen
            if component_status.get(component_name) == "frozen":
                continue
                
            # Check gradient accumulation status
            if component_name in grad_status:
                if grad_status[component_name]["should_update"]:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                # Normal update
                optimizer.step()
                optimizer.zero_grad()
        
        return result
```

### **Configuration-Driven Training Phases**

```yaml
# training_phases.yaml - RL Curriculum Learning Phases
training_phases:
  environment_learning:
    component_lr: 
      encoder: 1e-3      # Fast environment learning
      policy: 3e-4       # Standard policy learning
      critic: 1e-4       # Slow value learning
    frozen_components: []
    
  policy_optimization:
    component_lr:
      encoder: 1e-4      # Slower environment updates
      policy: 3e-4       # Focus on policy improvement
      critic: 1e-4
    frozen_components: []
    gradient_accumulation:
      encoder: 4         # Stabilize environment understanding
    
  fine_tuning:
    component_lr:
      encoder: 1e-5      # Very slow environment changes
      policy: 1e-4       # Fine-tune policy
      critic: 1e-5       # Stabilize value estimates
    frozen_components: [encoder]  # Freeze environment understanding
    eval_mode_components: [auxiliary]  # Auxiliary tasks in eval mode
    
  exploitation:
    component_lr:
      policy: 1e-5       # Minimal policy changes
      critic: 1e-5       # Minimal value changes
    frozen_components: [encoder, auxiliary]  # Pure exploitation mode
```

### **Why This Pattern is Powerful**

1. **Solves Real Training Problems**: Fine-grained training control is essential for curriculum learning, transfer learning, and memory management
2. **Clean Separation of Concerns**: Trainer decides strategy, agent implements mechanism, components report status
3. **Dynamic and Flexible**: Training strategy can change mid-training based on performance or curriculum
4. **Bidirectional Contract**: Trainer sets intent, receives confirmation of actual behavior
5. **Component Agnostic**: Works with any MettaModule without modification
6. **Configuration-Driven**: Different training strategies can be A/B tested via YAML

This showcases the true power of the modular architecture: **complex training strategies become simple configuration changes**, while maintaining clean separation between training logic (trainer), orchestration (agent), and computation (network components).

## **Basic Trainer ↔ Network Communication**

Traditional training systems have tight coupling between trainers and networks, making it hard to implement adaptive training strategies. The TensorDict-based communication pattern lets trainers inject signals (gradient norms, training phase, performance metrics) that components can read and respond to. This enables adaptive training where network behavior changes based on training dynamics without requiring explicit trainer-network APIs. The communication is bidirectional - components can also report status back to the trainer. This pattern is essential for sophisticated training strategies like curriculum learning, adaptive regularization, and performance-based model selection.

```python
# Trainer signals flow through TensorDict
class EnhancedTrainer:
    def training_step(self, batch):
        # Trainer adds signals to TensorDict
        td = TensorDict({
            "observation": batch.observation,
            "_training_phase": "finetuning",
            "_gradient_norms": self.get_gradient_norms(),
            "_step": self.current_step,
            "_performance_metrics": self.get_recent_metrics()
        })
        
        # Agent can adapt based on trainer signals
        result = self.agent(td)
        return result

class AdaptiveAgent:
    def forward(self, td: TensorDict) -> TensorDict:
        # Agent reads trainer signals and adapts network
        if td.get("_gradient_norms", {}).get("policy", 0) > 10.0:
            self._add_gradient_clipping("policy")
        
        if td.get("_training_phase") == "finetuning":
            self._enable_normalization_mode()
        
        # Agent adds its own signals
        td["_agent_mode"] = self.current_mode
        td["_component_states"] = self.get_component_states()
        
        # Network can read both trainer and agent signals
        return self.network(td)

# Three-way communication through TensorDict
# trainer_signals → agent_adaptation → network_execution → results
```

## **Advanced Wrapper Examples: Solving Real Training Challenges**

### 1. **Adaptive Gradient Clipping Wrapper**

Traditional gradient clipping uses fixed thresholds that may be too conservative (slowing training) or too permissive (allowing instability). An adaptive wrapper would monitor gradient norms from the trainer and automatically adjust clipping values based on recent history. This provides stable training without manual hyperparameter tuning. The wrapper maintains its own gradient statistics and communicates the current clipping value back to the trainer for monitoring. This is especially valuable for long training runs where optimal clipping values may change as the model converges.

```python
class AdaptiveGradientWrapper(MettaModule):
    """Automatically adjusts gradient clipping based on training signals."""
    
    def __init__(self, wrapped_component: MettaModule, 
                 initial_clip_value: float = 1.0,
                 adaptation_rate: float = 0.1):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.current_clip_value = initial_clip_value
        self.adaptation_rate = adaptation_rate
        self.gradient_history = []
    
    def forward(self, td: TensorDict) -> TensorDict:
        # Read gradient norm signal from trainer
        gradient_norm = td.get("_gradient_norms", {}).get(self.component_name, 0.0)
        
        # Adapt clipping value based on gradient history
        if gradient_norm > 0:
            self.gradient_history.append(gradient_norm)
            if len(self.gradient_history) > 100:
                self.gradient_history.pop(0)
            
            # Increase clipping if gradients are consistently high
            avg_gradient = sum(self.gradient_history) / len(self.gradient_history)
            if avg_gradient > self.current_clip_value * 2:
                self.current_clip_value *= (1 + self.adaptation_rate)
            elif avg_gradient < self.current_clip_value * 0.5:
                self.current_clip_value *= (1 - self.adaptation_rate)
        
        # Forward through wrapped component
        td = self.wrapped_component(td)
        
        # Apply adaptive clipping to outputs
        for out_key in self.out_keys:
            if out_key in td:
                td[out_key] = torch.clamp(td[out_key], 
                                        -self.current_clip_value, 
                                        self.current_clip_value)
        
        # Report back to trainer
        td[f"_adaptive_clip_{self.component_name}"] = self.current_clip_value
        
        return td

# Usage: Wrapper automatically adapts to training dynamics
policy = LinearModule(128, 4, "features", "action_logits")
adaptive_policy = AdaptiveGradientWrapper(policy, component_name="policy")
```

### 2. **Training Phase-Aware Wrapper**

Different training phases (pretraining, finetuning, evaluation) often benefit from different component behaviors - more regularization during pretraining, output smoothing during finetuning, clean execution during evaluation. A training phase wrapper would automatically adjust component behavior based on signals from the trainer. This eliminates the need to manually modify components or swap them out during different training phases. The wrapper can apply different noise levels, dropout rates, or smoothing based on the current phase. This is particularly useful for curriculum learning and multi-stage training protocols.

```python
class TrainingPhaseWrapper(MettaModule):
    """Changes component behavior based on training phase."""
    
    def __init__(self, wrapped_component: MettaModule, 
                 phase_behaviors: Dict[str, Dict[str, Any]]):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.phase_behaviors = phase_behaviors
        self.current_behavior = {}
    
    def forward(self, td: TensorDict) -> TensorDict:
        # Read training phase from trainer
        training_phase = td.get("_training_phase", "training")
        
        # Update behavior based on phase
        if training_phase in self.phase_behaviors:
            self.current_behavior = self.phase_behaviors[training_phase]
        
        # Apply phase-specific modifications to input
        if "input_noise" in self.current_behavior:
            noise_level = self.current_behavior["input_noise"]
            for in_key in self.in_keys:
                if in_key in td:
                    noise = torch.randn_like(td[in_key]) * noise_level
                    td[in_key] = td[in_key] + noise
        
        # Forward through wrapped component
        td = self.wrapped_component(td)
        
        # Apply phase-specific modifications to output
        if "output_dropout" in self.current_behavior:
            dropout_rate = self.current_behavior["output_dropout"]
            for out_key in self.out_keys:
                if out_key in td and self.training:
                    td[out_key] = F.dropout(td[out_key], p=dropout_rate)
        
        if "output_smoothing" in self.current_behavior:
            smoothing_factor = self.current_behavior["output_smoothing"]
            for out_key in self.out_keys:
                if out_key in td:
                    # Apply exponential smoothing
                    if hasattr(self, f"_smoothed_{out_key}"):
                        smoothed = getattr(self, f"_smoothed_{out_key}")
                        new_smoothed = (smoothing_factor * smoothed + 
                                      (1 - smoothing_factor) * td[out_key])
                        setattr(self, f"_smoothed_{out_key}", new_smoothed)
                        td[out_key] = new_smoothed
                    else:
                        setattr(self, f"_smoothed_{out_key}", td[out_key].clone())
        
        return td

# Usage: Different behaviors for different training phases
phase_behaviors = {
    "simple_env": {"exploration_noise": 0.3, "epsilon": 0.8},  # High exploration in simple env
    "medium_env": {"exploration_noise": 0.1, "epsilon": 0.4},  # Moderate exploration  
    "complex_env": {"exploration_noise": 0.05, "epsilon": 0.1}, # Low exploration, exploit knowledge
    "evaluation": {}  # Clean execution, no exploration
}

policy_network = LinearModule(128, 4, "state_features", "action_logits")
curriculum_policy = TrainingPhaseWrapper(policy_network, phase_behaviors)
```


## **Configuration-Driven Wrapper Composition**

Hard-coding wrapper stacks in Python limits experimentation and makes it difficult to A/B test different training strategies. Configuration-driven wrapper composition would let you specify complex wrapper stacks in YAML, making sophisticated training behaviors configurable rather than programmatic. This enables rapid experimentation with different combinations of wrappers and their parameters. Researchers could easily test whether adding output smoothing improves results, or compare different performance monitoring thresholds. The approach maintains the composability of wrappers while adding the flexibility of declarative configuration.

```yaml
# YAML configuration for complex wrapper stacks
components:
  - name: policy
    class: LinearModule
    config: {out_features: 4, activation: softmax}
    wrappers:
      - type: performance_monitoring
        config:
          performance_targets: {action_logits: 0.85}
          alert_thresholds: {latency_ms: 50, memory_mb: 500}
      
      - type: training_phase_aware
        config:
          phase_behaviors:
            pretraining: {input_noise: 0.05, output_dropout: 0.1}
            finetuning: {output_smoothing: 0.95}
            evaluation: {}
      
      - type: adaptive_gradient
        config:
          initial_clip_value: 1.0
          adaptation_rate: 0.1
      
      - type: clipping
        config:
          action_logits: [-5.0, 5.0]
    
    in_keys: [features]
    out_keys: [action_logits]
```

## **Why This Pattern is Powerful**

1. **Composability**: Stack multiple behaviors without modifying base components
2. **Reusability**: Same wrappers work with any MettaModule 
3. **Separation of Concerns**: Training logic separate from computation logic
4. **Configuration-Driven**: Complex behaviors specified in YAML
5. **Signal-Based Communication**: Rich bidirectional communication via TensorDict
6. **Zero Base Component Changes**: Enhance behavior without touching original code

The wrapper pattern transforms complex training challenges into composable, reusable solutions that can be mixed and matched as needed. Each wrapper focuses on one aspect (monitoring, adaptation, phase-awareness) but they compose naturally to create sophisticated training pipelines.
The wrapper pattern transforms complex training challenges into composable, reusable solutions that can be mixed and matched as needed. Each wrapper focuses on one aspect (monitoring, adaptation, phase-awareness) but they compose naturally to create sophisticated training pipelines.

---

## Deep Dive: Advanced Wrapper Patterns and Learning Rate Control

### Component-Based Learning Rates: The Optimizer Challenge

One of the most requested features in sophisticated training systems is **component-level learning rate control**. Traditional approaches require either global learning rates (too coarse) or hard-coded per-component logic (brittle). The wrapper pattern provides an elegant solution that maintains the separation of concerns while enabling fine-grained control.

#### The Challenge: Learning Rates Live in Optimizers

Unlike other training behaviors (clipping, normalization, freezing), learning rates exist in the **optimizer**, not the forward pass. This creates a coordination challenge:

```python
# The problem: These live in different places
component.forward(td)           # Forward pass - where wrappers work
optimizer.step()                # Optimizer step - where learning rates live
```

#### The Solution: Two-Level Control System

The elegant solution uses the same intent → wrap → report pattern, but coordinates between wrappers (reporting desired rates) and trainer (updating optimizers):

```python
class LearningRateWrapper(MettaModule):
    """Reports desired learning rate without affecting computation."""
    
    def __init__(self, wrapped_component: MettaModule, learning_rate: float, component_name: str):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.learning_rate = learning_rate
        self.component_name = component_name
    
    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        # Forward pass is unchanged - no computation overhead
        return self.wrapped_component._compute(td)
    
    def forward(self, td: TensorDict) -> TensorDict:
        td = super().forward(td)
        
        # Report desired learning rate back to trainer
        td["_component_lr_requests"] = td.get("_component_lr_requests", {})
        td["_component_lr_requests"][self.component_name] = self.learning_rate
        
        return td

class ComponentAwareTrainer:
    """Trainer that coordinates wrapper requests with optimizer updates."""
    
    def __init__(self, agent):
        self.agent = agent
        # Create separate optimizers for each component
        self.component_optimizers = {}
        for component_name, component in agent.network.components.items():
            self.component_optimizers[component_name] = torch.optim.Adam(
                component.parameters(), lr=1e-3
            )
    
    def training_step(self, batch):
        # Trainer sets learning rate intent
        td = TensorDict({
            "observation": batch.observation,
            "_component_lr": {"encoder": 1e-4, "policy": 3e-4, "value": 1e-5}
        })
        
        # Agent wraps components and reports requests
        result = self.agent(td)
        
        # Trainer reads requests and updates optimizers
        lr_requests = result.get("_component_lr_requests", {})
        for component_name, requested_lr in lr_requests.items():
            if component_name in self.component_optimizers:
                optimizer = self.component_optimizers[component_name]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = requested_lr
        
        # Standard training continues
        loss = self.compute_loss(result, batch)
        loss.backward()
        
        # Update each component with its own learning rate
        for component_name, optimizer in self.component_optimizers.items():
            optimizer.step()
            optimizer.zero_grad()
```

### Advanced Learning Rate Patterns

**Dynamic Learning Rate Adaptation**
```python
class AdaptiveLearningRateTrainer(ComponentAwareTrainer):
    def training_step(self, batch):
        # React to training dynamics
        gradient_norms = self.get_gradient_norms()
        
        # Adaptive learning rate based on gradient health
        adaptive_lrs = {}
        for component_name, grad_norm in gradient_norms.items():
            if grad_norm > 10.0:  # Exploding gradients
                adaptive_lrs[component_name] = 1e-5  # Emergency slow down
            elif grad_norm < 0.01:  # Vanishing gradients  
                adaptive_lrs[component_name] = 3e-3  # Speed up
            else:
                adaptive_lrs[component_name] = 1e-3  # Normal rate
        
        td = TensorDict({
            "observation": batch.observation,
            "_component_lr": adaptive_lrs,
            "_gradient_norms": gradient_norms
        })
        
        return super().training_step_with_td(td, batch)
```

**Curriculum Learning with Learning Rate Schedules**
```python
curriculum_phases = {
    "warmup": {"encoder": 1e-3, "policy": 1e-4, "value": 1e-4},
    "main_training": {"encoder": 1e-4, "policy": 3e-4, "value": 1e-4}, 
    "fine_tuning": {"encoder": 1e-5, "policy": 1e-4, "value": 1e-5}
}

def get_learning_rates_for_phase(phase: str, step: int):
    base_lrs = curriculum_phases[phase]
    # Apply decay within phase
    decay_factor = 0.99 ** (step % 1000)
    return {k: v * decay_factor for k, v in base_lrs.items()}
```

### Wrapper Unwrapping and Activation Control

Real-world training requires the ability to **debug**, **save clean models**, and **dynamically adjust** wrapper behavior. The architecture provides two complementary approaches:

#### Activation Flags: Runtime Control

For most use cases, **activation flags** provide the cleanest solution - wrappers remain but can be disabled:

```python
class ActivatableMettaModule(MettaModule):
    """Base class for wrappers with activation control."""
    
    def __init__(self, wrapped_component: MettaModule, component_name: str, active: bool = True):
        super().__init__(
            in_keys=wrapped_component.in_keys,
            out_keys=wrapped_component.out_keys,
            input_shapes=wrapped_component.input_shapes,
            output_shapes=wrapped_component.output_shapes
        )
        self.wrapped_component = wrapped_component
        self.component_name = component_name
        self.active = active
        self.wrapper_type = self.__class__.__name__
    
    def forward(self, td: TensorDict) -> TensorDict:
        if not self.active:
            # Passthrough mode - behave like unwrapped component
            return self.wrapped_component(td)
        else:
            # Active mode - apply wrapper behavior
            return self._active_forward(td)
    
    def _active_forward(self, td: TensorDict) -> TensorDict:
        """Override for wrapper-specific behavior when active."""
        return self.wrapped_component(td)

class ActivatableNoGradWrapper(ActivatableMettaModule):
    """Freezing wrapper with activation control."""
    
    def _active_forward(self, td: TensorDict) -> TensorDict:
        with torch.no_grad():
            td = self.wrapped_component(td)
        
        td["_component_status"] = td.get("_component_status", {})
        td["_component_status"][self.component_name] = "frozen"
        return td
```

**Runtime Wrapper Control**
```python
# Temporarily disable wrapper for debugging
debug_td = TensorDict({
    "observation": batch.observation,
    "_deactivate_wrappers": {"policy": ["ClippingWrapper"]},  # Debug without clipping
    "_activate_wrappers": {"encoder": ["PerformanceMonitoringWrapper"]}  # Enable monitoring
})

result = agent(debug_td)
```

### Full Unwrapping: Clean Model Extraction

For model saving, deployment, or memory optimization, complete unwrapping removes wrapper objects:

```python
class WrapperTracker:
    """Tracks wrapper state and enables unwrapping."""
    
    def __init__(self):
        self.wrapper_history: Dict[str, List[tuple]] = {}
        self.active_wrappers: Dict[str, Set[str]] = {}
    
    def track_wrap(self, component_name: str, wrapper_type: str, original_component: MettaModule):
        if component_name not in self.wrapper_history:
            self.wrapper_history[component_name] = []
            self.active_wrappers[component_name] = set()
        
        self.wrapper_history[component_name].append((wrapper_type, original_component))
        self.active_wrappers[component_name].add(wrapper_type)
    
    def get_original_component(self, component_name: str) -> Optional[MettaModule]:
        """Get the original unwrapped component."""
        if component_name not in self.wrapper_history or not self.wrapper_history[component_name]:
            return None
        return self.wrapper_history[component_name][0][1]

class SmartAgent:
    """Agent with sophisticated wrapper management."""
    
    def __init__(self, network: ModularNetwork):
        self.network = network
        self.wrapper_tracker = WrapperTracker()
    
    def get_clean_network(self) -> ModularNetwork:
        """Extract network with all wrappers removed."""
        clean_network = ModularNetwork()
        
        for component_name in self.network.components.keys():
            original = self.wrapper_tracker.get_original_component(component_name)
            if original is not None:
                clean_network.add_component(component_name, original)
            else:
                clean_network.add_component(component_name, self.network.get_component(component_name))
        
        return clean_network
    
    def unwrap_component(self, component_name: str) -> bool:
        """Remove all wrappers from a component."""
        original = self.wrapper_tracker.get_original_component(component_name)
        if original is not None:
            self.network.swap_component(component_name, original)
            self.wrapper_tracker.active_wrappers[component_name] = set()
            return True
        return False
```

**Production Deployment Workflow**
```python
# Development: Full wrapper stack for training
training_agent = SmartAgent(network)
training_agent.wrap_component("encoder", PerformanceMonitoringWrapper, {})
training_agent.wrap_component("policy", AdaptiveGradientWrapper, {})

# Training with full monitoring and adaptation
train_with_wrappers(training_agent)

# Production: Clean model extraction
production_network = training_agent.get_clean_network()
torch.save(production_network.state_dict(), "production_model.pt")

# Deployment: Optimized inference
optimized_network = optimize_for_inference(production_network)
deploy_to_production(optimized_network)
```

### The Unified Training Control System

The most powerful aspect is how all these patterns compose into a **unified training control system**:

```python
# Complex training strategy as simple configuration
sophisticated_training_td = TensorDict({
    "observation": batch.observation,
    
    # Learning rate control
    "_component_lr": {"encoder": 1e-4, "policy": 3e-4, "value": 1e-5},
    
    # Gradient management
    "_frozen_components": ["encoder"],
    "_gradient_accumulation": {"policy": 4},
    "_gradient_clip": {"policy": 1.0},
    
    # Performance optimization
    "_eval_mode_components": ["auxiliary"],
    "_output_clipping": {"action_logits": (-1.0, 1.0)},
    
    # Debugging controls
    "_deactivate_wrappers": {"encoder": ["PerformanceMonitoringWrapper"]},
    "_activate_wrappers": {"policy": ["GradientMonitoringWrapper"]},
    
    # Training phase
    "_training_phase": "fine_tuning",
    "_step": 50000
})

# Execute sophisticated training strategy
result = agent(sophisticated_training_td)

# Rich feedback about actual training state
actual_lrs = result["_component_lr_actual"]
component_status = result["_component_status"]
wrapper_status = result["_wrapper_status"]
performance_metrics = result["_performance_metrics"]
gradient_stats = result["_gradient_statistics"]
```

### Key Benefits of This Unified Approach

**1. Configuration-Driven Complexity**
- Complex training strategies become YAML configurations
- A/B test different approaches without code changes
- Version control training strategies alongside model code

**2. Runtime Debugging and Adaptation**
- Temporarily disable problematic wrappers during debugging
- Dynamically adjust training based on performance metrics
- Emergency training interventions (gradient explosion, memory pressure)

**3. Clean Production Deployment**
- Extract optimized models without training artifacts
- Maintain training complexity without deployment overhead
- Easy transition from research to production

**4. Bidirectional Training Contracts**
- Trainer states intent through TensorDict signals
- Agent implements strategy through dynamic wrapping
- Components report actual behavior back to trainer
- Closed-loop system ensures training strategy matches reality

## Real-World Training Scenarios

**Scenario 1: Large Model Memory Management**
```python
# Detect memory pressure and adapt
if gpu_memory_usage > 0.9:
    td["_gradient_accumulation"] = {"transformer_layers": 8}
    td["_frozen_components"] = ["auxiliary_heads"]
    td["_component_lr"] = {k: v * 0.5 for k, v in current_lrs.items()}
```

**Scenario 2: Curriculum Learning Automation**
```python
# Automatic curriculum progression
if student_performance > 0.8:
    td["_training_phase"] = "advanced"
    td["_unfrozen_components"] = ["fine_grained_encoder"]
    td["_component_lr"] = advanced_learning_rates
```

**Scenario 3: Research Experiment Management**
```python
# A/B test training strategies
training_strategies = {
    "baseline": {"_component_lr": {"all": 1e-3}},
    "adaptive": {"_gradient_accumulation": {"encoder": 4}, "_component_lr": {"policy": 3e-4}},
    "curriculum": {"_frozen_components": ["encoder"], "_training_phase": "guided"}
}

for strategy_name, config in training_strategies.items():
    experiment_td = base_td.clone()
    experiment_td.update(config)
    performance = run_experiment(agent, experiment_td)
    results[strategy_name] = performance
```

This unified system transforms sophisticated training control from a **complex engineering challenge** into **elegant configuration management**. The wrapper pattern provides the mechanism, TensorDict provides the communication channel, and the activation system provides the flexibility - together creating a training framework that scales from simple research prototypes to production systems.

# Training Control: Traditional vs Metta Architecture

## The Problem: Traditional Training Control

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRADITIONAL APPROACH                               │
│                         (Tight Coupling Hell)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    Hard-coded     ┌─────────────────┐
│     TRAINER     │    per-component  │   COMPONENT A   │
│                 │    logic in       │                 │
│ - Global LR     │ ─────────────────► │ - Has LR logic  │
│ - Global freeze │                   │ - Has freeze    │
│ - Global clip   │                   │ - Has clip      │
└─────────────────┘                   │ - Has clip      │
         │                            │   logic         │
         │                            └─────────────────┘
         │                                     
         │         ┌─────────────────┐        
         │         │   COMPONENT B   │        
         └────────►│                 │        
                   │ - Different LR  │        
                   │   logic         │        
                   │ - Different     │        
                   │   freeze logic  │        
                   └─────────────────┘        

Problems:
❌ Logic scattered across components
❌ Hard to change training strategy  
❌ Component explosion (FrozenLinear, FastLinear, etc.)
❌ No runtime adaptation
❌ Testing requires mocking everything
❌ Production deployment still has training code
```

## The Solution: Metta's Elegant Training Control Triangle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METTA APPROACH                                    │
│                    (Clean Separation of Concerns)                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    1. TRAINER SETS INTENT
                    ╔═══════════════════╗
                    ║     TRAINER       ║
                    ║                   ║
                    ║ "I want:          ║
                    ║ • policy lr=3e-4  ║
                    ║ • freeze encoder  ║
                    ║ • clip outputs"   ║
                    ╚═══════════════════╝
                            │
                            │ TensorDict signals
                            ▼
            ┌─────────────────────────────────────┐
            │        "_component_lr": {           │
            │          "policy": 3e-4,           │
            │          "encoder": 1e-4           │
            │        },                          │
            │        "_frozen_components":       │
            │          ["encoder"],              │
            │        "_gradient_clip": {         │
            │          "policy": 1.0             │
            │        }                           │
            └─────────────────────────────────────┘
                            │
                            ▼
                    ╔═══════════════════╗
                    ║      AGENT        ║
                    ║                   ║
                    ║ "I'll wrap        ║
                    ║ components to     ║
                    ║ make it happen"   ║
                    ╚═══════════════════╝
                            │
            2. AGENT WRAPS DYNAMICALLY
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ NoGradWrapper(  │ │LearningRateWrap │ │ ClippingWrapper │
│   encoder       │ │ per(policy)     │ │   (policy)      │
│ )               │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
            3. COMPONENTS REPORT BACK
                            ▼
            ┌─────────────────────────────────────┐
            │     "_component_lr_actual": {      │
            │       "policy": 3e-4,             │
            │       "encoder": 1e-4             │
            │     },                            │
            │     "_component_status": {        │
            │       "encoder": "frozen"         │
            │     },                            │
            │     "_gradient_clip_actual": {    │
            │       "policy": 1.0               │
            │     }                             │
            └─────────────────────────────────────┘
                            │
                            ▼
                    ╔═══════════════════╗
                    ║     TRAINER       ║
                    ║                   ║
                    ║ "Perfect! Here's  ║
                    ║ what actually     ║
                    ║ happened..."      ║
                    ╚═══════════════════╝

BENEFITS:
✅ Zero component changes required
✅ Training strategy = configuration
✅ Runtime debugging & adaptation  
✅ Clean production deployment
✅ A/B test training approaches
✅ Components focus on computation only
```

## The Elegance: Signal-Based Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL FLOW ELEGANCE                                │
└──────────────────────────────────────────────────────────────────────────────┘

Intent Signals                 Wrapper Responses              Component Reports
     │                              │                              │
     ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐              ┌─────────────┐
│"_component_ │   ────────────►│ LRWrapper   │ ────────────►│"_component_ │
│lr": {       │                │ reads &     │              │lr_actual":  │
│ "policy":   │                │ implements  │              │ {...}       │
│  3e-4       │                │             │              │             │
│}            │                └─────────────┘              └─────────────┘
└─────────────┘                                                     │
                                                                    │
┌─────────────┐                ┌─────────────┐              ┌─────────────┐
│"_frozen_    │   ────────────►│ NoGradWrap  │ ────────────►│"_component_ │
│components": │                │ reads &     │              │status": {   │
│ ["encoder"] │                │ implements  │              │ "encoder":  │
│             │                │             │              │ "frozen"    │
└─────────────┘                └─────────────┘              └─────────────┘
                                                                    │
┌─────────────┐                ┌─────────────┐              ┌─────────────┐
│"_gradient_  │   ────────────►│ ClipWrapper │ ────────────►│"_gradient_  │
│clip": {     │                │ reads &     │              │clip_actual"│
│ "policy":   │                │ implements  │              │ {...}       │
│  1.0        │                │             │              │             │
└─────────────┘                └─────────────┘              └─────────────┘

KEY INSIGHT: Each wrapper is a MICRO-SERVICE for one training concern!
```

## Real-World Power: Configuration-Driven Training

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FROM RESEARCH TO PRODUCTION                               │
└──────────────────────────────────────────────────────────────────────────────┘

RESEARCH PHASE:
config/multi_agent_training.yaml
─────────────────────────────
training_phases:
  coordination_learning:
    _component_lr: {shared_encoder: 1e-3, explorer_policy: 1e-4, collector_policy: 1e-4, guardian_policy: 1e-4}
    _frozen_components: []
    _agent_populations: {explorers: 8, collectors: 4, guardians: 2}
    
  specialization_training:  
    _component_lr: {shared_encoder: 1e-4, explorer_policy: 3e-4, collector_policy: 2e-4, guardian_policy: 1e-4}
    _parameter_sharing: {shared_encoder: "all_agents"}
    _gradient_accumulation: {shared_encoder: 4}
    
  fine_tuning:
    _component_lr: {shared_encoder: 1e-5, explorer_policy: 1e-4, collector_policy: 1e-4, guardian_policy: 1e-4} 
    _frozen_components: [shared_encoder]
    _sync_frequency: {explorer_policy: 50, collector_policy: 100, guardian_policy: 200}

DEBUGGING PHASE:
debug_config.yaml
─────────────────
# Same base config, but add:
_deactivate_wrappers:
  policy: [ClippingWrapper]     # Debug without clipping
_activate_wrappers:  
  encoder: [PerformanceMonitor] # Monitor encoder performance

PRODUCTION PHASE:
# Extract clean network - no wrapper overhead
clean_network = agent.get_clean_network()
torch.save(clean_network.state_dict(), "production.pt")

THE POWER: Same components, same code - different behaviors through configuration!
```

## Why This Architecture Wins

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPETITIVE ADVANTAGES                              │
└─────────────────────────────────────────────────────────────────────────────┘

🚀 RESEARCH VELOCITY
   • A/B test training strategies without code changes
   • Rapid iteration on curriculum learning approaches  
   • Emergency training interventions via config

🔧 MAINTENANCE SIMPLICITY  
   • Components never change - only wrappers stack
   • Training logic centralized in wrapper library
   • Clear separation: computation vs training strategy

🏭 PRODUCTION READINESS
   • Clean model extraction for deployment
   • No training artifacts in production code
   • Performance optimization through unwrapping

🧪 DEBUGGING POWER
   • Runtime wrapper activation/deactivation
   • Component-level performance monitoring
   • Isolation of training behaviors for debugging

🎯 TEAM SCALABILITY
   • Junior researchers: use existing wrapper library
   • Senior researchers: create new wrappers once, use everywhere
   • MLOps teams: config-driven deployment pipelines
```

**Bottom Line: Metta transforms complex training control from a hard-coded engineering challenge into elegant configuration management.**

---

## Key Selling Points: Why Teams Should Adopt This Architecture

### 🎯 **For Researchers: "Complex Training Strategies Become Simple Configuration"**

```yaml
# Want to try curriculum learning? Just change the config:
training_phases:
  warmup:    {_component_lr: {encoder: 1e-3, policy: 1e-4}}
  advanced:  {_component_lr: {encoder: 1e-4, policy: 3e-4}, _frozen_components: []}
  mastery:   {_component_lr: {encoder: 1e-5, policy: 1e-4}, _frozen_components: [encoder]}

# A/B test different approaches without touching code:
strategies = ["standard", "curriculum", "adaptive_lr", "memory_efficient"]
for strategy in strategies:
    performance[strategy] = run_experiment(load_config(f"{strategy}.yaml"))
```

**Value Proposition**: Experiment with sophisticated training protocols in minutes, not days.

### ⚙️ **For Engineers: "Zero Component Modifications Required"**

```python
# Your components never change:
encoder = LinearModule(64, 128, "observation", "features")  # Same forever
policy = LinearModule(128, 4, "features", "action_logits")   # Same forever

# But training behavior evolves through wrappers:
training_td["_component_lr"] = {"encoder": 1e-4, "policy": 3e-4}        # Learning rates
training_td["_frozen_components"] = ["encoder"]                         # Freezing
training_td["_gradient_accumulation"] = {"policy": 4}                   # Memory management
training_td["_eval_mode_components"] = ["auxiliary"]                    # Mode switching

# Result: Industrial-strength training with research-simple components
```

**Value Proposition**: Build once, enhance forever - no refactoring required.

### 🏭 **For Production Teams: "Clean Deployment Without Training Overhead"**

```python
# Development: Full wrapper stack for sophisticated training
research_agent = SmartAgent(network)
research_agent.wrap_component("encoder", PerformanceMonitoringWrapper, {})
research_agent.wrap_component("policy", AdaptiveGradientWrapper, {})
research_agent.wrap_component("value", MemoryOptimizationWrapper, {})

# Training: All the complexity you need
train_with_full_monitoring(research_agent)

# Production: Clean extraction with zero training artifacts
production_network = research_agent.get_clean_network()
torch.save(production_network.state_dict(), "production_model.pt")  # Pure computation

# Deployment: Optimized inference with no training code
deploy_optimized_model(production_network)  # 10-50% faster, 30-70% less memory
```

**Value Proposition**: Research complexity during development, production simplicity at deployment.

### 🐛 **For DevOps/MLOps: "Runtime Debugging and Emergency Intervention"**

```python
# Emergency training intervention - no code deployment required
emergency_config = {
    "_gradient_clip": {"all_components": 0.5},          # Emergency gradient clipping
    "_component_lr": {k: v * 0.1 for k, v in current_lrs.items()},  # Slow down learning
    "_frozen_components": ["unstable_component"],        # Emergency freeze
    "_deactivate_wrappers": {"policy": ["ClippingWrapper"]}  # Debug without constraints
}

# Apply instantly via config update - no redeployment
training_system.update_config(emergency_config)
```

**Value Proposition**: Production training systems with runtime adaptability and emergency controls.