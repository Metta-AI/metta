# N-Channel Observation System Integration

## Overview

Your n-channel 2D numpy array observation system is **exceptionally well-designed** and represents a significant advancement over the current token-based observation system in MettaGrid. This guide shows how to integrate it with the existing repository infrastructure.

## Current vs. Proposed Observation Systems

### Current MettaGrid System
- **Token-based**: Observations are lists of `(location, type, value)` tuples
- **Sparse representation**: Only occupied positions are encoded
- **Feature-driven**: Different entity types have different features
- **Location**: `mettagrid/src/metta/mettagrid/observation_encoder.hpp`

```cpp
// Current system example
struct ObservationToken {
    PackedCoordinate location;  // 16-bit packed coordinate
    ObservationType type;      // Token type identifier
    ObservationType value;     // Token value
};
```

### Your Channel-Based System
- **Dense representation**: Full 2D grid with multiple channels
- **Semantic layers**: Each channel represents a specific observation type
- **Temporal behaviors**: INSTANT, DYNAMIC, PERSISTENT channel types
- **Extensible**: Easy to add new observation channels

## Integration Strategy

### Option 1: Extend MettaGrid C++ Environment
**Best for performance-critical applications**

1. **Create Channel Observation Encoder**:
```cpp
// observation_channel_encoder.hpp
class ChannelObservationEncoder {
public:
    ChannelObservationEncoder(int num_channels, int radius);

    // Convert token-based observations to channel format
    torch::Tensor tokens_to_channels(
        const std::vector<ObservationToken>& tokens,
        int agent_x, int agent_y
    );

    // Apply temporal decay to DYNAMIC channels
    void apply_decay(torch::Tensor& observation, const ChannelRegistry& registry);

    // Clear INSTANT channels
    void clear_instant(torch::Tensor& observation, const ChannelRegistry& registry);

private:
    int num_channels_;
    int radius_;
    std::map<std::string, int> channel_indices_;
};
```

2. **Integrate with MettaGridCore**:
```cpp
// Extend MettaGridCore with channel observations
class MettaGridCore {
public:
    // Existing token-based observations
    std::vector<ObservationToken> get_observations();

    // New channel-based observations
    torch::Tensor get_channel_observations(const ChannelRegistry& registry);

    // Dual-mode support
    enum ObservationMode { TOKENS, CHANNELS };
    void set_observation_mode(ObservationMode mode);
};
```

### Option 2: Python Post-Processing Layer
**Best for rapid prototyping and flexibility**

1. **Create Observation Processor**:
```python
# metta/sim/observation_processor.py
from typing import Dict, List, Tuple
import numpy as np
import torch

from channel_system import ChannelRegistry, ChannelHandler
from metta.mettagrid import MettaGridEnv

class ChannelObservationProcessor:
    """Converts MettaGrid token observations to channel format."""

    def __init__(self, registry: ChannelRegistry, radius: int = 5):
        self.registry = registry
        self.radius = radius
        self.observation_shape = (len(registry), radius * 2 + 1, radius * 2 + 1)

    def process_observation(
        self,
        token_obs: np.ndarray,
        agent_pos: Tuple[int, int],
        world_layers: Dict[str, np.ndarray],
        **kwargs
    ) -> torch.Tensor:
        """Convert token observation to channel format."""

        # Initialize channel observation
        channel_obs = torch.zeros(self.observation_shape, dtype=torch.float32)

        # Process each channel
        for handler in self.registry.get_all_handlers().values():
            channel_idx = self.registry.get_index(handler.name)

            # Extract relevant data from tokens and world layers
            channel_data = self._extract_channel_data(
                token_obs, world_layers, handler.name, agent_pos, **kwargs
            )

            # Process through handler
            handler.process(
                channel_obs,
                channel_idx,
                self._create_config(),
                agent_pos,
                **channel_data
            )

        return channel_obs

    def _extract_channel_data(
        self, token_obs: np.ndarray, world_layers: Dict[str, np.ndarray],
        channel_name: str, agent_pos: Tuple[int, int], **kwargs
    ) -> Dict:
        """Extract relevant data for a specific channel."""

        # Map channel names to data extraction logic
        extractors = {
            "SELF_HP": self._extract_self_hp,
            "ALLIES_HP": self._extract_allies_hp,
            "RESOURCES": self._extract_resources,
            "OBSTACLES": self._extract_obstacles,
            # ... add more extractors
        }

        extractor = extractors.get(channel_name)
        if extractor:
            return extractor(token_obs, world_layers, agent_pos, **kwargs)
        return {}
```

2. **Integrate with Simulation Runner**:
```python
# metta/sim/simulation.py
from .observation_processor import ChannelObservationProcessor

class Simulation:
    def __init__(self, config: SimulationConfig):
        # ... existing initialization ...

        # Add channel observation support
        if config.use_channel_observations:
            self.channel_processor = ChannelObservationProcessor(
                registry=config.channel_registry,
                radius=config.observation_radius
            )

    def _process_observations(self, raw_obs, agent_positions):
        """Process observations based on configuration."""

        if hasattr(self, 'channel_processor'):
            # Convert to channel format
            channel_obs = []
            for i, obs in enumerate(raw_obs):
                agent_pos = agent_positions[i]
                world_layers = self._extract_world_layers(obs)

                # Extract additional data for channels
                allies, enemies = self._extract_agent_data(obs, i)
                self_hp = self._extract_self_hp(obs, i)
                goal_pos = self._extract_goal_position(obs, i)

                channel_obs.append(self.channel_processor.process_observation(
                    obs, agent_pos, world_layers,
                    self_hp01=self_hp,
                    allies=allies,
                    enemies=enemies,
                    goal_world_pos=goal_pos
                ))

            return channel_obs

        # Fall back to original token processing
        return raw_obs
```

## Repository Integration Points

### 1. MettaGrid Environment Extension
```python
# metta/mettagrid/src/metta/mettagrid/gym_env.py
class MettaGridGymEnv(MettaGridCore, GymEnv):
    def __init__(self, env_config, channel_registry=None, **kwargs):
        super().__init__(env_config, **kwargs)

        if channel_registry:
            from channel_system import ChannelObservationProcessor
            self.channel_processor = ChannelObservationProcessor(
                channel_registry,
                radius=env_config.observation_radius
            )
            self.use_channels = True

    def step(self, action):
        # ... existing step logic ...

        if self.use_channels:
            # Convert observations to channel format
            channel_obs = []
            for i, obs in enumerate(observations):
                agent_pos = self.get_agent_position(i)
                world_layers = self.extract_world_layers(obs)

                # Additional data extraction
                channel_data = self.extract_channel_data(obs, i, agent_pos)

                channel_obs.append(
                    self.channel_processor.process_observation(
                        obs, agent_pos, world_layers, **channel_data
                    )
                )

            observations = channel_obs

        return observations, rewards, terminals, truncations, infos
```

### 2. Agent Policy Integration
```python
# metta/agent/src/metta/agent/metta_agent.py
class MettaAgent:
    def __init__(self, observation_config, channel_registry=None):
        # ... existing initialization ...

        if channel_registry:
            # Use channel observation space
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(channel_registry), 11, 11),  # R=5
                dtype=np.float32
            )
            self.use_channels = True

    def act(self, observations):
        if self.use_channels:
            # Observations are already in channel format
            # Process through channel-aware policy
            return self._channel_policy_act(observations)

        # Fall back to original token processing
        return self._token_policy_act(observations)
```

### 3. Training Integration
```python
# metta/agent/src/metta/agent/pytorch/base.py
class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=2):
        super().__init__()

        # Support both token and channel observations
        if hasattr(env, 'use_channels') and env.use_channels:
            # Channel observation processing
            self.obs_channels = env.observation_space.shape[0]
            self.obs_size = env.observation_space.shape[1] * env.observation_space.shape[2]

            # Flatten spatial dimensions for LSTM input
            self.channel_encoder = nn.Sequential(
                nn.Flatten(start_dim=1),  # Keep batch dim, flatten spatial
                nn.Linear(self.obs_size, input_size),
                nn.ReLU()
            )
        else:
            # Original token processing
            self.obs_shape = env.single_observation_space.shape
            # ... existing token processing logic
```

## Data Extraction Utilities

### Token to Channel Data Extraction
```python
# metta/sim/observation_utils.py
class MettaGridChannelExtractor:
    """Extract channel-relevant data from MettaGrid token observations."""

    @staticmethod
    def extract_self_hp(token_obs: np.ndarray, agent_id: int) -> float:
        """Extract agent's health from token observation."""
        # Find agent token and extract health feature
        agent_tokens = ObservationHelper.find_features_by_type(
            token_obs, TokenTypes.AGENT_TYPE_ID
        )
        # Extract health value and normalize to [0,1]
        return MettaGridChannelExtractor._normalize_health(health_value)

    @staticmethod
    def extract_allies_hp(token_obs: np.ndarray, agent_id: int) -> List[Tuple[int, int, float]]:
        """Extract allies' positions and health."""
        allies = []
        ally_tokens = ObservationHelper.find_features_by_type(
            token_obs, TokenTypes.ALLY_TYPE_ID
        )

        for token in ally_tokens:
            pos = ObservationHelper.get_positions_from_tokens([token])[0]
            health = MettaGridChannelExtractor._extract_health_from_token(token)
            allies.append((pos[0], pos[1], health))

        return allies

    @staticmethod
    def extract_world_layers(token_obs: np.ndarray, world_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Extract world layer data from tokens."""
        layers = {
            'RESOURCES': np.zeros(world_size, dtype=np.float32),
            'OBSTACLES': np.zeros(world_size, dtype=np.float32),
            'TERRAIN_COST': np.ones(world_size, dtype=np.float32),
        }

        # Process resource tokens
        resource_tokens = ObservationHelper.find_features_by_type(
            token_obs, TokenTypes.RESOURCE_TYPE_ID
        )
        for token in resource_tokens:
            pos = ObservationHelper.get_positions_from_tokens([token])[0]
            amount = token[2]  # Value field
            layers['RESOURCES'][pos[1], pos[0]] = amount

        # Process obstacle tokens
        obstacle_tokens = ObservationHelper.find_features_by_type(
            token_obs, TokenTypes.OBSTACLE_TYPE_ID
        )
        for token in obstacle_tokens:
            pos = ObservationHelper.get_positions_from_tokens([token])[0]
            layers['OBSTACLES'][pos[1], pos[0]] = 1.0

        return layers
```

## Configuration Integration

### Channel System Configuration
```python
# metta/sim/simulation_config.py
@dataclass
class SimulationConfig:
    # ... existing fields ...

    # Channel observation system
    use_channel_observations: bool = False
    channel_registry: Optional[ChannelRegistry] = None
    observation_radius: int = 5

    # Channel-specific parameters
    gamma_known: float = 0.95  # Decay for KNOWN_EMPTY
    gamma_dmg: float = 0.9     # Decay for DAMAGE_HEAT
    gamma_trail: float = 0.8   # Decay for TRAILS
    gamma_sig: float = 0.85    # Decay for ALLY_SIGNAL

    def create_channel_registry(self) -> ChannelRegistry:
        """Create and configure channel registry."""
        from channel_system import (
            ChannelRegistry, SelfHPHandler, AlliesHPHandler,
            WorldLayerHandler, VisibilityHandler, KnownEmptyHandler
        )

        registry = ChannelRegistry()

        # Register core channels
        registry.register(SelfHPHandler(), 0)
        registry.register(AlliesHPHandler(), 1)
        registry.register(WorldLayerHandler("RESOURCES", "RESOURCES"), 2)
        registry.register(VisibilityHandler(), 3)
        registry.register(KnownEmptyHandler(), 4)

        return registry
```

## Performance Considerations

### Memory Optimization
```python
# Use appropriate dtypes and memory layouts
channel_obs = torch.zeros(
    (num_channels, 2*radius+1, 2*radius+1),
    dtype=torch.float16,  # Reduce memory usage
    device=device
)

# Reuse observation tensors when possible
if hasattr(self, '_obs_buffer'):
    self._obs_buffer.zero_()
    obs = self._obs_buffer
else:
    obs = torch.zeros(observation_shape, dtype=torch_dtype, device=device)
    self._obs_buffer = obs
```

### Batch Processing
```python
# Process multiple agents in parallel
def process_batch_observations(
    self,
    batch_token_obs: List[np.ndarray],
    batch_agent_positions: List[Tuple[int, int]],
    batch_world_layers: List[Dict[str, np.ndarray]]
) -> torch.Tensor:
    """Process observations for multiple agents in batch."""

    batch_size = len(batch_token_obs)
    batch_obs = torch.zeros(
        (batch_size, self.num_channels, self.obs_size, self.obs_size),
        dtype=self.dtype,
        device=self.device
    )

    for i in range(batch_size):
        obs = self.process_observation(
            batch_token_obs[i],
            batch_agent_positions[i],
            batch_world_layers[i]
        )
        batch_obs[i] = obs

    return batch_obs
```

## Testing and Validation

### Integration Tests
```python
# tests/test_channel_observations.py
class TestChannelObservations:
    def test_token_to_channel_conversion(self):
        """Test conversion from token to channel format."""
        # Create mock token observation
        token_obs = self._create_mock_token_obs()

        # Convert to channels
        channel_obs = self.processor.process_observation(
            token_obs, (5, 5), self.mock_world_layers
        )

        # Validate channel structure
        assert channel_obs.shape == (13, 11, 11)  # 13 channels, R=5
        assert channel_obs.dtype == torch.float32

    def test_channel_decay(self):
        """Test temporal decay of DYNAMIC channels."""
        # Set up observation with DYNAMIC channel values
        obs = torch.zeros((13, 11, 11))
        obs[7, 5, 5] = 1.0  # KNOWN_EMPTY channel

        # Apply decay
        self.registry.apply_decay(obs, self.config)

        # Verify decay was applied
        assert obs[7, 5, 5] < 1.0
        assert obs[7, 5, 5] == 0.95  # gamma_known = 0.95

    def test_channel_clearing(self):
        """Test clearing of INSTANT channels."""
        # Set up observation with INSTANT channel values
        obs = torch.ones((13, 11, 11))
        obs[0, 5, 5] = 0.8  # SELF_HP channel

        # Clear instant channels
        self.registry.clear_instant(obs)

        # Verify INSTANT channels were cleared
        assert obs[0, 5, 5] == 0.0
        # Verify DYNAMIC channels were preserved
        assert obs[7, 5, 5] == 1.0
```

## Migration Strategy

### Phase 1: Parallel Implementation
1. **Implement channel system alongside existing token system**
2. **Add configuration flag to switch between modes**
3. **Maintain full backward compatibility**

### Phase 2: Gradual Migration
1. **Migrate high-priority agents to channel observations**
2. **Update training pipelines to support both formats**
3. **Add performance benchmarks comparing the two approaches**

### Phase 3: Full Adoption
1. **Deprecate token-based observations for new projects**
2. **Maintain token support for legacy compatibility**
3. **Update documentation and examples**

## Benefits of Integration

### **Enhanced Agent Capabilities**
- **Semantic Understanding**: Agents can reason about different observation types
- **Temporal Memory**: DYNAMIC channels provide memory of past events
- **Spatial Reasoning**: Dense 2D representation enables better spatial understanding

### **Research and Development**
- **Extensible Framework**: Easy to add new observation channels
- **Comparative Studies**: Compare performance with different observation formats
- **Advanced Features**: Support for complex observation processing

### **Performance and Scalability**
- **GPU Acceleration**: PyTorch tensor operations enable GPU processing
- **Batch Processing**: Efficient processing of multiple agents
- **Memory Efficiency**: Optimized tensor operations and memory layouts

Your channel-based observation system represents a **significant advancement** that could greatly enhance the MettaGrid environment's capabilities for agent learning and research!
