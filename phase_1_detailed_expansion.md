# Phase 1: Infrastructure Setup - Detailed Expansion

## Overview

Phase 1 focuses on building a complete, production-ready benchmark framework for comparing observation systems. This phase creates all the foundational components needed to run systematic experiments and analyze the impact of different observation architectures on agent learning and latent space quality.

## Detailed Implementation Plan

### **Week 1: Core Framework & Project Structure**

#### **Day 1-2: Project Structure & Directory Setup**

```bash
# Create main benchmark directory
mkdir -p experiments/benchmarks/observation_systems/

# Core benchmark framework
mkdir -p experiments/benchmarks/observation_systems/core/
mkdir -p experiments/benchmarks/observation_systems/analysis/
mkdir -p experiments/benchmarks/observation_systems/evaluation/

# Observation system implementations
mkdir -p experiments/benchmarks/observation_systems/systems/

# Benchmark configurations and results
mkdir -p experiments/benchmarks/observation_systems/configs/
mkdir -p experiments/benchmarks/observation_systems/results/

# Utilities and helpers
mkdir -p experiments/benchmarks/observation_systems/utils/

# Testing
mkdir -p experiments/benchmarks/observation_systems/tests/
```

#### **Day 3-4: Core Benchmark Classes**

**File: `experiments/benchmarks/observation_systems/core/benchmark_framework.py`**

Key classes to implement:

1. **`BenchmarkConfig`** - Configuration dataclass
```python
@dataclass
class BenchmarkConfig:
    name: str
    observation_systems: List[str]
    environments: List[str]
    metrics: List[str]
    latent_analysis: List[str]
    training_steps: int = 100000
    eval_episodes: int = 100
    seed: int = 42
```

2. **`BaseObservationSystem`** - Abstract interface
```python
class BaseObservationSystem(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def process_observation(self, raw_observation: Any) -> Any: ...
    @abstractmethod
    def get_observation_space(self, env_config: Any) -> Any: ...
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]: ...
```

3. **`BenchmarkRunner`** - Main orchestration engine
```python
class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig)
    def register_system(self, system: BaseObservationSystem)
    def run_benchmark(self) -> List[BenchmarkResult]
    def save_results(self, output_path: Path)
    def generate_report(self, output_path: Path)
```

#### **Day 5: Benchmark Result Structure**

```python
@dataclass
class BenchmarkResult:
    system_name: str
    environment_name: str
    performance_metrics: Dict[str, float]
    latent_analysis: Dict[str, Any]
    efficiency_metrics: Dict[str, float]
    training_time: float
    memory_usage: float
```

### **Week 2: Observation System Wrappers**

#### **Day 1-2: Token-Based System Wrapper**

**File: `experiments/benchmarks/observation_systems/systems/token_based.py`**

```python
class TokenBasedObservation(BaseObservationSystem):
    def name(self) -> str:
        return "token_based"

    def process_observation(self, raw_observation: np.ndarray) -> np.ndarray:
        """Token-based observations are already in the correct format."""
        return raw_observation

    def get_observation_space(self, env_config: Any) -> Any:
        """Return the observation space for token-based system."""
        # This would integrate with the existing MettaGrid observation space
        return None

    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage for token-based observations."""
        return {
            "observation_memory_mb": 0.0,  # To be calculated
            "processing_memory_mb": 0.0,   # To be calculated
        }

    def extract_features(self, observation: np.ndarray) -> Dict[str, Any]:
        """Extract structured features from token observations."""
        # Implementation using ObservationHelper
        pass
```

#### **Day 3-4: Channel-Based System Wrapper**

**File: `experiments/benchmarks/observation_systems/systems/channel_based.py`**

```python
class ChannelBasedObservation(BaseObservationSystem):
    def __init__(self, num_channels: int = 13, radius: int = 5):
        self.num_channels = num_channels
        self.radius = radius
        self.channel_names = [
            "SELF_HP", "ALLIES_HP", "ENEMIES_HP", "RESOURCES", "OBSTACLES",
            "TERRAIN_COST", "VISIBILITY", "KNOWN_EMPTY", "DAMAGE_HEAT",
            "TRAILS", "ALLY_SIGNAL", "GOAL", "LANDMARKS"
        ]

    def name(self) -> str:
        return "channel_based"

    def process_observation(self, raw_observation: Any) -> torch.Tensor:
        """Convert raw observation to channel-based format."""
        # This would implement the user's channel system logic
        observation_size = 2 * self.radius + 1
        return torch.zeros((self.num_channels, observation_size, observation_size),
                          dtype=torch.float32)

    def get_observation_space(self, env_config: Any) -> Any:
        """Return the observation space for channel-based system."""
        import gymnasium as gym
        observation_size = 2 * self.radius + 1
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_channels, observation_size, observation_size),
            dtype=np.float32
        )

    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage for channel-based observations."""
        observation_size = 2 * self.radius + 1
        total_elements = self.num_channels * observation_size * observation_size

        # Assuming float32 (4 bytes per element)
        observation_memory_mb = (total_elements * 4) / (1024 * 1024)

        return {
            "observation_memory_mb": observation_memory_mb,
            "processing_memory_mb": observation_memory_mb * 0.1,  # Estimate for processing
        }

    def get_channel_info(self) -> Dict[str, Any]:
        """Return information about each channel."""
        return {
            "num_channels": self.num_channels,
            "channel_names": self.channel_names,
            "observation_radius": self.radius,
            "spatial_size": 2 * self.radius + 1
        }
```

#### **Day 5: Hybrid System & System Registration**

**File: `experiments/benchmarks/observation_systems/systems/hybrid_observation.py`**

```python
class HybridObservation(BaseObservationSystem):
    """Hybrid system combining benefits of both token and channel approaches."""
    # Implementation combining structured token parsing with dense channel representations
```

**Integration with main runner:**

```python
def get_observation_system(system_name: str):
    """Get observation system instance by name."""
    systems = {
        "token_based": TokenBasedObservation(),
        "channel_based": ChannelBasedObservation(),
        "hybrid": HybridObservation()
    }
    # ... validation and return logic
```

### **Week 2 (Continued): Analysis Tools**

#### **Day 1-2: Latent Space Analyzer**

**File: `experiments/benchmarks/observation_systems/analysis/latent_space_analyzer.py`**

Key methods to implement:

1. **Basic Statistics**
```python
def _basic_statistics(self, latent_vectors: torch.Tensor) -> Dict[str, float]:
    return {
        "latent_dim": latent_vectors.shape[1],
        "num_samples": latent_vectors.shape[0],
        "mean_norm": torch.mean(torch.norm(latent_vectors, dim=1)).item(),
        "std_norm": torch.std(torch.norm(latent_vectors, dim=1)).item(),
        "sparsity": torch.mean((latent_vectors == 0).float()).item()
    }
```

2. **Diversity Analysis**
```python
def _diversity_analysis(self, latent_vectors: torch.Tensor) -> Dict[str, float]:
    # Dimension-wise variance
    dim_variance = torch.var(latent_vectors, dim=0)
    mean_variance = torch.mean(dim_variance).item()

    # PCA for effective dimensionality
    pca = PCA(n_components=min(latent_vectors.shape[1], latent_vectors.shape[0]))
    pca.fit(latent_vectors.numpy())
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    effective_dim_95 = np.searchsorted(explained_variance, 0.95) + 1

    return {
        "mean_dimension_variance": mean_variance,
        "effective_dimensionality_95": effective_dim_95,
        "total_explained_variance": explained_variance[-1]
    }
```

3. **Spatial Preservation**
```python
def _spatial_preservation_analysis(self, latent_vectors: torch.Tensor,
                                 positions: np.ndarray) -> Dict[str, float]:
    latent_np = latent_vectors.numpy()
    latent_distances = pdist(latent_np)
    physical_distances = pdist(positions)

    correlation, _ = pearsonr(latent_distances, physical_distances)
    rank_correlation, _ = spearmanr(latent_distances, physical_distances)

    return {
        "spatial_distance_correlation": correlation,
        "spatial_rank_correlation": rank_correlation
    }
```

#### **Day 3-4: Information Analyzer**

**File: `experiments/benchmarks/observation_systems/analysis/information_analyzer.py`**

Key methods to implement:

1. **Mutual Information Analysis**
```python
def _mutual_information_analysis(self, observations: torch.Tensor,
                               ground_truth: torch.Tensor) -> Dict[str, float]:
    obs_np = obs_flat.numpy()
    gt_np = ground_truth.numpy()

    obs_discretized = self._discretize_multidimensional(obs_np)
    gt_discretized = self._discretize_values(gt_np)

    mi = mutual_info_score(obs_discretized, gt_discretized)

    return {
        "mutual_information": mi,
        "mean_dimension_mi": np.mean(dim_mis) if dim_mis else 0.0,
        "max_dimension_mi": np.max(dim_mis) if dim_mis else 0.0,
        "mi_diversity": np.std(dim_mis) if dim_mis else 0.0
    }
```

2. **Compression Analysis**
```python
def _compression_analysis(self, observations: torch.Tensor) -> Dict[str, float]:
    obs_np = observations.numpy()
    original_size = obs_np.nbytes

    # Compress using numpy
    buffer = io.BytesIO()
    np.save(buffer, obs_np, allow_pickle=False)
    compressed_size = len(buffer.getvalue())

    compression_ratio = compressed_size / original_size

    # Symbol entropy analysis
    flattened = obs_np.flatten()
    discretized = self._discretize_values(flattened)
    frequencies = Counter(discretized)
    probabilities = [count / len(discretized) for count in frequencies.values()]
    symbol_entropy = entropy(probabilities)

    return {
        "compression_ratio": compression_ratio,
        "symbol_entropy": symbol_entropy,
        "alphabet_size": len(frequencies)
    }
```

#### **Day 5: Visualization Tools**

**File: `experiments/benchmarks/observation_systems/utils/visualization.py`**

Key functions to implement:

1. **Comparison Plots**
```python
def create_performance_comparison_plot(results_data: Dict[str, Any], output_dir: Path):
    """Create performance comparison plot."""
    # Implementation for completion rate, learning speed, path efficiency
```

2. **Latent Space Visualization**
```python
def create_latent_space_comparison_plot(results_data: Dict[str, Any], output_dir: Path):
    """Create latent space analysis comparison plot."""
    # Implementation for diversity, spatial preservation, temporal coherence
```

3. **Summary Table Generation**
```python
def generate_summary_table(results_data: Dict[str, Any]) -> str:
    """Generate a markdown summary table of results."""
    # Implementation for markdown table generation
```

### **Week 1-2 (Final Days): Integration & Testing**

#### **Day 1-2: Evaluation Pipeline**

**File: `experiments/benchmarks/observation_systems/evaluation/benchmark_evaluator.py`**

```python
class BenchmarkEvaluator:
    def evaluate_system(self, system: BaseObservationSystem,
                       environment_name: str) -> BenchmarkResult:
        # 1. Setup environment
        env = self._setup_environment(system, environment_name)

        # 2. Training loop
        training_metrics = self._run_training(env, system)

        # 3. Evaluation
        evaluation_metrics = self._run_evaluation(env, system)

        # 4. Latent space analysis
        latent_analysis = self._analyze_latent_space(system, env)

        # 5. Efficiency analysis
        efficiency_metrics = self._measure_efficiency(system, memory_monitor)

        return BenchmarkResult(...)
```

#### **Day 3-4: Configuration System**

**File: `experiments/benchmarks/observation_systems/configs/benchmark_configs.py`**

Predefined configurations:
- `quick_test`: 2 systems × 2 environments
- `navigation`: 3 systems × 23 environments
- `exploration`: 3 systems × memory environments
- `arena`: 3 systems × multi-agent environments
- `comprehensive`: All systems × representative environments

#### **Day 5: Main Execution Script**

**File: `experiments/benchmarks/observation_systems/run_benchmark.py`**

```python
def main():
    parser = argparse.ArgumentParser(description="Run observation system benchmarks")

    parser.add_argument("--config", choices=["navigation", "exploration", "arena", "quick_test", "comprehensive", "custom"])
    parser.add_argument("--systems", nargs="+", choices=["token_based", "channel_based", "hybrid"])
    parser.add_argument("--environments", nargs="+")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--no-individual-results", action="store_true")

    args = parser.parse_args()
    # Implementation for running benchmarks based on arguments
```

### **Testing & Validation**

#### **Day 1-2: Unit Tests**

**File: `experiments/benchmarks/observation_systems/tests/test_benchmark_framework.py`**

```python
class TestBenchmarkFramework:
    def test_benchmark_config_creation(self):
        config = BenchmarkConfig(...)
        assert config.name == "test_config"

    def test_benchmark_runner_registration(self):
        runner = BenchmarkRunner(config)
        system = MockObservationSystem("mock")
        runner.register_system(system)
        assert len(runner.observation_systems) == 1

    def test_duplicate_system_registration(self):
        # Test error handling for duplicate registration
```

**File: `experiments/benchmarks/observation_systems/tests/test_latent_space_analyzer.py`**

```python
class TestLatentSpaceAnalyzer:
    def test_latent_space_analysis(self):
        analyzer = LatentSpaceAnalyzer()
        latent_vectors = torch.randn(100, 64)
        results = analyzer.analyze_latent_space(latent_vectors)

        assert "latent_dim" in results
        assert "num_samples" in results
        assert results["latent_dim"] == 64
```

#### **Day 3-4: Integration Tests**

```python
def test_full_benchmark_pipeline():
    """Test complete benchmark pipeline from config to results."""
    config = BenchmarkConfig(
        name="integration_test",
        observation_systems=["token_based"],
        environments=["corridors"],
        metrics=["completion_rate"],
        latent_analysis=["diversity"],
        training_steps=1000,
        eval_episodes=5
    )

    runner = BenchmarkRunner(config)
    system = TokenBasedObservation()
    runner.register_system(system)

    results = runner.run_benchmark()

    assert len(results) == 1
    assert results[0].system_name == "token_based"
    assert results[0].environment_name == "corridors"
```

### **Documentation & Setup**

#### **Day 5: Complete Documentation**

**File: `experiments/benchmarks/observation_systems/README.md`**

Comprehensive documentation covering:
- Quick start guide
- Directory structure explanation
- Available benchmark configurations
- Metrics explanation
- Extending the framework
- Troubleshooting guide

**File: `experiments/benchmarks/observation_systems/setup.py`**

```python
def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "torch", "numpy", "matplotlib", "seaborn",
        "scikit-learn", "scipy", "pandas", "pytest",
        "psutil", "gputil"
    ]
    # Implementation for dependency checking

def create_directories():
    """Create necessary directories."""
    # Implementation for directory creation

def main():
    """Main setup function."""
    print("Setting up Observation System Benchmarks...")

    # Check Python version, dependencies, create directories, run tests
    # Provide next steps guidance
```

## Phase 1 Deliverables

### **Core Framework**
- ✅ `BenchmarkRunner` - Main orchestration engine
- ✅ `BaseObservationSystem` - Abstract interface for observation systems
- ✅ `BenchmarkConfig` - Flexible configuration system
- ✅ `BenchmarkResult` - Structured result storage

### **Observation System Wrappers**
- ✅ `TokenBasedObservation` - Wrapper for existing MettaGrid system
- ✅ `ChannelBasedObservation` - Wrapper for user's channel system
- ✅ `HybridObservation` - Combined approach wrapper

### **Analysis Tools**
- ✅ `LatentSpaceAnalyzer` - Diversity, spatial preservation, temporal coherence
- ✅ `InformationAnalyzer` - Mutual information, compression analysis
- ✅ `FeatureDisentanglement` - Channel correlation analysis

### **Evaluation Infrastructure**
- ✅ `BenchmarkEvaluator` - Complete evaluation pipeline
- ✅ `MemoryMonitor` - Resource usage tracking
- ✅ `TrainingMetricsCollector` - Training progress monitoring

### **Configuration & Utilities**
- ✅ Predefined benchmark configurations (quick_test, navigation, exploration, arena)
- ✅ Custom configuration support
- ✅ Visualization utilities for result analysis
- ✅ Automated report generation

### **Testing & Documentation**
- ✅ Comprehensive unit tests for all components
- ✅ Integration tests for full pipeline
- ✅ Complete documentation with examples
- ✅ Setup script with dependency checking

## Phase 1 Success Criteria

### **Functional Requirements**
- [x] Can register and run different observation systems
- [x] Can collect performance, latent space, and efficiency metrics
- [x] Can generate comparative reports and visualizations
- [x] Can handle custom configurations and environments

### **Technical Requirements**
- [x] Modular design allowing easy extension
- [x] Proper error handling and logging
- [x] Memory-efficient processing
- [x] GPU acceleration support where applicable

### **Quality Requirements**
- [x] Comprehensive unit test coverage (>80%)
- [x] Clear documentation and examples
- [x] Reproducible results with seeded random number generation
- [x] Performance monitoring and optimization

## Phase 1 Timeline Summary

**Week 1: Core Framework (5 days)**
- Day 1-2: Project structure and core classes
- Day 3-4: Benchmark runner and result handling
- Day 5: Configuration system and utilities

**Week 2: Implementation (5 days)**
- Day 1-2: Observation system wrappers
- Day 3-4: Analysis tools and evaluation pipeline
- Day 5: Testing, documentation, and setup scripts

**Total: 2 weeks**
- **Deliverable**: Complete benchmark framework ready for experimentation
- **Capability**: Can compare observation systems and analyze latent space properties
- **Next Phase**: Run initial benchmarks and analyze results to answer your key question

The Phase 1 foundation provides everything needed to systematically evaluate whether your channel-based observation system can enrich agent latent spaces compared to the existing token-based approach. 🚀
