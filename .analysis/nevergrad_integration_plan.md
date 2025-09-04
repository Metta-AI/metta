# Nevergrad Integration Plan for Metta Sweep Infrastructure

## Executive Summary

This document outlines the integration plan for Facebook Research's **Nevergrad** gradient-free optimization library into the Metta sweep infrastructure. Nevergrad offers over 40 state-of-the-art optimization algorithms including evolutionary strategies, Bayesian optimization, and population-based methods that would significantly expand Metta's hyperparameter optimization capabilities.

---

## 1. Nevergrad Overview

### Key Features
- **40+ Optimization Algorithms**: Including CMA-ES, PSO, Differential Evolution, TBPSA, and meta-optimizers
- **Rich Parametrization API**: Supports continuous, discrete, categorical, and mixed parameter spaces
- **Built-in Parallelization**: Native support for parallel evaluation with `num_workers`
- **Multi-objective Optimization**: Pareto front computation for multiple objectives
- **Constraint Handling**: Both cheap constraints and constraint violations
- **Ask-Tell Interface**: Compatible with Metta's current architecture

### Popular Algorithms
- **NGOpt/NgIohTuned**: Meta-optimizer that adapts to problem characteristics
- **CMA-ES**: Excellent for continuous optimization with moderate dimensionality
- **TwoPointsDE**: Robust differential evolution variant, good for parallel evaluation
- **PSO**: Particle Swarm Optimization, robust and parallelizable
- **TBPSA**: Test-Based Population-Size Adaptation, excellent for noisy functions
- **DiscreteOnePlusOne**: Specialized for discrete/categorical parameters

---

## 2. Current Metta Architecture Analysis

### Existing Components
```
Optimizer Protocol → ProteinOptimizer → MettaProtein → Protein Algorithm
                                     ↓
                               Observations
```

### Key Interfaces
```python
class Optimizer(Protocol):
    def suggest(self, observations: list[Observation], n_suggestions: int = 1) -> list[dict[str, Any]]

class Observation:
    score: float
    cost: float
    suggestion: dict
```

---

## 3. Integration Approach

### Strategy: Adapter Pattern
Create a `NevergradOptimizer` adapter that implements the Metta `Optimizer` protocol while internally using Nevergrad's ask-tell interface.

### Architecture
```
Metta Sweep System
    ↓
Optimizer Protocol
    ↓
NevergradOptimizer (Adapter)
    ↓
Nevergrad ng.optimizers.*
```

---

## 4. Implementation Components

### 4.1 Parameter Space Translation

#### Metta ProteinConfig → Nevergrad Parametrization

```python
# Metta ParameterConfig
{
    "learning_rate": ParameterConfig(
        min=1e-5, max=1e-2,
        distribution="log_normal",
        mean=1e-3, scale="auto"
    )
}

# Translates to Nevergrad
ng.p.Log(lower=1e-5, upper=1e-2)
```

### 4.2 Core Components to Implement

#### A. NevergradConfig (`metta/sweep/nevergrad_config.py`)
```python
from typing import Literal, Any, Dict
from pydantic import Field
from metta.mettagrid.config import Config

class NevergradParameterConfig(Config):
    """Configuration for a single Nevergrad parameter."""
    
    param_type: Literal["scalar", "log", "choice", "array", "transition_choice"]
    lower: float | None = None
    upper: float | None = None
    init: float | None = None
    choices: list[Any] | None = None
    integer: bool = False
    shape: tuple[int, ...] | None = None

class NevergradOptimizerConfig(Config):
    """Configuration for Nevergrad optimizer."""
    
    algorithm: str = Field(
        default="NGOpt",
        description="Nevergrad optimizer name (e.g., NGOpt, CMA, TwoPointsDE, PSO)"
    )
    parameters: Dict[str, NevergradParameterConfig] = Field(
        default_factory=dict,
        description="Parameter definitions"
    )
    num_workers: int = Field(
        default=1,
        description="Number of parallel evaluations"
    )
    # Algorithm-specific settings
    algorithm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional algorithm-specific parameters"
    )
```

#### B. NevergradOptimizer (`metta/sweep/optimizer/nevergrad.py`)
```python
import logging
from typing import Any, Dict, List
import nevergrad as ng
from metta.sweep.models import Observation
from metta.sweep.nevergrad_config import NevergradOptimizerConfig

logger = logging.getLogger(__name__)

class NevergradOptimizer:
    """Adapter for Nevergrad optimizers."""
    
    def __init__(self, config: NevergradOptimizerConfig):
        self.config = config
        self.parametrization = self._build_parametrization()
        self.optimizer = self._create_optimizer()
        self.candidate_map = {}  # Map suggestion dict to ng.p.Parameter
        
    def _build_parametrization(self) -> ng.p.Parameter:
        """Convert config to Nevergrad parametrization."""
        params = {}
        
        for name, param_config in self.config.parameters.items():
            if param_config.param_type == "scalar":
                param = ng.p.Scalar(
                    lower=param_config.lower,
                    upper=param_config.upper,
                    init=param_config.init
                )
                if param_config.integer:
                    param = param.set_integer_casting()
                    
            elif param_config.param_type == "log":
                param = ng.p.Log(
                    lower=param_config.lower,
                    upper=param_config.upper,
                    init=param_config.init
                )
                
            elif param_config.param_type == "choice":
                param = ng.p.Choice(param_config.choices)
                
            elif param_config.param_type == "transition_choice":
                param = ng.p.TransitionChoice(
                    param_config.choices,
                    repetitions=param_config.shape[0] if param_config.shape else 1
                )
                
            elif param_config.param_type == "array":
                param = ng.p.Array(shape=param_config.shape)
                if param_config.lower is not None:
                    param = param.set_bounds(
                        lower=param_config.lower,
                        upper=param_config.upper
                    )
            else:
                raise ValueError(f"Unknown param_type: {param_config.param_type}")
                
            params[name] = param
            
        # Create instrumentation for multiple parameters
        if len(params) == 1 and "default" in params:
            return params["default"]
        else:
            return ng.p.Instrumentation(**params)
    
    def _create_optimizer(self) -> ng.optimizers.Optimizer:
        """Create Nevergrad optimizer instance."""
        optimizer_class = ng.optimizers.registry[self.config.algorithm]
        
        return optimizer_class(
            parametrization=self.parametrization,
            budget=1000000,  # Set high, controlled by sweep
            num_workers=self.config.num_workers,
            **self.config.algorithm_kwargs
        )
    
    def suggest(self, observations: list[Observation], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate hyperparameter suggestions using Nevergrad."""
        
        # Tell Nevergrad about all observations
        for obs in observations:
            # Check if we have a candidate for this suggestion
            suggestion_key = self._dict_to_key(obs.suggestion)
            
            if suggestion_key in self.candidate_map:
                candidate = self.candidate_map[suggestion_key]
            else:
                # Create candidate from observation (inoculation)
                candidate = self._create_candidate_from_dict(obs.suggestion)
                
            # Tell the optimizer about this evaluation
            self.optimizer.tell(candidate, obs.score)
            
        logger.info(f"Loaded {len(observations)} observations into Nevergrad")
        
        # Generate new suggestions
        suggestions = []
        for _ in range(n_suggestions):
            candidate = self.optimizer.ask()
            
            # Convert Nevergrad candidate to dict
            suggestion_dict = self._candidate_to_dict(candidate)
            
            # Store mapping for later tell
            suggestion_key = self._dict_to_key(suggestion_dict)
            self.candidate_map[suggestion_key] = candidate
            
            suggestions.append(suggestion_dict)
            logger.debug(f"Generated suggestion: {suggestion_dict}")
            
        return suggestions
    
    def _candidate_to_dict(self, candidate: ng.p.Parameter) -> Dict[str, Any]:
        """Convert Nevergrad candidate to dict format."""
        if isinstance(self.parametrization, ng.p.Instrumentation):
            # Multiple parameters
            result = {}
            for key, value in candidate.kwargs.items():
                result[key] = self._extract_value(value)
            return result
        else:
            # Single parameter
            return {"default": self._extract_value(candidate.value)}
    
    def _extract_value(self, value):
        """Extract Python value from parameter value."""
        if hasattr(value, 'tolist'):  # numpy array
            return value.tolist()
        return value
    
    def _create_candidate_from_dict(self, suggestion: Dict[str, Any]) -> ng.p.Parameter:
        """Create Nevergrad candidate from dict (for inoculation)."""
        if isinstance(self.parametrization, ng.p.Instrumentation):
            return self.parametrization.spawn_child(new_value=((), suggestion))
        else:
            return self.parametrization.spawn_child(new_value=suggestion.get("default"))
    
    def _dict_to_key(self, d: dict) -> str:
        """Convert dict to hashable key for mapping."""
        return str(sorted(d.items()))
```

#### C. Multi-objective Support (`metta/sweep/optimizer/nevergrad_multiobjective.py`)
```python
class NevergradMultiObjectiveOptimizer(NevergradOptimizer):
    """Nevergrad optimizer with multi-objective support."""
    
    def __init__(self, config: NevergradOptimizerConfig, objectives: List[str]):
        super().__init__(config)
        self.objectives = objectives
        
        # Set reference point for multi-objective optimization
        # This should be configured based on problem
        self.reference_point = ng.p.MultiobjectiveReference()
    
    def suggest(self, observations: list[Observation], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate suggestions for multi-objective optimization."""
        
        # Tell with multiple objectives
        for obs in observations:
            suggestion_key = self._dict_to_key(obs.suggestion)
            
            if suggestion_key in self.candidate_map:
                candidate = self.candidate_map[suggestion_key]
            else:
                candidate = self._create_candidate_from_dict(obs.suggestion)
            
            # Extract multiple objectives from observation
            losses = [getattr(obs, obj) for obj in self.objectives]
            self.optimizer.tell(candidate, losses)
        
        # Rest is same as single-objective
        return super().suggest([], n_suggestions)
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get current Pareto front."""
        pareto_params = self.optimizer.pareto_front()
        return [self._candidate_to_dict(p) for p in pareto_params]
```

### 4.3 Integration with Existing System

#### Update SweepTool to support Nevergrad (`metta/tools/sweep.py`)
```python
# Add to imports
from metta.sweep.optimizer.nevergrad import NevergradOptimizer
from metta.sweep.nevergrad_config import NevergradOptimizerConfig

# Add optimizer type enum
class OptimizerType(StrEnum):
    PROTEIN = "protein"
    NEVERGRAD = "nevergrad"

# Add to SweepTool class
optimizer_type: OptimizerType = OptimizerType.PROTEIN
nevergrad_config: Optional[NevergradOptimizerConfig] = None

# In invoke method, create optimizer based on type
if self.optimizer_type == OptimizerType.NEVERGRAD:
    if self.nevergrad_config is None:
        raise ValueError("nevergrad_config required when optimizer_type is NEVERGRAD")
    optimizer = NevergradOptimizer(self.nevergrad_config)
else:
    optimizer = ProteinOptimizer(self.protein_config)
```

---

## 5. Example Configurations

### 5.1 Basic CMA-ES for Continuous Parameters
```python
from metta.sweep.nevergrad_config import NevergradOptimizerConfig, NevergradParameterConfig

config = NevergradOptimizerConfig(
    algorithm="CMA",
    parameters={
        "learning_rate": NevergradParameterConfig(
            param_type="log",
            lower=1e-5,
            upper=1e-2,
            init=1e-3
        ),
        "batch_size": NevergradParameterConfig(
            param_type="scalar",
            lower=8,
            upper=128,
            integer=True,
            init=32
        )
    },
    num_workers=8
)
```

### 5.2 Discrete Optimization with Choices
```python
config = NevergradOptimizerConfig(
    algorithm="DiscreteOnePlusOne",
    parameters={
        "architecture": NevergradParameterConfig(
            param_type="choice",
            choices=["conv", "lstm", "transformer"]
        ),
        "activation": NevergradParameterConfig(
            param_type="choice",
            choices=["relu", "gelu", "swish"]
        ),
        "layers": NevergradParameterConfig(
            param_type="transition_choice",
            choices=list(range(1, 13)),
            shape=(1,)
        )
    }
)
```

### 5.3 Meta-Optimizer (NGOpt)
```python
config = NevergradOptimizerConfig(
    algorithm="NGOpt",  # Auto-adapts to problem
    parameters={
        "lr": NevergradParameterConfig(param_type="log", lower=1e-5, upper=1e-1),
        "momentum": NevergradParameterConfig(param_type="scalar", lower=0.0, upper=0.999),
        "weight_decay": NevergradParameterConfig(param_type="log", lower=1e-6, upper=1e-2),
    },
    num_workers=16
)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests
```python
# tests/sweep/optimizer/test_nevergrad.py

def test_nevergrad_parameter_translation():
    """Test conversion between Metta and Nevergrad parameters."""
    
def test_nevergrad_suggest_tell_cycle():
    """Test the suggest-observe cycle."""
    
def test_nevergrad_multiple_algorithms():
    """Test different Nevergrad algorithms."""
    
def test_nevergrad_constraint_handling():
    """Test constraint violation handling."""
```

### 6.2 Integration Tests
```python
# tests/sweep/integration/test_nevergrad_sweep.py

def test_nevergrad_with_sweep_controller():
    """Test full sweep with Nevergrad optimizer."""
    
def test_nevergrad_parallel_evaluation():
    """Test parallel job evaluation."""
    
def test_nevergrad_resume_from_observations():
    """Test resuming with existing observations."""
```

### 6.3 Benchmark Tests
```python
# tests/sweep/benchmarks/test_optimizer_comparison.py

def test_compare_protein_vs_nevergrad():
    """Compare optimization performance."""
    
def test_nevergrad_algorithm_comparison():
    """Compare different Nevergrad algorithms."""
```

---

## 7. Migration Path

### Phase 1: Core Implementation (Week 1)
- [ ] Implement `NevergradConfig` classes
- [ ] Implement `NevergradOptimizer` adapter
- [ ] Add unit tests for parameter translation
- [ ] Test with simple optimization problems

### Phase 2: Integration (Week 2)
- [ ] Update `SweepTool` to support optimizer selection
- [ ] Add configuration examples
- [ ] Integration testing with sweep controller
- [ ] Documentation updates

### Phase 3: Advanced Features (Week 3)
- [ ] Multi-objective optimization support
- [ ] Constraint handling
- [ ] Custom acquisition functions
- [ ] Performance benchmarking

### Phase 4: Production Readiness (Week 4)
- [ ] Performance optimization
- [ ] Extensive testing on real workloads
- [ ] Migration guide for existing sweeps
- [ ] Monitoring and debugging tools

---

## 8. Benefits and Trade-offs

### Benefits
1. **Algorithm Diversity**: Access to 40+ optimization algorithms
2. **Better Performance**: CMA-ES, PSO often outperform random/genetic algorithms
3. **Multi-objective**: Native Pareto front computation
4. **Constraint Handling**: Built-in constraint violation support
5. **Well-tested**: Mature library with extensive real-world usage
6. **Active Development**: Regularly updated with new algorithms

### Trade-offs
1. **Additional Dependency**: Adds ~100MB dependency (numpy, scipy, etc.)
2. **Learning Curve**: Different parametrization API from Protein
3. **Migration Effort**: Existing sweeps need configuration updates
4. **Debugging Complexity**: Another layer in the optimization stack

### Risk Mitigation
- Keep Protein as default optimizer (backward compatibility)
- Provide clear migration guides and examples
- Implement comprehensive logging for debugging
- Add performance benchmarks to validate improvements

---

## 9. Recommended Nevergrad Algorithms for Metta Use Cases

### For RL Hyperparameter Tuning
- **NGOpt**: Meta-optimizer, good default choice
- **TwoPointsDE**: Excellent for parallel evaluation (high `num_workers`)
- **TBPSA**: Good for noisy RL environments

### For Architecture Search
- **DiscreteOnePlusOne**: Discrete choices
- **PortfolioDiscreteOnePlusOne**: Mixed discrete/continuous

### For Large-scale Parallel Sweeps
- **ScrHammersleySearchPlusMiddlePoint**: One-shot optimization
- **PSO**: Scales well with workers

### For Fine-tuning Near Optimum
- **CMA**: Excellent for local refinement
- **OnePlusOne**: Simple, robust for small adjustments

---

## 10. Conclusion

Integrating Nevergrad into Metta's sweep infrastructure would provide significant benefits:
- Access to state-of-the-art optimization algorithms
- Better support for different problem types (continuous, discrete, mixed)
- Native multi-objective optimization
- Improved parallel scaling

The integration can be done incrementally with minimal disruption to existing workflows by:
1. Using the adapter pattern to maintain protocol compatibility
2. Keeping Protein as the default optimizer
3. Providing clear configuration examples
4. Implementing comprehensive testing

The estimated development time is 4 weeks for full production readiness, with basic functionality available after 1 week.

---

## Appendix: Installation Requirements

```bash
# Add to requirements/requirements.txt or pyproject.toml
nevergrad>=1.0.0

# Optional dependencies for specific algorithms
# scipy>=1.7.0  # For CMA-ES and other algorithms
# scikit-learn>=1.0.0  # For some meta-optimizers
```

## References

1. [Nevergrad GitHub Repository](https://github.com/facebookresearch/nevergrad)
2. [Nevergrad Documentation](https://facebookresearch.github.io/nevergrad/)
3. [Nevergrad Paper](https://arxiv.org/abs/1812.05934)
4. [Algorithm Selection Guide](https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer)