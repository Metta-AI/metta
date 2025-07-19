from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

from metta.common.util.config import copy_omegaconf_config
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.mettagrid.curriculum.sampling import SampledTaskCurriculum
from metta.mettagrid.curriculum.util import config_from_path

logger = logging.getLogger(__name__)

# Try to import nevergrad if available
try:
    import nevergrad as ng
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False
    logger.info("Nevergrad not available, using built-in genetic operators")


class GeneticBuckettedCurriculum(RandomCurriculum):
    """
    A curriculum that maintains a population of tasks and evolves them using genetic algorithms.
    
    Unlike BuckettedCurriculum which generates all possible tasks, this maintains a fixed-size
    population and evolves it based on task performance.
    """
    
    def __init__(
        self,
        env_cfg_template_path: str,
        buckets: Dict[str, Dict[str, Any]],
        env_overrides: Optional[DictConfig] = None,
        default_bins: int = 1,
        population_size: int = 100,
        replacement_rate: float = 0.1,
        mutation_rate: float = 0.3,
        use_nevergrad: bool = False,
        **kwargs
    ):
        """
        Args:
            env_cfg_template_path: Path to environment config template
            buckets: Parameter ranges/values for task generation
            env_overrides: Additional config overrides
            default_bins: Default number of bins for range parameters
            population_size: Size of the task population
            replacement_rate: Fraction of population to replace each generation (default 0.1)
            mutation_rate: Probability of mutation vs crossover for new tasks
            use_nevergrad: Whether to use nevergrad for optimization (if available)
            **kwargs: Additional arguments passed to RandomCurriculum
        """
        self.env_cfg_template_path = env_cfg_template_path
        self.buckets = buckets
        self.default_bins = default_bins
        self.population_size = population_size
        self.replacement_rate = replacement_rate
        self.mutation_rate = mutation_rate
        self.use_nevergrad = use_nevergrad and NEVERGRAD_AVAILABLE
        
        # Expand buckets to get parameter specs
        self.expanded_buckets = self._expand_buckets(buckets, default_bins)
        
        # Initialize nevergrad optimizer if requested
        self.optimizer = None
        if self.use_nevergrad:
            self._init_nevergrad_optimizer()
        
        # Load base config
        base_cfg = config_from_path(env_cfg_template_path, env_overrides)
        self.env_cfg_template = copy_omegaconf_config(base_cfg)
        
        # Initialize population
        self._id_to_curriculum = {}
        self._id_to_params = {}
        self._initialize_population()
        
        # Initialize parent class with current population
        tasks = {task_id: 1.0 for task_id in self._id_to_curriculum.keys()}
        super().__init__(tasks=tasks, env_overrides=env_overrides, **kwargs)
        
        # Sync _curricula with our population
        self._curricula = self._id_to_curriculum.copy()
    
    def _expand_buckets(self, buckets: Dict[str, Dict[str, Any]], default_bins: int) -> Dict[str, Dict[str, Any]]:
        """Expand bucket specifications to include metadata about parameter types and ranges."""
        expanded = {}
        for param, spec in buckets.items():
            if "values" in spec:
                expanded[param] = {
                    "type": "discrete",
                    "values": spec["values"]
                }
            elif "range" in spec:
                lo, hi = spec["range"]
                expanded[param] = {
                    "type": "continuous",
                    "range": (lo, hi),
                    "want_int": isinstance(lo, int) and isinstance(hi, int)
                }
            else:
                raise ValueError(f"Invalid bucket spec for {param}: {spec}")
        return expanded
    
    def _init_nevergrad_optimizer(self):
        """Initialize nevergrad optimizer for the parameter space."""
        if not NEVERGRAD_AVAILABLE:
            return
            
        # Create parameter space for nevergrad
        params = {}
        for param_name, spec in self.expanded_buckets.items():
            if spec["type"] == "discrete":
                params[param_name] = ng.p.Choice(spec["values"])
            else:  # continuous
                lo, hi = spec["range"]
                if spec.get("want_int", False):
                    params[param_name] = ng.p.IntegerScalar(lower=lo, upper=hi)
                else:
                    params[param_name] = ng.p.Scalar(lower=lo, upper=hi)
        
        self.parameter_space = ng.p.Dict(**params)
        self.optimizer = ng.optimizers.NGOpt(parametrization=self.parameter_space, budget=10000)
    
    def _initialize_population(self):
        """Initialize the task population with random parameter combinations."""
        for i in range(self.population_size):
            params = self._sample_random_params()
            task_id = self._create_task_id(params)
            self._create_task(task_id, params)
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameters within the bucket constraints."""
        params = {}
        for param_name, spec in self.expanded_buckets.items():
            if spec["type"] == "discrete":
                params[param_name] = random.choice(spec["values"])
            else:  # continuous
                lo, hi = spec["range"]
                if spec.get("want_int", False):
                    params[param_name] = random.randint(int(lo), int(hi))
                else:
                    params[param_name] = random.uniform(lo, hi)
        return params
    
    def _create_task_id(self, params: Dict[str, Any]) -> str:
        """Create a unique task ID from parameters."""
        parts = []
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
        return ";".join(parts)
    
    def _create_task(self, task_id: str, params: Dict[str, Any]):
        """Create a new task with given parameters."""
        self._id_to_curriculum[task_id] = SampledTaskCurriculum(
            task_id, self.env_cfg_template, params
        )
        self._id_to_params[task_id] = params
    
    def _curriculum_from_id(self, id: str) -> Curriculum:
        return self._id_to_curriculum[id]
    
    def complete_task(self, id: str, score: float):
        """Complete a task and potentially evolve the population."""
        # Update task weight based on score
        if id in self._task_weights:
            # Use exponential moving average to update weight
            alpha = 0.1  # Learning rate
            self._task_weights[id] = alpha * score + (1 - alpha) * self._task_weights[id]
        
        # Call parent's complete_task
        super().complete_task(id, score)
        
        # Evolve population after task completion
        self._evolve_population()
    
    def _evolve_population(self):
        """Evolve the population by replacing lowest-performing tasks."""
        # Get current task weights/scores
        task_scores = []
        for task_id in list(self._id_to_curriculum.keys()):
            # Use learning progress tracker weights if available
            if hasattr(self, '_task_weights'):
                weight = self._task_weights.get(task_id, 0.0)
            else:
                weight = 1.0
            task_scores.append((task_id, weight))
        
        # Sort by score
        task_scores.sort(key=lambda x: x[1])
        
        # Determine how many tasks to replace
        num_to_replace = max(1, int(self.population_size * self.replacement_rate))
        
        # Ensure we keep at least 2 tasks for crossover
        max_removable = max(0, len(task_scores) - 2)
        num_to_replace = min(num_to_replace, max_removable)
        
        if num_to_replace == 0:
            return  # Cannot evolve with too few tasks
        
        # Remove lowest-scoring tasks
        tasks_to_remove = [task_id for task_id, _ in task_scores[:num_to_replace]]
        
        # Keep track of remaining tasks for parent selection
        remaining_task_scores = [(task_id, score) for task_id, score in task_scores 
                                 if task_id not in tasks_to_remove]
        
        for task_id in tasks_to_remove:
            del self._id_to_curriculum[task_id]
            del self._id_to_params[task_id]
            if hasattr(self, '_task_weights') and task_id in self._task_weights:
                del self._task_weights[task_id]
            # Also remove from parent class's _curricula
            if hasattr(self, '_curricula') and task_id in self._curricula:
                del self._curricula[task_id]
        
        # Generate new tasks using genetic operators
        for _ in range(num_to_replace):
            if random.random() < self.mutation_rate:
                # Mutation - use remaining tasks only
                new_params = self._mutate(remaining_task_scores)
            else:
                # Crossover - use remaining tasks only
                new_params = self._crossover(remaining_task_scores)
            
            # Ensure we don't create duplicate tasks
            new_task_id = self._create_task_id(new_params)
            attempts = 0
            while new_task_id in self._id_to_curriculum and attempts < 10:
                # Add small perturbation to create unique task
                new_params = self._perturb_params(new_params)
                new_task_id = self._create_task_id(new_params)
                attempts += 1
            
            if new_task_id not in self._id_to_curriculum:
                self._create_task(new_task_id, new_params)
                # Add to parent class's _curricula
                if hasattr(self, '_curricula'):
                    self._curricula[new_task_id] = self._id_to_curriculum[new_task_id]
                if hasattr(self, '_task_weights'):
                    # Initialize new task with average weight
                    if self._task_weights:
                        avg_weight = sum(self._task_weights.values()) / len(self._task_weights)
                    else:
                        avg_weight = 1.0
                    self._task_weights[new_task_id] = avg_weight
    
    def _select_parent(self, task_scores: List[Tuple[str, float]]) -> str:
        """Select a parent task proportional to its weight."""
        if not task_scores:
            raise ValueError("No tasks available for parent selection")
            
        # Filter out tasks with zero or negative weights
        valid_tasks = [(task_id, max(0.001, score)) for task_id, score in task_scores if score > 0]
        
        if not valid_tasks:
            # Fallback to random selection if all weights are zero
            return random.choice([task_id for task_id, _ in task_scores])
        
        # Weighted random selection
        total_weight = sum(score for _, score in valid_tasks)
        r = random.uniform(0, total_weight)
        cumsum = 0
        for task_id, score in valid_tasks:
            cumsum += score
            if cumsum >= r:
                return task_id
        
        return valid_tasks[-1][0]  # Fallback
    
    def _mutate(self, task_scores: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Mutate a parent task to create a new task."""
        parent_id = self._select_parent(task_scores)
        parent_params = self._id_to_params[parent_id].copy()
        
        # Select parameter to mutate
        param_to_mutate = random.choice(list(parent_params.keys()))
        spec = self.expanded_buckets[param_to_mutate]
        
        if spec["type"] == "discrete":
            # For discrete values, select a different value
            current_value = parent_params[param_to_mutate]
            other_values = [v for v in spec["values"] if v != current_value]
            if other_values:
                parent_params[param_to_mutate] = random.choice(other_values)
        else:  # continuous
            # Apply mutation operator
            operator = random.choice(["incr", "decr", "double", "half"])
            current_value = parent_params[param_to_mutate]
            lo, hi = spec["range"]
            
            if operator == "incr":
                delta = (hi - lo) * 0.1  # 10% increment
                new_value = current_value + delta
            elif operator == "decr":
                delta = (hi - lo) * 0.1  # 10% decrement
                new_value = current_value - delta
            elif operator == "double":
                new_value = current_value * 2
            else:  # half
                new_value = current_value / 2
            
            # Clamp to range
            new_value = max(lo, min(hi, new_value))
            
            if spec.get("want_int", False):
                new_value = int(round(new_value))
            
            parent_params[param_to_mutate] = new_value
        
        return parent_params
    
    def _crossover(self, task_scores: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Crossover two parent tasks to create a new task."""
        parent1_id = self._select_parent(task_scores)
        parent2_id = self._select_parent(task_scores)
        
        parent1_params = self._id_to_params[parent1_id]
        parent2_params = self._id_to_params[parent2_id]
        
        # Create child by randomly selecting from each parent
        child_params = {}
        for param_name in parent1_params.keys():
            if random.random() < 0.5:
                child_params[param_name] = parent1_params[param_name]
            else:
                child_params[param_name] = parent2_params[param_name]
        
        return child_params
    
    def _perturb_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add small perturbation to parameters to ensure uniqueness."""
        perturbed = params.copy()
        
        # Select a random continuous parameter to perturb
        continuous_params = [
            param for param, spec in self.expanded_buckets.items()
            if spec["type"] == "continuous"
        ]
        
        if continuous_params:
            param_to_perturb = random.choice(continuous_params)
            spec = self.expanded_buckets[param_to_perturb]
            lo, hi = spec["range"]
            
            # Add small noise
            noise = (hi - lo) * 0.01 * (random.random() - 0.5)
            new_value = perturbed[param_to_perturb] + noise
            new_value = max(lo, min(hi, new_value))
            
            if spec.get("want_int", False):
                new_value = int(round(new_value))
            
            perturbed[param_to_perturb] = new_value
        
        return perturbed
    
    def stats(self) -> Dict[str, float]:
        """Return curriculum statistics for logging purposes."""
        stats = super().stats()
        stats["population_size"] = len(self._id_to_curriculum)
        stats["genetic_curriculum"] = 1.0
        return stats