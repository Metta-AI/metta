#!/usr/bin/env python3
"""
Recipe: Progressive Debugging of Complex Training Pipelines
============================================================

Problem: Your training pipeline has multiple components and you need to 
systematically test each one to identify issues.

Solution: Use Axiom's progressive debugging approach to incrementally test
components from simplest to most complex, mocking expensive operations.

This recipe demonstrates:
- Starting with minimal viable components
- Progressively adding complexity
- Mocking expensive operations for fast iteration
- Testing individual components in isolation
- Building confidence through incremental validation
"""

import os
import sys
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from metta.sweep.axiom.core import Pipeline
from metta.sweep.axiom.experiment_spec import AxiomControls
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment
)


# =============================================================================
# PROGRESSIVE DEBUGGING FRAMEWORK
# =============================================================================

@dataclass
class DebugLevel:
    """Represents a debugging level with specific components enabled."""
    name: str
    description: str
    components: Dict[str, bool]  # Which components are real vs mocked
    validation_fn: Optional[Callable] = None


class ProgressiveDebugger:
    """
    Framework for progressive debugging of training pipelines.
    """
    
    def __init__(self, base_spec: TrainAndEvalSpec):
        """
        Initialize progressive debugger.
        
        Args:
            base_spec: Base experiment specification
        """
        self.base_spec = base_spec
        self.levels = self._define_debug_levels()
        self.results = {}
        
    def _define_debug_levels(self) -> List[DebugLevel]:
        """Define progressive debugging levels from simplest to full."""
        return [
            DebugLevel(
                name="level_0_mock_all",
                description="Everything mocked - tests pipeline structure",
                components={
                    "env": False,
                    "network": False,
                    "optimizer": False,
                    "rollouts": False,
                    "advantages": False,
                    "loss": False,
                    "updates": False,
                },
                validation_fn=self._validate_pipeline_structure
            ),
            
            DebugLevel(
                name="level_1_env_only",
                description="Real environment, everything else mocked",
                components={
                    "env": True,
                    "network": False,
                    "optimizer": False,
                    "rollouts": False,
                    "advantages": False,
                    "loss": False,
                    "updates": False,
                },
                validation_fn=self._validate_env_interaction
            ),
            
            DebugLevel(
                name="level_2_network",
                description="Real env + network forward pass",
                components={
                    "env": True,
                    "network": True,
                    "optimizer": False,
                    "rollouts": False,
                    "advantages": False,
                    "loss": False,
                    "updates": False,
                },
                validation_fn=self._validate_network_forward
            ),
            
            DebugLevel(
                name="level_3_rollouts",
                description="Real env + network + rollout collection",
                components={
                    "env": True,
                    "network": True,
                    "optimizer": False,
                    "rollouts": True,
                    "advantages": False,
                    "loss": False,
                    "updates": False,
                },
                validation_fn=self._validate_rollouts
            ),
            
            DebugLevel(
                name="level_4_advantages",
                description="Add advantage computation",
                components={
                    "env": True,
                    "network": True,
                    "optimizer": False,
                    "rollouts": True,
                    "advantages": True,
                    "loss": False,
                    "updates": False,
                },
                validation_fn=self._validate_advantages
            ),
            
            DebugLevel(
                name="level_5_loss",
                description="Add loss computation",
                components={
                    "env": True,
                    "network": True,
                    "optimizer": False,
                    "rollouts": True,
                    "advantages": True,
                    "loss": True,
                    "updates": False,
                },
                validation_fn=self._validate_loss
            ),
            
            DebugLevel(
                name="level_6_full",
                description="Full pipeline with all components",
                components={
                    "env": True,
                    "network": True,
                    "optimizer": True,
                    "rollouts": True,
                    "advantages": True,
                    "loss": True,
                    "updates": True,
                },
                validation_fn=self._validate_full_pipeline
            ),
        ]
    
    def create_debug_pipeline(self, level: DebugLevel) -> Pipeline:
        """
        Create a pipeline for a specific debug level.
        
        Args:
            level: Debug level to create pipeline for
            
        Returns:
            Configured pipeline with appropriate mocking
        """
        pipeline = Pipeline(name=f"debug_{level.name}")
        
        # Initialize
        pipeline.stage("init", lambda _: {
            "debug_level": level.name,
            "components": level.components,
            "start_time": time.time()
        })
        
        # Environment setup
        if level.components["env"]:
            pipeline.stage("setup_env", self._real_env_setup, expose=True)
        else:
            pipeline.stage("setup_env", self._mock_env_setup, expose=True)
        
        # Network initialization
        if level.components["network"]:
            pipeline.stage("init_network", self._real_network_init, expose=True)
        else:
            pipeline.stage("init_network", self._mock_network_init, expose=True)
        
        # Optimizer setup
        if level.components["optimizer"]:
            pipeline.stage("init_optimizer", self._real_optimizer_init, expose=True)
        else:
            pipeline.stage("init_optimizer", self._mock_optimizer_init, expose=True)
        
        # Training loop components
        pipeline.stage("train_loop", 
                      self._create_train_loop(level), expose=True)
        
        # Validation
        pipeline.stage("validate", lambda s: {
            **s, 
            "validation_result": level.validation_fn(s) if level.validation_fn else True
        })
        
        # Finalize
        pipeline.stage("finalize", lambda s: {
            **s,
            "duration": time.time() - s["start_time"],
            "success": s.get("validation_result", False)
        })
        
        return pipeline
    
    def _create_train_loop(self, level: DebugLevel) -> Callable:
        """Create training loop function for given debug level."""
        def train_loop(state: Dict[str, Any]) -> Dict[str, Any]:
            print(f"  Running training loop at {level.name}")
            
            # Rollout collection
            if level.components["rollouts"]:
                state = self._real_collect_rollouts(state)
            else:
                state = self._mock_collect_rollouts(state)
            
            # Advantage computation
            if level.components["advantages"]:
                state = self._real_compute_advantages(state)
            else:
                state = self._mock_compute_advantages(state)
            
            # Loss computation
            if level.components["loss"]:
                state = self._real_compute_loss(state)
            else:
                state = self._mock_compute_loss(state)
            
            # Parameter updates
            if level.components["updates"]:
                state = self._real_update_parameters(state)
            else:
                state = self._mock_update_parameters(state)
            
            return state
        
        return train_loop
    
    # Mock implementations (fast, predictable)
    
    def _mock_env_setup(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Setting up environment")
        state["env"] = "MockEnv"
        state["observation_space"] = (4,)
        state["action_space"] = 2
        return state
    
    def _mock_network_init(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Initializing network")
        state["network"] = "MockNetwork"
        state["num_parameters"] = 1000
        return state
    
    def _mock_optimizer_init(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Initializing optimizer")
        state["optimizer"] = "MockOptimizer"
        state["learning_rate"] = 3e-4
        return state
    
    def _mock_collect_rollouts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Collecting rollouts")
        state["rollouts"] = {
            "observations": [[0, 0, 0, 0]] * 100,
            "actions": [0] * 100,
            "rewards": [1.0] * 100,
            "dones": [False] * 99 + [True],
        }
        state["num_rollout_steps"] = 100
        return state
    
    def _mock_compute_advantages(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Computing advantages")
        state["advantages"] = [0.1] * 100
        state["returns"] = [10.0] * 100
        return state
    
    def _mock_compute_loss(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Computing loss")
        state["policy_loss"] = -0.01
        state["value_loss"] = 0.5
        state["entropy_loss"] = 0.01
        state["total_loss"] = 0.48
        return state
    
    def _mock_update_parameters(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [MOCK] Updating parameters")
        state["grad_norm"] = 0.5
        state["parameters_updated"] = True
        return state
    
    # Real implementations (actual computation)
    
    def _real_env_setup(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Setting up environment")
        # In practice, would create actual environment
        state["env"] = "RealEnv"
        state["observation_space"] = (84, 84, 3)
        state["action_space"] = 4
        return state
    
    def _real_network_init(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Initializing network")
        # In practice, would create actual neural network
        state["network"] = "RealNetwork"
        state["num_parameters"] = 1_234_567
        return state
    
    def _real_optimizer_init(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Initializing optimizer")
        # In practice, would create actual optimizer
        state["optimizer"] = "Adam"
        state["learning_rate"] = 3e-4
        state["betas"] = (0.9, 0.999)
        return state
    
    def _real_collect_rollouts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Collecting rollouts")
        # In practice, would interact with environment
        import numpy as np
        np.random.seed(42)
        
        state["rollouts"] = {
            "observations": np.random.randn(1000, 4).tolist(),
            "actions": np.random.randint(0, 2, 1000).tolist(),
            "rewards": (np.random.randn(1000) * 0.1 + 0.5).tolist(),
            "dones": [False] * 999 + [True],
        }
        state["num_rollout_steps"] = 1000
        return state
    
    def _real_compute_advantages(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Computing advantages")
        # In practice, would compute GAE
        import numpy as np
        
        rewards = state["rollouts"]["rewards"]
        gamma = 0.99
        
        # Simplified advantage computation
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + gamma * running_return
            returns.insert(0, running_return)
        
        advantages = np.array(returns) - np.mean(returns)
        advantages = (advantages / (np.std(advantages) + 1e-8)).tolist()
        
        state["advantages"] = advantages
        state["returns"] = returns
        return state
    
    def _real_compute_loss(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Computing loss")
        # In practice, would compute actual PPO loss
        import numpy as np
        
        advantages = np.array(state.get("advantages", [0]))
        
        state["policy_loss"] = float(-np.mean(advantages) * 0.01)
        state["value_loss"] = float(np.mean(advantages**2))
        state["entropy_loss"] = 0.01
        state["total_loss"] = state["policy_loss"] + state["value_loss"] - state["entropy_loss"]
        return state
    
    def _real_update_parameters(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("  [REAL] Updating parameters")
        # In practice, would perform gradient descent
        import numpy as np
        
        state["grad_norm"] = float(np.random.randn() * 0.1 + 0.5)
        state["parameters_updated"] = True
        state["step_count"] = state.get("step_count", 0) + 1
        return state
    
    # Validation functions
    
    def _validate_pipeline_structure(self, state: Dict[str, Any]) -> bool:
        """Validate basic pipeline structure."""
        required_keys = ["debug_level", "components", "env", "network"]
        return all(key in state for key in required_keys)
    
    def _validate_env_interaction(self, state: Dict[str, Any]) -> bool:
        """Validate environment is set up correctly."""
        return (state.get("env") == "RealEnv" and 
                "observation_space" in state and
                "action_space" in state)
    
    def _validate_network_forward(self, state: Dict[str, Any]) -> bool:
        """Validate network initialization."""
        return (state.get("network") == "RealNetwork" and
                state.get("num_parameters", 0) > 0)
    
    def _validate_rollouts(self, state: Dict[str, Any]) -> bool:
        """Validate rollout collection."""
        if "rollouts" not in state:
            return False
        
        rollouts = state["rollouts"]
        return (len(rollouts.get("observations", [])) > 0 and
                len(rollouts.get("actions", [])) > 0 and
                len(rollouts.get("rewards", [])) > 0)
    
    def _validate_advantages(self, state: Dict[str, Any]) -> bool:
        """Validate advantage computation."""
        return ("advantages" in state and 
                "returns" in state and
                len(state["advantages"]) > 0)
    
    def _validate_loss(self, state: Dict[str, Any]) -> bool:
        """Validate loss computation."""
        return all(key in state for key in 
                  ["policy_loss", "value_loss", "total_loss"])
    
    def _validate_full_pipeline(self, state: Dict[str, Any]) -> bool:
        """Validate full pipeline execution."""
        return (state.get("parameters_updated") == True and
                state.get("grad_norm", 0) > 0)
    
    def run_progressive_debug(self, max_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Run progressive debugging through all levels.
        
        Args:
            max_level: Maximum level to debug to (None for all)
            
        Returns:
            Results from each level
        """
        print("\n" + "="*70)
        print("PROGRESSIVE DEBUGGING")
        print("="*70)
        
        levels_to_run = self.levels[:max_level] if max_level else self.levels
        
        for i, level in enumerate(levels_to_run):
            print(f"\n{'='*60}")
            print(f"Level {i}: {level.name}")
            print(f"Description: {level.description}")
            print(f"{'='*60}")
            
            # Create and run pipeline for this level
            pipeline = self.create_debug_pipeline(level)
            start_time = time.time()
            
            try:
                result = pipeline.run()
                duration = time.time() - start_time
                
                # Check validation
                if result.get("validation_result"):
                    print(f"✓ Level {i} PASSED in {duration:.2f}s")
                    self.results[level.name] = {
                        "success": True,
                        "duration": duration,
                        "result": result
                    }
                else:
                    print(f"✗ Level {i} FAILED validation")
                    self.results[level.name] = {
                        "success": False,
                        "duration": duration,
                        "result": result
                    }
                    
                    # Stop on first failure
                    print(f"\n⚠ Stopping at level {i} due to failure")
                    print("  Debug the issue at this level before proceeding")
                    break
                    
            except Exception as e:
                print(f"✗ Level {i} CRASHED: {e}")
                self.results[level.name] = {
                    "success": False,
                    "error": str(e)
                }
                break
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print debugging summary."""
        print("\n" + "="*70)
        print("DEBUGGING SUMMARY")
        print("="*70)
        
        for level_name, result in self.results.items():
            status = "✓ PASS" if result.get("success") else "✗ FAIL"
            duration = result.get("duration", 0)
            print(f"{level_name:30s} | {status:8s} | {duration:6.2f}s")
        
        # Find where debugging should focus
        failed_levels = [k for k, v in self.results.items() if not v.get("success")]
        if failed_levels:
            print(f"\n⚠ First failure at: {failed_levels[0]}")
            print("  Focus debugging efforts here")
        else:
            print("\n✓ All levels passed! Pipeline is working correctly")


# =============================================================================
# QUICK DEBUG UTILITIES
# =============================================================================

def quick_component_test(component_name: str, component_func: Callable, 
                         test_input: Dict[str, Any]) -> bool:
    """
    Quick test of a single component.
    
    Args:
        component_name: Name of component being tested
        component_func: Function to test
        test_input: Input state for testing
        
    Returns:
        True if component works correctly
    """
    print(f"\nQuick Test: {component_name}")
    print("-" * 40)
    
    try:
        start_time = time.time()
        result = component_func(test_input)
        duration = time.time() - start_time
        
        print(f"✓ Completed in {duration:.3f}s")
        print(f"  Output keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def binary_search_debug(pipeline: Pipeline, test_stages: List[str]) -> str:
    """
    Use binary search to find failing stage.
    
    Args:
        pipeline: Pipeline to debug
        test_stages: List of stage names to test
        
    Returns:
        Name of first failing stage
    """
    print("\nBinary Search Debugging")
    print("-" * 40)
    
    def test_up_to_stage(stage_index: int) -> bool:
        """Test pipeline up to given stage index."""
        try:
            # Create partial pipeline
            partial = Pipeline(name="partial")
            for i in range(stage_index + 1):
                stage = pipeline.stages[i]
                partial.stage(stage.name, stage.func)
            
            # Run and check
            result = partial.run()
            return result is not None
        except:
            return False
    
    # Binary search
    left, right = 0, len(test_stages) - 1
    first_fail = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if test_up_to_stage(mid):
            print(f"  ✓ Stages 0-{mid} work")
            left = mid + 1
        else:
            print(f"  ✗ Stage {mid} or earlier fails")
            first_fail = mid
            right = mid - 1
    
    if first_fail >= 0:
        print(f"\n⚠ First failing stage: {test_stages[first_fail]}")
        return test_stages[first_fail]
    else:
        print("\n✓ All stages work")
        return ""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution demonstrating progressive debugging.
    """
    print("="*70)
    print("PROGRESSIVE DEBUGGING COOKBOOK")
    print("="*70)
    print("\nThis recipe demonstrates systematic debugging of complex")
    print("training pipelines by progressively adding components.\n")
    
    # Create base spec
    spec = create_quick_experiment()
    
    # Initialize debugger
    debugger = ProgressiveDebugger(spec)
    
    # Run progressive debugging
    results = debugger.run_progressive_debug()
    
    # Example: Quick component tests
    print("\n" + "="*70)
    print("QUICK COMPONENT TESTS")
    print("="*70)
    
    test_state = {"seed": 42}
    
    quick_component_test(
        "Environment Setup",
        debugger._real_env_setup,
        test_state
    )
    
    quick_component_test(
        "Network Initialization", 
        debugger._real_network_init,
        test_state
    )
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Start with all components mocked for fast iteration")
    print("2. Progressively add real components one at a time")
    print("3. Validate each level before proceeding")
    print("4. Stop at first failure to focus debugging efforts")
    print("5. Use quick tests for individual component validation")


if __name__ == "__main__":
    main()