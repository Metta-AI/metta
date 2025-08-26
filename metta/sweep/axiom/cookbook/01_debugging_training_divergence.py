#!/usr/bin/env python3
"""
Recipe: Debugging Training Divergence
=====================================

Problem: Your PPO implementation achieves different scores than a reference
implementation, and you need to identify which component is causing the divergence.

Solution: Use Axiom's override capability to systematically replace components
with known-good implementations until you isolate the issue.

This recipe demonstrates:
- Creating a debuggable training pipeline
- Overriding specific components
- Comparing results to isolate issues
- Best practices for debugging RL training
"""

import os
import sys
from typing import Dict, Any, Callable
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from metta.sweep.axiom.core import Pipeline
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment
)
from metta.sweep.axiom.experiment_spec import AxiomControls


# =============================================================================
# REFERENCE IMPLEMENTATIONS (Known to be correct)
# =============================================================================

def reference_gae_advantages(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reference implementation of GAE (Generalized Advantage Estimation).
    This is your known-good implementation from a working codebase.
    """
    print("  [REF] Computing advantages using reference GAE implementation")
    
    # In real scenario, this would compute actual GAE
    # For demo, we'll return mock values
    state["advantages"] = "reference_gae_advantages"
    state["returns"] = "reference_gae_returns"
    state["advantage_mean"] = 0.0
    state["advantage_std"] = 1.0
    
    return state


def reference_ppo_loss(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reference implementation of PPO loss computation.
    """
    print("  [REF] Computing loss using reference PPO implementation")
    
    # In real scenario, this would compute actual PPO loss
    state["policy_loss"] = -0.01
    state["value_loss"] = 0.5
    state["entropy_loss"] = 0.01
    state["total_loss"] = state["policy_loss"] + state["value_loss"] - state["entropy_loss"]
    
    return state


def reference_adam_optimizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reference implementation of Adam optimizer step.
    """
    print("  [REF] Applying gradients using reference Adam implementation")
    
    # In real scenario, this would apply actual Adam updates
    state["gradients_applied"] = True
    state["grad_norm"] = 0.5
    state["learning_rate"] = 3e-4
    
    return state


# =============================================================================
# YOUR IMPLEMENTATIONS (Potentially buggy)
# =============================================================================

def your_gae_advantages(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your implementation of GAE that might have issues.
    """
    print("  [YOUR] Computing advantages using your GAE implementation")
    
    # This might have a bug - for demo, we'll return different values
    state["advantages"] = "your_gae_advantages"
    state["returns"] = "your_gae_returns"
    state["advantage_mean"] = 0.1  # Slightly different - could indicate bug!
    state["advantage_std"] = 0.9   # Different std - another potential issue!
    
    return state


def your_ppo_loss(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your implementation of PPO loss computation.
    """
    print("  [YOUR] Computing loss using your PPO implementation")
    
    # Your implementation might have different values
    state["policy_loss"] = -0.02  # Different! Could be the issue
    state["value_loss"] = 0.45
    state["entropy_loss"] = 0.01
    state["total_loss"] = state["policy_loss"] + state["value_loss"] - state["entropy_loss"]
    
    return state


def your_adam_optimizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your implementation of Adam optimizer.
    """
    print("  [YOUR] Applying gradients using your Adam implementation")
    
    state["gradients_applied"] = True
    state["grad_norm"] = 0.8  # Higher grad norm - could indicate issue
    state["learning_rate"] = 3e-4
    
    return state


# =============================================================================
# DEBUGGING WORKFLOW
# =============================================================================

@dataclass
class DebugResult:
    """Results from a debugging run."""
    component_tested: str
    implementation_used: str
    final_reward: float
    divergence_score: float
    state: Dict[str, Any]


class TrainingDivergenceDebugger:
    """
    Helper class to systematically debug training divergence.
    """
    
    def __init__(self, base_spec: TrainAndEvalSpec):
        """
        Initialize debugger with a base experiment spec.
        
        Args:
            base_spec: The experiment configuration to debug
        """
        self.base_spec = base_spec
        self.results: list[DebugResult] = []
        self.baseline_result = None
        
    def create_training_pipeline(self, use_reference: Dict[str, bool]) -> Pipeline:
        """
        Create a training pipeline with configurable component implementations.
        
        Args:
            use_reference: Dict mapping component names to whether to use reference impl
            
        Returns:
            Configured pipeline for training
        """
        pipeline = Pipeline(name="debug_trainer")
        
        # Data collection (not typically the issue)
        pipeline.stage("collect_rollouts", 
                      lambda s: {**s, "rollouts": "collected", "num_samples": 1024})
        
        # Advantage computation - OFTEN THE CULPRIT!
        advantage_func = (reference_gae_advantages if use_reference.get("advantages", False)
                         else your_gae_advantages)
        pipeline.stage("compute_advantages", advantage_func, expose=True)
        
        # Loss computation - ANOTHER COMMON ISSUE
        loss_func = (reference_ppo_loss if use_reference.get("loss", False)
                    else your_ppo_loss)
        pipeline.stage("compute_loss", loss_func, expose=True)
        
        # Optimizer step - SOMETIMES THE ISSUE
        optimizer_func = (reference_adam_optimizer if use_reference.get("optimizer", False)
                         else your_adam_optimizer)
        pipeline.stage("optimizer_step", optimizer_func, expose=True)
        
        # Final metrics
        pipeline.stage("compute_metrics", 
                      lambda s: {**s, "final_reward": self._compute_reward(s)})
        
        return pipeline
    
    def _compute_reward(self, state: Dict[str, Any]) -> float:
        """
        Compute a mock final reward based on state.
        In reality, this would come from evaluation.
        """
        # Mock calculation showing how different components affect final score
        base_reward = 100.0
        
        # Advantages affect reward
        if "reference_gae" in state.get("advantages", ""):
            base_reward += 10
        
        # Loss affects reward  
        if state.get("policy_loss", 0) == -0.01:  # Reference value
            base_reward += 5
            
        # Optimizer affects reward
        if state.get("grad_norm", 0) == 0.5:  # Reference value
            base_reward += 3
            
        return base_reward
    
    def run_baseline(self) -> DebugResult:
        """
        Run with all YOUR implementations to establish baseline.
        """
        print("\n" + "="*60)
        print("RUNNING BASELINE (All YOUR implementations)")
        print("="*60)
        
        pipeline = self.create_training_pipeline(use_reference={})
        result = pipeline.run()
        
        debug_result = DebugResult(
            component_tested="baseline",
            implementation_used="all_yours",
            final_reward=result["final_reward"],
            divergence_score=0.0,  # Baseline
            state=result
        )
        
        self.baseline_result = debug_result
        self.results.append(debug_result)
        
        print(f"\nBaseline reward: {debug_result.final_reward}")
        return debug_result
    
    def test_component(self, component: str, component_path: str, 
                      reference_impl: Callable) -> DebugResult:
        """
        Test a specific component by replacing it with reference implementation.
        
        Args:
            component: Name of component being tested
            component_path: Pipeline path to component (e.g., "compute_advantages")
            reference_impl: Reference implementation to use
            
        Returns:
            Debug results
        """
        print("\n" + "="*60)
        print(f"TESTING: {component}")
        print("="*60)
        
        # Create pipeline with YOUR implementations
        pipeline = self.create_training_pipeline(use_reference={})
        
        # Override just this component with reference
        pipeline.override(component_path, reference_impl)
        
        # Run and collect results
        result = pipeline.run()
        
        debug_result = DebugResult(
            component_tested=component,
            implementation_used=f"reference_{component}",
            final_reward=result["final_reward"],
            divergence_score=abs(result["final_reward"] - self.baseline_result.final_reward),
            state=result
        )
        
        self.results.append(debug_result)
        
        print(f"\nReward with reference {component}: {debug_result.final_reward}")
        print(f"Divergence from baseline: {debug_result.divergence_score}")
        
        return debug_result
    
    def run_full_debugging_suite(self):
        """
        Run complete debugging suite testing each component.
        """
        # Run baseline first
        self.run_baseline()
        
        # Test each component
        components_to_test = [
            ("advantages", "compute_advantages", reference_gae_advantages),
            ("loss", "compute_loss", reference_ppo_loss),
            ("optimizer", "optimizer_step", reference_adam_optimizer),
        ]
        
        print("\n" + "#"*60)
        print("# SYSTEMATIC COMPONENT TESTING")
        print("#"*60)
        
        for name, path, ref_impl in components_to_test:
            self.test_component(name, path, ref_impl)
        
        # Analyze results
        self.analyze_results()
        
    def analyze_results(self):
        """
        Analyze debugging results to identify the problematic component.
        """
        print("\n" + "#"*60)
        print("# ANALYSIS RESULTS")
        print("#"*60)
        
        # Sort by divergence score
        sorted_results = sorted(self.results[1:], key=lambda r: r.divergence_score, reverse=True)
        
        print("\nComponent Impact Analysis:")
        print("-" * 40)
        
        for result in sorted_results:
            impact = "HIGH" if result.divergence_score > 5 else "MEDIUM" if result.divergence_score > 2 else "LOW"
            print(f"{result.component_tested:12s} | Divergence: {result.divergence_score:6.2f} | Impact: {impact}")
        
        # Identify likely culprit
        if sorted_results:
            culprit = sorted_results[0]
            if culprit.divergence_score > 5:
                print(f"\n🔍 LIKELY CULPRIT: {culprit.component_tested}")
                print(f"   Replacing this component with reference implementation")
                print(f"   changed reward by {culprit.divergence_score:.2f} points")
                
                # Provide specific debugging advice
                if culprit.component_tested == "advantages":
                    print("\n   Debugging advice:")
                    print("   - Check your GAE lambda and gamma values")
                    print("   - Verify value function predictions")
                    print("   - Ensure correct bootstrapping at episode boundaries")
                elif culprit.component_tested == "loss":
                    print("\n   Debugging advice:")
                    print("   - Verify PPO clip coefficient is applied correctly")
                    print("   - Check value function coefficient")
                    print("   - Ensure entropy bonus has correct sign")
                elif culprit.component_tested == "optimizer":
                    print("\n   Debugging advice:")
                    print("   - Check learning rate schedule")
                    print("   - Verify gradient clipping threshold")
                    print("   - Ensure correct weight decay settings")


# =============================================================================
# ADVANCED: Using with Real TrainAndEval Experiment
# =============================================================================

def debug_real_experiment():
    """
    Demonstrates debugging with actual TrainAndEvalExperiment.
    """
    print("\n" + "="*70)
    print("ADVANCED: Debugging with TrainAndEvalExperiment")
    print("="*70)
    
    # Create base experiment
    spec = create_quick_experiment()
    experiment = TrainAndEvalExperiment(spec)
    
    # Get the pipeline
    pipeline = experiment._create_pipeline({})
    
    # Create custom training stage with exposed components
    def create_debuggable_train_stage(spec):
        """Create a training stage with exposed components for debugging."""
        
        def train_with_exposed_components(state: Dict[str, Any]) -> Dict[str, Any]:
            # Create sub-pipeline for training components
            train_pipeline = Pipeline(name="trainer")
            
            # Expose all algorithmic components
            train_pipeline.stage("collect_rollouts", 
                               lambda s: {**s, "rollouts": "collected"})
            train_pipeline.stage("compute_advantages", 
                               your_gae_advantages, expose=True)
            train_pipeline.stage("compute_loss", 
                               your_ppo_loss, expose=True)
            train_pipeline.stage("optimizer_step", 
                               your_adam_optimizer, expose=True)
            
            # Run training pipeline
            train_result = train_pipeline.run()
            
            return {
                **state,
                **train_result,
                "training_complete": True,
                "policy_uri": f"file://{spec.run_dir}/policy.pt"
            }
        
        return train_with_exposed_components
    
    # Replace the training stage with our debuggable version
    # (In practice, you'd modify TrainAndEvalExperiment to expose these)
    for i, stage in enumerate(pipeline.stages):
        if stage.name == "train":
            pipeline.stages[i].func = create_debuggable_train_stage(spec)
            break
    
    print("\nPipeline configured for debugging")
    print("Components available for override:")
    print("  - train.compute_advantages")
    print("  - train.compute_loss")
    print("  - train.optimizer_step")
    
    # Now you could systematically test by overriding
    # pipeline.override("train.compute_advantages", reference_gae_advantages)
    # result = experiment.run("test_advantages")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution demonstrating the debugging workflow.
    """
    print("="*70)
    print("TRAINING DIVERGENCE DEBUGGING COOKBOOK")
    print("="*70)
    print("\nScenario: Your PPO implementation gets 20% lower scores than reference.")
    print("Goal: Identify which component is causing the divergence.\n")
    
    # Create a simple experiment spec
    spec = TrainAndEvalSpec(
        name="debug_divergence",
        controls=AxiomControls(seed=42, enforce_determinism=True)
    )
    
    # Initialize debugger
    debugger = TrainingDivergenceDebugger(spec)
    
    # Run full debugging suite
    debugger.run_full_debugging_suite()
    
    # Advanced example (commented out to keep output clean)
    # debug_real_experiment()
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Test components in isolation by overriding with reference implementations")
    print("2. The component with highest divergence when replaced is likely the issue")
    print("3. Always test in order: advantages → loss → optimizer → other")
    print("4. Use deterministic seeds for reproducible debugging")
    print("5. Save specs that reproduce issues for future regression testing")


if __name__ == "__main__":
    main()