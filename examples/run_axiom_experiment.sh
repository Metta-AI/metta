#!/bin/bash
# Example: Running Axiom experiments using the run_experiment.py entrypoint

echo "=== Axiom Experiment Runner Examples ==="
echo

# Example 1: Quick test experiment (training + evaluation)
echo "1. Quick test experiment:"
echo "   ./tools/run_experiment.py metta.sweep.axiom.train_and_eval.create_experiment \\"
echo "       --args name=quick_test total_timesteps=10000 num_agents=4 num_eval_episodes=5"
echo

# Example 2: Eval-only experiment (no training)
echo "2. Eval-only experiment with existing policy:"
echo "   ./tools/run_experiment.py metta.sweep.axiom.train_and_eval.create_experiment \\"
echo "       --args name=eval_only policy_path=file://./checkpoints/trained_policy num_agents=24 num_eval_episodes=100"
echo

# Example 3: Full training experiment
echo "3. Full training experiment:"
echo "   ./tools/run_experiment.py metta.sweep.axiom.train_and_eval.create_experiment \\"
echo "       --args name=full_training total_timesteps=1000000 num_agents=24 \\"
echo "       rollout_workers=8 batch_size=512 minibatch_size=64 \\"
echo "       num_eval_episodes=100 enable_combat=true"
echo

# Example 4: Dry run to see experiment configuration
echo "4. Dry run (see config without running):"
echo "   ./tools/run_experiment.py metta.sweep.axiom.train_and_eval.create_experiment \\"
echo "       --args name=test total_timesteps=100 --dry-run"
echo

# Example 5: Running with custom tag
echo "5. Running with custom tag:"
echo "   ./tools/run_experiment.py metta.sweep.axiom.train_and_eval.create_experiment \\"
echo "       --args name=my_exp total_timesteps=10000 --tag v1_baseline"
echo

echo "=== Key Differences from run.py ==="
echo "- run.py: Executes Tool classes directly (TrainTool, SimTool)"
echo "- run_experiment.py: Executes AxiomExperiment classes with:"
echo "  * Spec-driven configuration"
echo "  * Pipeline composition"
echo "  * Manifest generation"
echo "  * Variation points for A/B testing"
echo "  * Built-in reproducibility controls"
echo

echo "=== Creating Custom Experiments ==="
echo "1. Define an ExperimentSpec subclass with your configuration"
echo "2. Create an AxiomExperiment subclass that uses the spec"
echo "3. Implement a factory function that returns the experiment"
echo "4. Run with: ./tools/run_experiment.py path.to.your.factory"