#!/bin/bash
# summarize_sweep.sh - Summarize sweep results and find best hyperparameters
# Usage: ./summarize_sweep.sh run=sweep_name

set -e

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./summarize_sweep.sh run=SWEEP_NAME"
    echo "Example: ./summarize_sweep.sh run=rollout_test"
    exit 1
fi

# Extract sweep name
sweep_name=$(echo "$1" | sed 's/run=//')
sweep_dir="train_dir/sweep/$sweep_name"

if [ ! -d "$sweep_dir" ]; then
    echo "❌ Sweep directory not found: $sweep_dir"
    echo "Available sweeps:"
    ls -1 train_dir/sweep/ 2>/dev/null || echo "  No sweeps found"
    exit 1
fi

echo "📊 SWEEP SUMMARY: $sweep_name"
echo "================================================================="

# Count rollouts
rollout_count=$(ls -1 "$sweep_dir/runs/" | grep -E "^${sweep_name}\.r\.[0-9]+$" | wc -l | tr -d ' ')
echo "🔄 Total Rollouts: $rollout_count"

if [ "$rollout_count" -eq 0 ]; then
    echo "❌ No completed rollouts found"
    exit 1
fi

echo ""
echo "🧬 HYPERPARAMETER SUGGESTIONS:"
echo "================================================================="

# Show suggestions for each rollout
for i in $(seq 0 $((rollout_count - 1))); do
    rollout_dir="$sweep_dir/runs/${sweep_name}.r.$i"
    if [ -f "$rollout_dir/protein_suggestion.yaml" ]; then
        echo "--- Rollout $i ---"

        # Extract suggestion UUID
        uuid=$(grep "suggestion_uuid" "$rollout_dir/protein_suggestion.yaml" | cut -d'"' -f4)
        echo "UUID: $uuid"

        # Extract hyperparameters (excluding UUID)
        echo "Parameters:"
        grep -v "suggestion_uuid" "$rollout_dir/protein_suggestion.yaml" | sed 's/^/  /'

        echo ""
    else
        echo "--- Rollout $i ---"
        echo "❌ No suggestion file found"
        echo ""
    fi
done

echo "🏆 TRAINED POLICIES:"
echo "================================================================="

# Show WandB links and policy names
for i in $(seq 0 $((rollout_count - 1))); do
    rollout_dir="$sweep_dir/runs/${sweep_name}.r.$i"
    if [ -f "$rollout_dir/protein_suggestion.yaml" ]; then
        uuid=$(grep "suggestion_uuid" "$rollout_dir/protein_suggestion.yaml" | cut -d'"' -f4)
        echo "Rollout $i:"
        echo "  WandB: https://wandb.ai/metta-research/metta/runs/$uuid"
        echo "  Policy: ${sweep_name}.r.$i:v0"
        echo ""
    fi
done

echo "📈 EVALUATION RESULTS:"
echo "================================================================="

# Look for evaluation results
eval_found=false
for i in $(seq 0 $((rollout_count - 1))); do
    rollout_dir="$sweep_dir/runs/${sweep_name}.r.$i"

    # Check for various result files
    if [ -f "$rollout_dir/eval_results.json" ]; then
        echo "--- Rollout $i Results ---"
        cat "$rollout_dir/eval_results.json"
        echo ""
        eval_found=true
    elif [ -f "$rollout_dir/results.yaml" ]; then
        echo "--- Rollout $i Results ---"
        cat "$rollout_dir/results.yaml"
        echo ""
        eval_found=true
    fi
done

if [ "$eval_found" = false ]; then
    echo "⚠️  No evaluation results found locally"
    echo "💡 Check WandB runs above for training metrics and scores"
fi

echo ""
echo "🎯 NEXT STEPS:"
echo "================================================================="
echo "1. 📊 View sweep dashboard: https://wandb.ai/metta-research/metta/sweeps"
echo "2. 🏆 Identify best performing run from WandB metrics"
echo "3. 🚀 Use best policy:"
echo "   ./devops/eval.sh policy_uri=wandb://run/BEST_UUID +sim=navigation"
echo "4. 💾 Download best policy:"
echo "   ./tools/download_policy.py --run-id BEST_UUID --output ./best_policy.pt"

echo ""
echo "📋 SWEEP CONFIGURATION:"
echo "================================================================="

# Show sweep config if available
if [ -f "$sweep_dir/runs/${sweep_name}.r.0/config.yaml" ]; then
    echo "Rollout count limit:"
    grep -A1 -B1 "rollout_count" "$sweep_dir/runs/${sweep_name}.r.0/config.yaml" 2>/dev/null || echo "  No rollout_count found (infinite sweep)"

    echo ""
    echo "Parameter space:"
    grep -A20 "parameters:" "$sweep_dir/runs/${sweep_name}.r.0/config.yaml" 2>/dev/null | head -20 || echo "  Config not found"
else
    echo "⚠️  Configuration not found"
fi

echo ""
echo "✅ Summary complete for sweep: $sweep_name"
