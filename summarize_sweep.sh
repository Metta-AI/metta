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

# Function to find the original sweep config file
find_sweep_config() {
    local sweep_name="$1"

    # Try to find the original config file used for this sweep
    # Look for sweep_params in the first run's config to identify the source
    if [ -f "$sweep_dir/runs/${sweep_name}.r.0/config.yaml" ]; then
        sweep_params=$(grep "sweep_params:" "$sweep_dir/runs/${sweep_name}.r.0/config.yaml" | cut -d' ' -f2)
        if [ -n "$sweep_params" ]; then
            # Convert sweep/protein_working to configs/sweep/protein_working.yaml
            config_file="configs/${sweep_params}.yaml"
            if [ -f "$config_file" ]; then
                echo "$config_file"
                return 0
            fi
        fi
    fi

    # Fallback: try common config file patterns
    for config_file in "configs/sweep/${sweep_name}.yaml" "configs/sweep/protein_${sweep_name}.yaml"; do
        if [ -f "$config_file" ]; then
            echo "$config_file"
            return 0
        fi
    done

    return 1
}

# Try to find and display the original sweep configuration
config_file=$(find_sweep_config "$sweep_name")
if [ -n "$config_file" ]; then
    echo "Configuration source: $config_file"
    echo ""

    # Extract rollout count
    echo "Rollout count limit:"
    rollout_limit=$(grep "rollout_count:" "$config_file" | head -1 | sed 's/.*rollout_count: *\([0-9]*\).*/\1/')
    if [ -n "$rollout_limit" ]; then
        echo "  $rollout_limit (configured limit)"
    else
        echo "  No rollout_count found (infinite sweep)"
    fi

    # Extract num_samples
    num_samples=$(grep "num_samples:" "$config_file" | head -1 | sed 's/.*num_samples: *\([0-9]*\).*/\1/')
    if [ -n "$num_samples" ]; then
        echo "  $num_samples samples per rollout"
    fi

    echo ""
    echo "Parameter space:"

    # Extract parameters section for Protein format
    if grep -q "sweep:" "$config_file"; then
        # Protein format - extract parameters under sweep.parameters
        awk '/sweep:/{flag=1; next} flag && /^[[:space:]]*parameters:/{param_flag=1; next} param_flag && /^[[:space:]]*[a-zA-Z]/ && !/^[[:space:]]*#/{print "  " $0} param_flag && /^[^[:space:]]/ && !/^[[:space:]]*#/{exit}' "$config_file" | head -20
    else
        # Legacy CARBS format - extract trainer parameters
        awk '/^trainer:/{flag=1; next} flag && /^[[:space:]]*[a-zA-Z]/{print "  " $0} flag && /^[^[:space:]]/{exit}' "$config_file" | head -10
    fi

    echo ""
    echo "Optimizer settings:"

    # Extract optimizer metadata
    if grep -q "metric:" "$config_file"; then
        metric=$(grep "metric:" "$config_file" | head -1 | sed 's/.*metric: *\([a-zA-Z_]*\).*/\1/')
        goal=$(grep "goal:" "$config_file" | head -1 | sed 's/.*goal: *\([a-zA-Z_]*\).*/\1/')
        echo "  Metric: $metric"
        echo "  Goal: $goal"
    fi

else
    echo "⚠️  Original configuration file not found"
    echo "Checked locations:"
    echo "  - configs/sweep/${sweep_name}.yaml"
    echo "  - configs/sweep/protein_${sweep_name}.yaml"
    echo "  - Sweep params from run config"
fi

echo ""
echo "✅ Summary complete for sweep: $sweep_name"
