#!/bin/bash
set -e

echo "=========================================="
echo "Running Missing Hyperparameter Presets"
echo "=========================================="
echo ""
echo "Testing 3 additional presets:"
echo "  - greedy_conservative"
echo "  - efficiency_heavy"
echo "  - sequential_baseline"
echo ""
echo "Same configuration as before:"
echo "  - 4 experiments (EXP1, EXP2, OxygenBottleneck, GermaniumRush)"
echo "  - 2 difficulties (easy, medium)"
echo "  - 3 clip modes (none, carbon, oxygen) × 2 rates"
echo "  - 3 agent counts (1, 2, 4)"
echo ""
echo "Total: ~300 additional tests"
echo ""

uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py \
    --output additional_presets_results.json \
    full \
    --experiments EXP1 EXP2 OxygenBottleneck GermaniumRush \
    --difficulties easy medium \
    --hyperparams greedy_conservative efficiency_heavy sequential_baseline \
    --clip-modes none carbon oxygen \
    --clip-rates 0.0 0.25 \
    --cogs 1 2 4 \
    --steps 1000

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: additional_presets_results.json"
echo "=========================================="
