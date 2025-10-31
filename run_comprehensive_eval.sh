#!/bin/bash
set -e

echo "=========================================="
echo "Comprehensive Scripted Agent Evaluation"
echo "=========================================="
echo ""
echo "Running evaluation across:"
echo "  - 4 experiments (EXP1, EXP2, OxygenBottleneck, GermaniumRush)"
echo "  - 2 difficulties (easy, medium)"
echo "  - 2 hyperparams (balanced, explorer_long)"
echo "  - 3 clip modes (none, carbon, oxygen)"
echo "  - 2 clip rates (0.0, 0.25)"
echo "  - 3 agent counts (1, 2, 4)"
echo ""
echo "Total: ~144 tests"
echo ""

uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py \
    --output comprehensive_eval_results.json \
    full \
    --experiments EXP1 EXP2 OxygenBottleneck GermaniumRush \
    --difficulties easy medium \
    --hyperparams balanced explorer_long \
    --clip-modes none carbon oxygen \
    --clip-rates 0.0 0.25 \
    --cogs 1 2 4 \
    --steps 1000

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: comprehensive_eval_results.json"
echo "=========================================="
