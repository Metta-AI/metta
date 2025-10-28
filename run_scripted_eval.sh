#!/bin/bash
# Full evaluation of scripted agent across all missions

cd /Users/daphnedemekas/Desktop/metta/packages/cogames

echo "=== TRAINING FACILITIES ==="
for mission in training_facility.1 training_facility.2 training_facility.3 training_facility.open_1; do
    echo ""
    echo "Testing $mission..."
    uv run cogames evaluate -m $mission -p scripted -e 3 2>&1 | grep -A 5 "Average Reward"
done

echo ""
echo "=== MACHINA EVAL MISSIONS ==="
for mission in machina_eval.oxygen_bottleneck machina_eval.germanium_rush machina_eval.silicon_workbench machina_eval.carbon_desert machina_eval.single_use_world machina_eval.slow_oxygen machina_eval.high_regen_sprint machina_eval.sparse_balanced machina_eval.germanium_clutch; do
    echo ""
    echo "Testing $mission..."
    uv run cogames evaluate -m $mission -p scripted -e 3 2>&1 | grep -A 5 "Average Reward"
done

echo ""
echo "=== EXPLORATION EXPERIMENTS ==="
for mission in machina_eval.exp01 machina_eval.exp02; do
    echo ""
    echo "Testing $mission..."
    uv run cogames evaluate -m $mission -p scripted -e 3 2>&1 | grep -A 5 "Average Reward"
done

